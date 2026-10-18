"""Unit checks for depth-sensitivity aggregation and file isolation."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import option_by_dates_sensitivity as study


def make_profiles(n: int) -> list[dict[str, float | int | str]]:
    """Synthetic, probability-consistent profiles; not market results."""
    result = []
    for key in study.SCENARIOS:
        european = key == "european_no_dividend"
        initial_price = (5.0 if european else 5.5)
        previous_total = 0.0
        for k in range(n + 1):
            t = k / n
            # Synthetic scenario: American early exercise at dividend and
            # cash exercise at expiry, European exercise only at expiry.
            total = (0.0 if k < n else 0.4) if european else (
                0.0 if t < 0.5 else (0.2 if k < n else 0.45)
            )
            first = total - previous_total
            previous_total = total
            pv = initial_price if european else initial_price - 0.2 * t
            result.append({
                "scenario": key, "step": k, "time": t,
                "mean_option_value": pv * math.exp(0.02 * t),
                "discounted_mean_option_value": pv,
                "mean_stock_price": 100.0,
                "unconditional_probability_mass": 1.0,
                "exercise_region_probability": first,
                "first_exercise_probability": first,
                "cumulative_first_exercise_probability": total,
                "survival_probability_after_step": 1.0 - total,
            })
    return result


def test_conversion_retains_numbers_and_key() -> None:
    converted = study.convert_row(make_profiles(2)[0])
    assert converted["scenario"] == study.SCENARIOS[0]
    assert converted["step"] == 0
    assert isinstance(converted["discounted_mean_option_value"], float)


def test_non_finite_rejected() -> None:
    record = make_profiles(2)[0]
    record["mean_option_value"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        study.convert_row(record)


def test_wrong_columns_rejected() -> None:
    record = make_profiles(2)[0]
    record.pop("mean_option_value")
    with pytest.raises(ValueError, match="columns"):
        study.convert_row(record)


def test_valid_summary_metrics() -> None:
    summaries = study.summarize_steps(make_profiles(2), 2)
    assert len(summaries) == 3
    cash = summaries[0]
    euro = summaries[-1]
    assert cash["early_exercise_probability"] == pytest.approx(0.2)
    assert cash["first_exercise_at_dividend"] == pytest.approx(0.2)
    assert cash["total_exercise_probability"] == pytest.approx(0.45)
    assert euro["early_exercise_probability"] == 0.0
    assert euro["maximum_martingale_deviation"] == pytest.approx(0.0)


def test_missing_scenario_rejected() -> None:
    rows = [r for r in make_profiles(2) if r["scenario"] != study.SCENARIOS[1]]
    with pytest.raises(ValueError, match="three scenarios"):
        study.summarize_steps(rows, 2)


def test_invalid_dates_rejected() -> None:
    rows = make_profiles(2)
    rows[1]["time"] = 0.7
    with pytest.raises(ValueError, match="dates"):
        study.summarize_steps(rows, 2)


def test_probability_mass_loss_rejected() -> None:
    rows = make_profiles(2)
    rows[0]["unconditional_probability_mass"] = 0.95
    with pytest.raises(ValueError, match="mass not conserved"):
        study.summarize_steps(rows, 2)


def test_stopping_distribution_inconsistency_rejected() -> None:
    rows = make_profiles(2)
    rows[1]["survival_probability_after_step"] = 0.9
    with pytest.raises(ValueError, match="survival/exercise"):
        study.summarize_steps(rows, 2)


def test_european_martingale_rejected_if_broken() -> None:
    rows = make_profiles(2)
    rows[-2]["discounted_mean_option_value"] -= 0.01
    with pytest.raises(ValueError, match="martingale"):
        study.summarize_steps(rows, 2)


def test_american_increasing_discounted_mean_rejected() -> None:
    rows = make_profiles(2)
    rows[1]["discounted_mean_option_value"] += 0.3
    with pytest.raises(ValueError, match="supermartingale"):
        study.summarize_steps(rows, 2)


def test_unmatched_forward_rejected() -> None:
    rows = make_profiles(2)
    rows[-1]["mean_stock_price"] = 99.0
    with pytest.raises(ValueError, match="forwards"):
        study.summarize_steps(rows, 2)


def test_baseline_parameter_validation(tmp_path: Path) -> None:
    path = tmp_path / "summary.txt"
    path.write_text("K=95.0, r=0.02, sigma=0.2, T=1.0, N=100", encoding="utf-8")
    with pytest.raises(ValueError, match="Baseline parameter mismatch"):
        study.check_baseline(path)


def test_read_only_n100_and_no_n100_output(monkeypatch: pytest.MonkeyPatch,
                                              tmp_path: Path) -> None:
    """Mock expensive engine; check output routing, never benchmark numerics."""
    result_root = tmp_path / "results"
    data = result_root / "data"
    figures = result_root / "figures"
    data.mkdir(parents=True)
    baseline_csv = data / "option_by_dates_profiles.csv"
    n100 = make_profiles(100)
    study._write_csv(baseline_csv, n100, study.FIELDS)
    original_bytes = baseline_csv.read_bytes()
    baseline_summary = result_root / "option_by_dates_summary.txt"
    baseline_summary.write_text(
        f"K={study.STRIKE}, r={study.RATE}, sigma={study.VOLATILITY}, "
        f"T={study.MATURITY}, N=100\n"
        f"Cash dividend D={study.DIVIDEND_AMOUNT} at t={study.DIVIDEND_TIME}\n"
        f"S0 cash-dividend scheme={study.SPOT};\n", encoding="utf-8"
    )
    monkeypatch.setattr(study, "RESULTS", result_root)
    monkeypatch.setattr(study, "DATA", data)
    monkeypatch.setattr(study, "FIGURES", figures)
    monkeypatch.setattr(study, "STABILITY_CSV", data / "option_by_dates_sensitivity_summary.csv")
    monkeypatch.setattr(study, "STABILITY_SUMMARY", result_root / "option_by_dates_sensitivity_summary.txt")
    calls: list[int] = []

    def fake_experiment(n: int) -> list[SimpleNamespace]:
        calls.append(n)
        groups = study._group(make_profiles(n))
        return [SimpleNamespace(rows=groups[key]) for key in study.SCENARIOS]

    monkeypatch.setattr(study, "run_experiment", fake_experiment)
    summary = study.run_sensitivity(baseline_csv, baseline_summary)
    assert calls == [50, 200]
    assert len(summary) == 9
    assert baseline_csv.read_bytes() == original_bytes
    assert (data / "option_by_dates_sensitivity_N50.csv").exists()
    assert (data / "option_by_dates_sensitivity_N200.csv").exists()
    assert not (data / "option_by_dates_sensitivity_N100.csv").exists()
    assert (figures / "option_by_dates_sensitivity_prices.png").exists()
    assert (figures / "option_by_dates_sensitivity_early_exercise.png").exists()
    assert (figures / "option_by_dates_sensitivity_dividend_spike.png").exists()
    with (data / "option_by_dates_sensitivity_summary.csv").open(encoding="utf-8") as fp:
        out = list(csv.DictReader(fp))
    assert len(out) == 9
    assert [int(x["nb_steps"]) for x in out] == [50] * 3 + [100] * 3 + [200] * 3
