"""Calendar-sensitive exercise policies, pricing and visualizations.

Consolidated without dropping individual regression cases.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from types import SimpleNamespace
import pytest
from experiments import exercise_profile_step_sensitivity as study
from src.node import Node
from src.option import PutOption
from experiments.option_value_exercise_profiles import (
    DIVIDEND_AMOUNT,
    DIVIDEND_TIME,
    RATE,
    SPOT,
    Scenario,
    exercise_decision,
    make_scenarios,
    profile_from_priced_tree,
    propagate,
    run_experiment,
    transitions,
)


# test_option_by_dates_sensitivity.py (tests preserved)
def _make_profiles(n: int) -> list[dict[str, float | int | str]]:
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
    # Construct deterministic date profiles with coherent survival and exercise mass.
    return result


def test_conversion_retains_numbers_and_key() -> None:
    converted = study.convert_row(_make_profiles(2)[0])
    assert converted["scenario"] == study.SCENARIOS[0]
    assert converted["step"] == 0
    assert isinstance(converted["discounted_mean_option_value"], float)


def test_non_finite_rejected() -> None:
    record = _make_profiles(2)[0]
    record["mean_option_value"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        study.convert_row(record)


# Reject missing required columns before computing summaries.
def test_wrong_columns_rejected() -> None:
    record = _make_profiles(2)[0]
    record.pop("mean_option_value")
    with pytest.raises(ValueError, match="columns"):
        study.convert_row(record)


def test_valid_summary_metrics() -> None:
    summaries = study.summarize_steps(_make_profiles(2), 2)
    assert len(summaries) == 3
    cash = summaries[0]
    euro = summaries[-1]
    assert cash["early_exercise_probability"] == pytest.approx(0.2)
    assert cash["first_exercise_at_dividend"] == pytest.approx(0.2)
    assert cash["total_exercise_probability"] == pytest.approx(0.45)
    assert euro["early_exercise_probability"] == 0.0
    # Summaries must preserve first-exercise, survival and martingale diagnostics.
    assert euro["maximum_martingale_deviation"] == pytest.approx(0.0)


def test_missing_scenario_rejected() -> None:
    rows = [r for r in _make_profiles(2) if r["scenario"] != study.SCENARIOS[1]]
    with pytest.raises(ValueError, match="three scenarios"):
        study.summarize_steps(rows, 2)


def test_invalid_dates_rejected() -> None:
    rows = _make_profiles(2)
    rows[1]["time"] = 0.7
    with pytest.raises(ValueError, match="dates"):
        study.summarize_steps(rows, 2)


def test_probability_mass_loss_rejected() -> None:
    # Reject any profile whose unconditional probability mass is below one.
    rows = _make_profiles(2)
    rows[0]["unconditional_probability_mass"] = 0.95
    with pytest.raises(ValueError, match="mass not conserved"):
        study.summarize_steps(rows, 2)


def test_stopping_distribution_inconsistency_rejected() -> None:
    rows = _make_profiles(2)
    rows[1]["survival_probability_after_step"] = 0.9
    with pytest.raises(ValueError, match="survival/exercise"):
        study.summarize_steps(rows, 2)


def test_european_martingale_rejected_if_broken() -> None:
    rows = _make_profiles(2)
    rows[-2]["discounted_mean_option_value"] -= 0.01
    with pytest.raises(ValueError, match="martingale"):
        # European discounted option values must satisfy the martingale check.
        study.summarize_steps(rows, 2)


def test_american_increasing_discounted_mean_rejected() -> None:
    rows = _make_profiles(2)
    rows[1]["discounted_mean_option_value"] += 0.3
    with pytest.raises(ValueError, match="supermartingale"):
        study.summarize_steps(rows, 2)


def test_unmatched_forward_rejected() -> None:
    rows = _make_profiles(2)
    rows[-1]["mean_stock_price"] = 99.0
    with pytest.raises(ValueError, match="forwards"):
        study.summarize_steps(rows, 2)


def test_baseline_parameter_validation(tmp_path: Path) -> None:
    path = tmp_path / "summary.txt"
    # Preserve the event-date and exercise-policy conventions in this case.
    path.write_text("K=95.0, r=0.02, sigma=0.2, T=1.0, N=100", encoding="utf-8")
    with pytest.raises(ValueError, match="Baseline parameter mismatch"):
        study.check_baseline(path)


def _prepare_readonly_baseline(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, bytes]:
    """Persist the N=100 reference, preserving its original bytes for audit."""
    result_root = tmp_path / "results"
    data = result_root / "data"
    figures = result_root / "figures"
    data.mkdir(parents=True)
    baseline_csv = data / "option_by_dates_profiles.csv"
    n100 = _make_profiles(100)
    study._write_csv(baseline_csv, n100, study.FIELDS)
    original_bytes = baseline_csv.read_bytes()
    baseline_summary = result_root / "option_by_dates_summary.txt"
    baseline_summary.write_text(
        f"K={study.STRIKE}, r={study.RATE}, sigma={study.VOLATILITY}, "
        f"T={study.MATURITY}, N=100\n"
        f"Cash dividend D={study.DIVIDEND_AMOUNT} at t={study.DIVIDEND_TIME}\n"
        f"S0 cash-dividend scheme={study.SPOT};\n", encoding="utf-8"
    )
    return (result_root, data, figures, baseline_csv,
            baseline_summary, original_bytes)



def _assert_sensitivity_evidence(data: Path, figures: Path) -> None:
    """Check N=50/200 exports and the three sensitivity figures."""
    assert (data / "option_by_dates_sensitivity_N50.csv").exists()
    assert (data / "option_by_dates_sensitivity_N200.csv").exists()
    assert not (data / "option_by_dates_sensitivity_N100.csv").exists()
    assert (figures / "option_by_dates_sensitivity_prices.png").exists()
    # Verify the expected numerical or workflow invariant on this case.
    assert (figures / "option_by_dates_sensitivity_early_exercise.png").exists()
    assert (figures / "option_by_dates_sensitivity_dividend_spike.png").exists()
    with (data / "option_by_dates_sensitivity_summary.csv").open(encoding="utf-8") as fp:
        out = list(csv.DictReader(fp))
    assert len(out) == 9
    assert [int(x["nb_steps"]) for x in out] == [50] * 3 + [100] * 3 + [200] * 3


def test_read_only_n100_and_no_n100_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Mock expensive engine; check output routing, never benchmark numerics."""
    (result_root, data, figures, baseline_csv,
     baseline_summary, original_bytes) = _prepare_readonly_baseline(tmp_path)
    # Preserve the event-date and exercise-policy conventions in this case.
    monkeypatch.setattr(study, "RESULTS", result_root)
    monkeypatch.setattr(study, "DATA", data)
    # The baseline N=100 file is an input reference and must never be overwritten.
    monkeypatch.setattr(study, "FIGURES", figures)
    monkeypatch.setattr(study, "STABILITY_CSV", data / "option_by_dates_sensitivity_summary.csv")
    monkeypatch.setattr(study, "STABILITY_SUMMARY", result_root / "option_by_dates_sensitivity_summary.txt")
    calls: list[int] = []

    def _fake_experiment(n: int) -> list[SimpleNamespace]:
        calls.append(n)
        groups = study._group(_make_profiles(n))
        return [SimpleNamespace(rows=groups[key]) for key in study.SCENARIOS]

    monkeypatch.setattr(study, "run_experiment", _fake_experiment)
    summary = study.run_sensitivity(baseline_csv, baseline_summary)
    assert calls == [50, 200]
    assert len(summary) == 9
    assert baseline_csv.read_bytes() == original_bytes
    _assert_sensitivity_evidence(data, figures)


# test_option_value_exercise_by_dates.py (tests preserved)
def _link_recombining_tree(
    root: Node, up: Node, down: Node, upper: Node, middle: Node, lower: Node
) -> None:
    """Share the middle leaf while assigning normalized branch probabilities."""
    root.next_up, root.next_down = up, down
    up.next_up, up.next_down = upper, middle
    down.next_up, down.next_down = middle, lower
    for node in (root, up, down):
        node.up_probability = node.down_probability = 0.5



def _simple_two_step_tree(american: bool) -> tuple[Node, PutOption, Scenario]:
    """Two-step 1/2-1/2 tree, recombining at S=100, with r=0."""
    root = Node(100.0)
    up = Node(110.0)
    down = Node(80.0)
    upup = Node(120.0)
    middle = Node(100.0)
    downdown = Node(80.0)

    _link_recombining_tree(root, up, down, upup, middle, downdown)

    upup.option_value = 0.0
    middle.option_value = 0.0
    downdown.option_value = 20.0
    # The small recombining tree isolates first-exercise and survival accounting.
    up.option_value = 0.0
    down.option_value = 20.0 if american else 10.0
    root.option_value = 10.0 if american else 5.0
    option = PutOption(100.0, is_american=american)
    scenario = Scenario("test", "test", 100.0, american)
    return root, option, scenario


def test_matched_initial_terminal_forward() -> None:
    s = make_scenarios()
    adjusted = SPOT - DIVIDEND_AMOUNT * math.exp(-RATE * DIVIDEND_TIME)
    assert s[0].spot == SPOT
    assert s[1].spot == pytest.approx(adjusted)
    assert s[2].spot == pytest.approx(adjusted)
    div_forward = SPOT * math.exp(RATE) - DIVIDEND_AMOUNT * math.exp(
        RATE * (1.0 - DIVIDEND_TIME)
    )
    # Use matching initial and terminal forwards in discounted value checks.
    assert adjusted * math.exp(RATE) == pytest.approx(div_forward)


def test_recombination_probabilities_merge() -> None:
    root, _, _ = _simple_two_step_tree(american=False)
    first = propagate({root: 1.0})
    second = propagate(first)
    assert sorted(first.values()) == [0.5, 0.5]
    assert sorted(second.values()) == [0.25, 0.25, 0.5]


def test_monomial_pruning_conserves_mass() -> None:
    root = Node(100.0)
    sole = Node(101.0)
    root.next_up = root.next_mid = root.next_down = sole
    root.up_probability = 0.0
    root.mid_probability = 1.0
    # A pruned single branch retains the full probability of its parent.
    root.down_probability = 0.0
    assert transitions(root) == [(sole, 1.0)]
    assert propagate({root: 1.0})[sole] == pytest.approx(1.0)


def test_duplicate_positive_successors_sum_weights() -> None:
    root = Node(100.0)
    child = Node(105.0)
    root.next_up = root.next_mid = root.next_down = child
    root.up_probability = 0.2
    root.mid_probability = 0.5
    root.down_probability = 0.3
    assert propagate({root: 1.0})[child] == pytest.approx(1.0)


def test_reject_invalid_probability_sum() -> None:
    root = Node(100.0)
    # Transition probabilities must form a normalized distribution.
    root.next_up = Node(105.0)
    root.next_down = Node(95.0)
    root.up_probability = 0.3
    root.down_probability = 0.3
    with pytest.raises(AssertionError, match="Outgoing probability mass"):
        transitions(root)


def test_european_discounted_average_is_martingale() -> None:
    root, option, scenario = _simple_two_step_tree(american=False)
    result = profile_from_priced_tree(root, option, 2, 1.0, 0.0, scenario)
    assert [r["discounted_mean_option_value"] for r in result.rows] == pytest.approx(
        [5.0, 5.0, 5.0]
    )
    assert [r["exercise_region_probability"] for r in result.rows] == pytest.approx(
        [0.0, 0.0, 0.25]
    )
    # European discounted mean values must remain stable across the lattice.
    assert result.total_first_exercise_probability == pytest.approx(0.25)
    assert result.expiry_without_exercise_probability == pytest.approx(0.75)


def test_american_discounted_mean_can_decrease() -> None:
    root, option, scenario = _simple_two_step_tree(american=True)
    result = profile_from_priced_tree(root, option, 2, 1.0, 0.0, scenario)
    assert [r["discounted_mean_option_value"] for r in result.rows] == pytest.approx(
        [10.0, 10.0, 5.0]
    )


def test_exercise_region_not_stopping_probability() -> None:
    root, option, scenario = _simple_two_step_tree(american=True)
    result = profile_from_priced_tree(root, option, 2, 1.0, 0.0, scenario)
    assert result.rows[1]["exercise_region_probability"] == pytest.approx(0.5)
    assert result.rows[2]["exercise_region_probability"] == pytest.approx(0.25)
    # Being in an exercise region is distinct from exercising for the first time.
    assert result.rows[1]["first_exercise_probability"] == pytest.approx(0.5)
    assert result.rows[2]["first_exercise_probability"] == pytest.approx(0.0)
    assert result.total_first_exercise_probability == pytest.approx(0.5)
    assert result.expiry_without_exercise_probability == pytest.approx(0.5)


def test_european_never_exercises_early() -> None:
    root, option, _ = _simple_two_step_tree(american=False)
    low = root.next_down
    assert low is not None
    assert exercise_decision(low, option, dt=0.5, rate=0, is_maturity=False) is False
    assert exercise_decision(low, option, dt=0.5, rate=0, is_maturity=True) is True


def test_american_strict_exercise_over_continuation() -> None:
    root, option, _ = _simple_two_step_tree(american=True)
    low = root.next_down
    assert low is not None
    assert exercise_decision(low, option, dt=0.5, rate=0, is_maturity=False) is True


# Preserve the event-date and exercise-policy conventions in this case.
def test_experiment_integration_small_tree() -> None:
    """Real src/ integration, deliberately light for the everyday test suite."""
    results = run_experiment(nb_steps=20)
    assert len(results) == 3
    for result in results:
        assert len(result.rows) == 21
        assert math.isclose(
            result.total_first_exercise_probability
            + result.expiry_without_exercise_probability,
            1.0, abs_tol=2e-10,
        )
    assert results[1].initial_price + 1e-9 >= results[2].initial_price
