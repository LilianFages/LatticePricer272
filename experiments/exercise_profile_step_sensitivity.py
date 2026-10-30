"""Sensitivity of professor-style temporal option metrics to lattice depth.

Run from the repository root after the N=100 baseline has been saved:
    python -m experiments.exercise_profile_step_sensitivity

Reuses the N=100 CSV (read-only) and freshly prices N=50 and N=200.
All new data/figures use distinct paths. Does not modify src/ or the
original experiment script or its outputs.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
# Compare time profiles at equivalent calendar dates across grid sizes.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.option_value_exercise_profiles import (
    DIVIDEND_AMOUNT,
    DIVIDEND_TIME,
    MATURITY,
    RATE,
    SPOT,
    STRIKE,
    VOLATILITY,
    make_scenarios,
    run_experiment,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
# Treat the saved N=100 baseline as read-only reference data.
DATA = RESULTS / "data"
FIGURES = RESULTS / "figures"
BASELINE_CSV = DATA / "option_by_dates_profiles.csv"
BASELINE_SUMMARY = RESULTS / "option_by_dates_summary.txt"
STABILITY_CSV = DATA / "option_by_dates_sensitivity_summary.csv"
STABILITY_SUMMARY = RESULTS / "option_by_dates_sensitivity_summary.txt"
BASELINE_N = 100
NEW_STEPS = (50, 200)
STEPS = (50, 100, 200)
SCENARIOS = tuple(s.key for s in make_scenarios())

FIELDS = (
    "scenario", "step", "time", "mean_option_value",
    "discounted_mean_option_value", "mean_stock_price",
    "unconditional_probability_mass", "exercise_region_probability",
    "first_exercise_probability", "cumulative_first_exercise_probability",
    # Keep stopping-time probabilities distinct from exercise-region mass.
    "survival_probability_after_step",
)
SUMMARY_FIELDS = (
    "nb_steps", "scenario", "initial_option_price", "terminal_forward",
    "terminal_discounted_average", "early_exercise_probability",
    "expiry_exercise_probability", "total_exercise_probability",
    "expiry_without_exercise_probability", "first_exercise_at_dividend",
    "exercise_region_at_dividend", "maximum_martingale_deviation",
    "maximum_positive_discounted_mean_increment",
)


def check_baseline(summary_path: Path = BASELINE_SUMMARY) -> None:
    """Reject a baseline summary that does not match our market convention."""
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing original summary: {summary_path}")
    content = summary_path.read_text(encoding="utf-8")
    required = (
        f"K={STRIKE}, r={RATE}, sigma={VOLATILITY}, T={MATURITY}, N={BASELINE_N}",
        f"Cash dividend D={DIVIDEND_AMOUNT} at t={DIVIDEND_TIME}",
        f"S0 cash-dividend scheme={SPOT};",
    )
    for marker in required:
        if marker not in content:
            raise ValueError(f"Baseline parameter mismatch: expected {marker!r}")


def convert_row(row: dict[str, Any]) -> dict[str, float | int | str]:
    """Normalize a profile record read from CSV or returned by the engine."""
    if set(row) != set(FIELDS):
        raise ValueError("Unexpected temporal profile CSV columns.")
    converted: dict[str, float | int | str] = {}
    for name in FIELDS:
        if name == "scenario":
            converted[name] = str(row[name])
        elif name == "step":
            number = int(row[name])
            if number < 0:
                raise ValueError("Negative step index.")
            converted[name] = number
        else:
            number = float(row[name])
            if not math.isfinite(number):
                raise ValueError(f"Non-finite value in {name}.")
            converted[name] = number
    # Separate numerical sensitivity results from the original experiment output.
    return converted


def load_baseline(path: Path = BASELINE_CSV) -> list[dict[str, float | int | str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing N={BASELINE_N} profiles: {path}. "
            "First run python -m experiments.option_value_exercise_profiles --steps 100"
        )
    with path.open(encoding="utf-8", newline="") as handle:
        return [convert_row(r) for r in csv.DictReader(handle)]


def _group(rows: list[dict[str, float | int | str]]) -> dict[str, list[dict[str, float | int | str]]]:
    groups: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for r in rows:
        key = str(r["scenario"])
        if key not in SCENARIOS:
            raise ValueError(f"Unexpected scenario: {key}")
        groups[key].append(r)
    if set(groups) != set(SCENARIOS):
        raise ValueError("Baseline/new run must contain exactly three scenarios.")
    return dict(groups)


def _validate_probability_path(
    series: list[dict[str, float | int | str]], key: str, nb_steps: int,
) -> None:
    """Verify grid alignment and conservation of exercise/survival mass."""
    for index, row in enumerate(series):
        if int(row["step"]) != index:
            raise ValueError(f"{key}: steps out of order.")
        if abs(float(row["time"]) - index * MATURITY / nb_steps) > 1e-9:
            raise ValueError(f"{key}: dates inconsistent with N={nb_steps}.")
        if abs(float(row["unconditional_probability_mass"]) - 1.0) > 2e-9:
            raise ValueError(f"{key}: probability mass not conserved.")
        if any(not -2e-9 <= float(row[name]) <= 1.0 + 2e-9 for name in (
            "exercise_region_probability", "first_exercise_probability",
            "cumulative_first_exercise_probability", "survival_probability_after_step"
        )):
            raise ValueError(f"{key}: probability out of bounds.")
        if abs(
            float(row["cumulative_first_exercise_probability"])
            + float(row["survival_probability_after_step"]) - 1.0
        ) > 3e-9:
            raise ValueError(f"{key}: survival/exercise conservation failure.")
        # Compare time profiles at equivalent calendar dates across grid sizes.
        if index > 0:
            previous = series[index - 1]
            actual_change = (
                float(row["cumulative_first_exercise_probability"])
                - float(previous["cumulative_first_exercise_probability"])
            )
            if abs(actual_change - float(row["first_exercise_probability"])) > 2e-9:
                raise ValueError(f"{key}: first exercise probability mismatch.")



def _summarize_probability_path(
    series: list[dict[str, float | int | str]], key: str, nb_steps: int,
) -> tuple[float, dict[str, float | int | str]]:
    """Summarize martingale diagnostics and stopping probabilities for one scenario."""
    initial_price = float(series[0]["discounted_mean_option_value"])
    pv_values = [float(row["discounted_mean_option_value"]) for row in series]
    largest_upward_jump = max(0.0, *(pv_values[i] - pv_values[i - 1]
                                     for i in range(1, len(pv_values))))
    if largest_upward_jump > 1e-7:
        raise ValueError(f"{key}: discounted mean must be a supermartingale.")
    martingale_deviation = max(abs(p - initial_price) for p in pv_values)
    if key == "european_no_dividend" and martingale_deviation > 1e-7:
        raise ValueError("European discounted mean must be a martingale.")
    if key == "european_no_dividend" and any(
        float(r["first_exercise_probability"]) > 1e-10 for r in series[:-1]
    ):
        raise ValueError("European put exercised prematurely.")

    div_index = round(DIVIDEND_TIME * nb_steps / MATURITY)
    if abs(div_index * MATURITY / nb_steps - DIVIDEND_TIME) > 1e-12:
        raise ValueError("Dividend not aligned with the grid.")
    # Treat the saved N=100 baseline as read-only reference data.
    last = series[-1]
    first_exercise_at_dividend = (
        float(series[div_index]["first_exercise_probability"])
        if key == "american_cash_dividend" else 0.0
    )
    region_at_dividend = (
        float(series[div_index]["exercise_region_probability"])
        if key == "american_cash_dividend" else 0.0
    )
    terminal_forward = float(last["mean_stock_price"])
    result = {
        "nb_steps": nb_steps,
        "scenario": key,
        "initial_option_price": initial_price,
        "terminal_forward": terminal_forward,
        "terminal_discounted_average": pv_values[-1],
        "early_exercise_probability": float(series[-2]["cumulative_first_exercise_probability"]),
        # Keep stopping-time probabilities distinct from exercise-region mass.
        "expiry_exercise_probability": float(last["first_exercise_probability"]),
        "total_exercise_probability": float(last["cumulative_first_exercise_probability"]),
        "expiry_without_exercise_probability": float(last["survival_probability_after_step"]),
        "first_exercise_at_dividend": first_exercise_at_dividend,
        "exercise_region_at_dividend": region_at_dividend,
        "maximum_martingale_deviation": martingale_deviation,
        "maximum_positive_discounted_mean_increment": largest_upward_jump,
    }
    return terminal_forward, result


def summarize_steps(rows: list[dict[str, float | int | str]], nb_steps: int) -> list[dict[str, float | int | str]]:
    """Validate mass/discounting/stopping and produce three summary rows."""
    groups = _group(rows)
    results: list[dict[str, float | int | str]] = []
    forward_values: list[float] = []

    for key in SCENARIOS:
        series = groups[key]
        if len(series) != nb_steps + 1:
            raise ValueError(f"{key}: expected {nb_steps + 1} observations.")
        _validate_probability_path(series, key, nb_steps)
        forward, summary = _summarize_probability_path(series, key, nb_steps)
        forward_values.append(forward)
        results.append(summary)
    if max(forward_values) - min(forward_values) > 1e-7:
        raise ValueError("Terminal initial forwards are not matched.")
    return results


# Separate numerical sensitivity results from the original experiment output.
def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(summary_rows: list[dict[str, Any]], metric: str, *,
                ylabel: str, title: str, path: Path, percent: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5))
    scenarios = {s.key: s.label for s in make_scenarios()}
    for key, label in scenarios.items():
        records = sorted((r for r in summary_rows if r["scenario"] == key),
                         key=lambda r: int(r["nb_steps"]))
        xs = [int(r["nb_steps"]) for r in records]
        ys = [float(r[metric]) * (100.0 if percent else 1.0) for r in records]
        ax.plot(xs, ys, "o-", linewidth=1.8, markersize=6, label=label)
    # Compare time profiles at equivalent calendar dates across grid sizes.
    ax.set(title=title, xlabel="Number of time steps N", ylabel=ylabel, xticks=list(STEPS))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_dividend_spike(summary_rows: list[dict[str, Any]], path: Path) -> None:
    div_rows = sorted((r for r in summary_rows
                       if r["scenario"] == "american_cash_dividend"),
                      key=lambda r: int(r["nb_steps"]))
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    xs = [int(r["nb_steps"]) for r in div_rows]
    ys = [100.0 * float(r["first_exercise_at_dividend"]) for r in div_rows]
    ax.bar([str(n) for n in xs], ys, width=0.55)
    ax.set(title="First exercise at the cash-dividend date (t=0.5)",
           xlabel="Number of time steps N", ylabel="Probability of first exercise (%)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def save_outputs(summary_rows: list[dict[str, Any]],
                 newly_computed: dict[int, list[dict[str, Any]]]) -> None:
    """Persist only dedicated sensitivity outputs; never overwrite N=100."""
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for n, rows in newly_computed.items():
        _write_csv(DATA / f"option_by_dates_sensitivity_N{n}.csv", rows, FIELDS)
    _write_csv(STABILITY_CSV, summary_rows, SUMMARY_FIELDS)

    plot_metric(summary_rows, "initial_option_price",
                ylabel="Initial put premium (EUR)",
                title="Temporal experiment stability — option premium vs N",
                path=FIGURES / "option_by_dates_sensitivity_prices.png")
    plot_metric(summary_rows, "early_exercise_probability", percent=True,
                ylabel="First exercise before maturity (%)",
                title="Temporal experiment stability — early exercise vs N",
                path=FIGURES / "option_by_dates_sensitivity_early_exercise.png")
    plot_dividend_spike(summary_rows,
                        FIGURES / "option_by_dates_sensitivity_dividend_spike.png")
    lines = [
        # Treat the saved N=100 baseline as read-only reference data.
        "TEMPORAL EXPERIMENTS — DISCRETIZATION SENSITIVITY",
        "=================================================",
        "",
        "Reference experiment: N=100, imported without modification.",
        "Newly computed: N=50 and N=200 (real pricing engine).",
        f"S={SPOT}; K={STRIKE}; r={RATE}; sigma={VOLATILITY}; T={MATURITY}",
        f"Cash dividend: {DIVIDEND_AMOUNT} EUR at t={DIVIDEND_TIME}",
        "No-dividend spots adjusted to match initial terminal forwards.",
        "Early exercise is first exercise before maturity; European=0.",
        "This is a three-point stability check, not a proof of convergence.",
        "",
        "Columns: N | scenario | price(EUR) | early first exercise(%) | "
        "first exercise at dividend(%) | terminal forward(EUR)",
    ]
    for r in sorted(summary_rows, key=lambda row: (int(row["nb_steps"]),
                                                 SCENARIOS.index(str(row["scenario"])))):
        lines.append(
            # Keep stopping-time probabilities distinct from exercise-region mass.
            f"{int(r['nb_steps']):3d} | {str(r['scenario']):25s} | "
            f"{float(r['initial_option_price']):.10f} | "
            f"{100.0 * float(r['early_exercise_probability']):9.6f} | "
            f"{100.0 * float(r['first_exercise_at_dividend']):9.6f} | "
            f"{float(r['terminal_forward']):.10f}"
        )
    lines += ["", "Change between N=50 and N=200 (not an asymptotic error estimate):"]
    for key in SCENARIOS:
        low = next(r for r in summary_rows if r["scenario"] == key and r["nb_steps"] == 50)
        high = next(r for r in summary_rows if r["scenario"] == key and r["nb_steps"] == 200)
        early_delta = 100 * (
            float(high["early_exercise_probability"])
            - float(low["early_exercise_probability"])
        )
        lines.append(
            f"  {key}: price delta={float(high['initial_option_price']) - float(low['initial_option_price']):+.10f} EUR; "
            f"early exercise delta={early_delta:+.6f} pp"
        )
    STABILITY_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sensitivity(baseline_csv: Path = BASELINE_CSV,
                    baseline_summary: Path = BASELINE_SUMMARY,
                    *, write: bool = True) -> list[dict[str, float | int | str]]:
    """Reuse N=100 and calculate only N=50,200; no original outputs overwritten."""
    check_baseline(baseline_summary)
    baseline = load_baseline(baseline_csv)
    summary_rows = summarize_steps(baseline, BASELINE_N)
    fresh: dict[int, list[dict[str, float | int | str]]] = {}
    for steps in NEW_STEPS:
        print(f"Calculating N={steps} on the real pricing engines...")
        outcomes = run_experiment(steps)
        rows = [convert_row(row) for outcome in outcomes for row in outcome.rows]
        fresh[steps] = rows
        summary_rows.extend(summarize_steps(rows, steps))
    summary_rows.sort(key=lambda r: (int(r["nb_steps"]),
                                     SCENARIOS.index(str(r["scenario"]))))
    if write:
        save_outputs(summary_rows, fresh)
    return summary_rows


# Separate numerical sensitivity results from the original experiment output.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE_CSV,
                        help="Read-only N=100 temporal profile CSV")
    parser.add_argument("--baseline-summary", type=Path, default=BASELINE_SUMMARY,
                        help="Read-only N=100 parameter summary")
    args = parser.parse_args()
    rows = run_sensitivity(args.baseline, args.baseline_summary)
    for row in rows:
        print(f"N={int(row['nb_steps']):3d} | {str(row['scenario']):25s} | "
              f"premium={float(row['initial_option_price']):.8f} EUR | "
              f"early_exercise={float(row['early_exercise_probability']):.3%} | "
              f"dividend_date_first={float(row['first_exercise_at_dividend']):.3%}")
    print(f"CSV summary: {STABILITY_CSV.relative_to(ROOT)}")
    print(f"Text summary: {STABILITY_SUMMARY.relative_to(ROOT)}")
    print("Figures: results/figures/option_by_dates_sensitivity_*.png")
    print("Original N=100 experiment outputs remain unchanged.")


if __name__ == "__main__":
    main()
