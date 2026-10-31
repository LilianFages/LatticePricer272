"""Phase 2: robust multi-strike accuracy versus computational cost.

Run from the repository root:
    python -m experiments.performance.accuracy_cost_multistrike --quick
    python -m experiments.performance.accuracy_cost_multistrike
    python -m experiments.performance.accuracy_cost_multistrike --baseline-only

This script reuses the original phase-1 CSV; it only times additional
N values. The pricing engines and phase-1 output files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

# Retain the phase-one baseline and time only additional discretizations.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analytics.accuracy_metrics import nondominated_indices

from experiments.performance.pricing_accuracy_cost import (
    MODELS,
    MODEL_LABELS,
    REPEATS,
    ROOT,
    STEP_COUNTS,
    STRIKES,
    Result,
    TimingObservation,
    bs_reference,
    git_head,
    measure_case,
    write_csv,
)

# Densify the original grid without repeating phase-1 measurements.
ADDITIONAL_STEPS = (80, 125, 175, 225, 250, 275, 325, 350)
TARGET_ERRORS = (0.02, 0.01, 0.005, 0.002, 0.001)
BASELINE_CSV = ROOT / "results" / "data" / "accuracy_cost_benchmark.csv"
BASELINE_SUMMARY = ROOT / "results" / "accuracy_cost_summary.txt"


@dataclass(frozen=True)
class Aggregate:
    model: str
    nb_steps: int
    worst_absolute_error: float
    mean_absolute_error: float
    root_mean_square_error: float
    worst_strike: float
    # Budget for valuing ONE arbitrary strike from the predefined basket.
    max_median_runtime_seconds: float
    # Estimated aggregate time for three SEPARATE valuations (not batch-timed).
    sum_median_runtime_seconds: float
    allocated_nodes_per_tree: int
    source: str  # phase1 or newly_measured


def _validated_baseline_metrics(
    data: dict[str, str], strike: float, key: tuple[str, int, float]
) -> tuple[float, float, float, float, int]:
    """Independently validate Black–Scholes error and positive measured cost."""
    reference = float(data["reference_price"])
    actual_reference = bs_reference("call", strike)
    if not math.isclose(reference, actual_reference, rel_tol=1e-11, abs_tol=1e-9):
        raise ValueError(
            f"Black-Scholes reference mismatch at K={strike}. "
            "The baseline may use different market parameters."
        )
    price = float(data["price"])
    error = abs(price - reference)
    stored_error = float(data["absolute_error"])
    if not math.isclose(error, stored_error, rel_tol=1e-10, abs_tol=1e-11):
        raise ValueError(f"Inconsistent baseline price/error at {key}.")
    t = float(data["median_runtime_seconds"])
    nodes = int(data["allocated_nodes"])
    if not (math.isfinite(t) and t > 0 and nodes > 0):
        raise ValueError(f"Invalid baseline cost data at {key}.")
    return reference, price, error, t, nodes


def _parse_baseline_call_record(
    data: dict[str, str], seen: set[tuple[str, int, float]]
) -> Result | None:
    """Validate price, reference, grid key and measured cost for one call."""
    if data["option_kind"] != "call":
        return None  # Phase 2 compares calls; puts remain in phase 1.
    if data["status"] != "success":
        raise ValueError("Phase-1 call grid contains failed cases.")
    model = data["model"]
    n = int(data["nb_steps"])
    strike = float(data["strike"])
    key = (model, n, strike)
    if key in seen:
        raise ValueError(f"Duplicate baseline row: {key}")
    seen.add(key)
    if model not in MODELS or n not in STEP_COUNTS or strike not in STRIKES:
        raise ValueError(f"Unexpected baseline combination: {key}")
    reference, price, error, t, nodes = _validated_baseline_metrics(data, strike, key)
    return Result(
            nb_steps=n,
            strike=strike,
            option_kind="call",
            model=model,
            # Aggregate error across all strikes before selecting an admissible configuration.
            reference_price=reference,
            price=price,
            absolute_error=error,
            relative_error=(error / reference if reference > 0 else None),
            median_runtime_seconds=t,
            min_runtime_seconds=float(data["min_runtime_seconds"]),
            max_runtime_seconds=float(data["max_runtime_seconds"]),
            allocated_nodes=nodes,
            status="success",
            failure_reason="",
    )


def load_baseline(path: Path = BASELINE_CSV) -> list[Result]:
    """Load and validate the *complete* phase-1 call cross-section.

    Match Black-Scholes references to avoid silently mixing market inputs.
    Explicitly reject incomplete grids or duplicate combinations.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing phase-1 CSV: {path}. Run the original accuracy_vs_cost "
            "experiment first, without --quick."
        )

    records: list[Result] = []
    seen: set[tuple[str, int, float]] = set()
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for data in csv.DictReader(handle):
                record = _parse_baseline_call_record(data, seen)
                if record is not None:
                    records.append(record)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid phase-1 CSV schema: {exc}") from exc

    # Read the saved reference measurements without recomputing the baseline.
    required = {
        (model, n, strike)
        for model in MODELS
        for n in STEP_COUNTS
        for strike in STRIKES
    }
    missing = required - seen
    if missing:
        raise ValueError(f"Incomplete phase-1 call grid; missing {len(missing)} rows.")
    return records


def validate_source_metadata(path: Path = BASELINE_SUMMARY) -> None:
    """Guard against a baseline CSV copied from another configuration."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing phase-1 summary: {path}")
    summary = path.read_text(encoding="utf-8")
    if (
        "spot=100.0, rate=0.02, sigma=0.2, T=1.0" not in summary
        or "Pruning threshold=1e-18" not in summary
        or "Timed repetitions per configuration=5" not in summary
    ):
        raise ValueError(
            "Phase-1 configuration/measurement metadata does not match "
            "the phase-2 protocol. Inspect the baseline summary."
        )


def _log_multistrike_measurement(
    idx: int, total: int, strike: float, n: int, model: str, result: Result
) -> None:
    """Print the observed numerical error and measured time, or failure."""
    if result.status == "success":
        print(
            f"[{idx:>3}/{total}] K={strike:>5g} N={n:>3} "
            f"{model:22s} error={result.absolute_error:.4e} "
            f"time={result.median_runtime_seconds * 1000:.2f} ms",
            flush=True,
        )
    else:
        print(
            f"[{idx:>3}/{total}] K={strike:>5g} N={n:>3} "
            f"{model:22s} FAILED {result.failure_reason}",
            flush=True,
        )


def measure_new_steps(
    steps: tuple[int, ...], repeats: int
) -> tuple[list[Result], list[TimingObservation]]:
    """Benchmark only missing N values; European calls at three strikes."""
    if repeats < 1:
        raise ValueError("At least one repetition is required.")
    if set(steps) & set(STEP_COUNTS):
        raise ValueError("Additional steps overlap with phase-1 steps.")
    if len(set(steps)) != len(steps) or any(n < 1 for n in steps):
        raise ValueError("Steps must be distinct positive integers.")

    records: list[Result] = []
    raw: list[TimingObservation] = []
    total = len(steps) * len(STRIKES) * len(MODELS)
    idx = 0
    for strike in STRIKES:
        reference = bs_reference("call", strike)
        for n in steps:
            for model in MODELS:
                idx += 1
                result, observations = measure_case(
                    # Time only the additional lattice resolutions, preserving the phase-one sample.
                    model=model, nb_steps=n, kind="call",
                    strike=strike, repeats=repeats, reference=reference,
                )
                records.append(result)
                raw.extend(observations)
                _log_multistrike_measurement(idx, total, strike, n, model, result)
    return records, raw


def _validated_strike_statistics(
    model: str, n: int, rows: list[Result], strikes: tuple[float, ...],
    models: tuple[str, ...]
) -> tuple[list[float], list[float], list[int]]:
    """Reject incomplete strike grids, price errors and invalid timings."""
    if model not in models:
        raise ValueError(f"Unknown model {model}")
    if len(rows) != len(strikes) or {r.strike for r in rows} != set(strikes):
        raise ValueError(f"Incomplete or duplicate strike coverage for {model}, N={n}.")
    errors = [float(r.absolute_error) for r in rows]
    times = [float(r.median_runtime_seconds) for r in rows]
    nodes = [int(r.allocated_nodes) for r in rows]
    if len(set(nodes)) != 1:
        raise ValueError(f"Node allocation changed with strike for {model}, N={n}.")
    if any(not math.isfinite(x) or x < 0 for x in errors):
        raise ValueError("Invalid absolute errors.")
    if any(not math.isfinite(x) or x <= 0 for x in times):
        raise ValueError("Invalid runtime.")
    return errors, times, nodes


def aggregate_cross_strike(
    records: list[Result],
    strikes: tuple[float, ...] = STRIKES,
    models: tuple[str, ...] = MODELS
) -> list[Aggregate]:
    """Aggregate one row per (model, N); reject incomplete strike groups."""
    groups: dict[tuple[str, int], list[Result]] = {}
    for record in records:
        if record.option_kind != "call":
            raise ValueError("Multi-strike aggregation accepts European calls only.")
        if record.status != "success":
            raise ValueError("Cannot aggregate failed cases as complete groups.")
        groups.setdefault((record.model, record.nb_steps), []).append(record)

    output = []
    for (model, n), rows in sorted(groups.items()):
        errors, times, nodes = _validated_strike_statistics(model, n, rows, strikes, models)
        worst = max(rows, key=lambda x: float(x.absolute_error))
        output.append(
            Aggregate(
                model=model,
                nb_steps=n,
                worst_absolute_error=max(errors),
                # Aggregate pricing error across the predefined strikes before ranking models.
                mean_absolute_error=statistics.mean(errors),
                root_mean_square_error=math.sqrt(statistics.mean(x*x for x in errors)),
                worst_strike=worst.strike,
                max_median_runtime_seconds=max(times),
                sum_median_runtime_seconds=sum(times),
                allocated_nodes_per_tree=nodes[0],
                source="phase1" if n in STEP_COUNTS else "newly_measured",
            )
        )
    return output


def pareto_multi(rows: list[Aggregate]) -> list[Aggregate]:
    """Observed multi-strike frontier with the original two-key ordering."""
    pairs = [(row.max_median_runtime_seconds, row.worst_absolute_error)
             for row in rows]
    front = [rows[index] for index in nondominated_indices(pairs)]
    return sorted(front, key=lambda r: (r.max_median_runtime_seconds, r.worst_absolute_error))

def target_choices(rows: list[Aggregate], targets: tuple[float, ...] = TARGET_ERRORS) -> list[dict]:
    """Select fastest feasible (model, N) for all strikes at once."""
    choices = []
    for target in targets:
        feasible = [r for r in rows if r.worst_absolute_error <= target]
        winner = (
            min(feasible, key=lambda r: (r.max_median_runtime_seconds, r.worst_absolute_error))
            if feasible else None
        )
        choices.append(
            {
                "target_eur": target,
                "status": "reached" if winner else "not_reached",
                "model": winner.model if winner else "",
                "nb_steps": winner.nb_steps if winner else "",
                "worst_absolute_error": winner.worst_absolute_error if winner else "",
                "max_median_runtime_seconds": winner.max_median_runtime_seconds if winner else "",
                "sum_median_runtime_seconds": winner.sum_median_runtime_seconds if winner else "",
            }
        )
    # Choose the fastest observed configuration satisfying each accuracy bound.
    return choices


def plot_curves(rows: list[Aggregate], field: str, path: Path, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    excluded = 0
    for model in MODELS:
        subset = sorted((r for r in rows if r.model == model), key=lambda r: r.nb_steps)
        xy = [(float(getattr(r, field)), r.worst_absolute_error) for r in subset]
        excluded += sum(1 for _, y in xy if y == 0.0)
        xy = [(x, y) for x, y in xy if x > 0.0 and y > 0.0]
        if xy:
            ax.loglog(*zip(*xy), marker="o", linewidth=1.6, markersize=4.8,
                      label=MODEL_LABELS[model])
    ax.set(xlabel=xlabel, ylabel="Maximum absolute pricing error over K=80,100,120 (€)",
           title="Cross-strike worst-case accuracy versus computational cost")
    ax.grid(which="both", alpha=0.20)
    # Aggregate error across all strikes before selecting an admissible configuration.
    ax.legend(fontsize=9)
    if excluded:
        ax.text(.02, .02, f"{excluded} exact-zero points excluded from log scale",
                transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_frontier(rows: list[Aggregate], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    for model in MODELS:
        part = [r for r in rows if r.model == model
                and r.worst_absolute_error > 0.0]
        ax.scatter([r.max_median_runtime_seconds for r in part],
                   [r.worst_absolute_error for r in part],
                   # Keep empirical runtimes separate from model-error estimates.
                   s=38, alpha=0.75, label=MODEL_LABELS[model])
    front = [r for r in pareto_multi(rows) if r.worst_absolute_error > 0]
    if front:
        ax.plot([r.max_median_runtime_seconds for r in front],
                [r.worst_absolute_error for r in front],
                linestyle="--", color="black", linewidth=1.3,
                label="Observed non-dominated frontier")
        for row in front:
            ax.annotate(f"{row.model.split('_')[-1]} N={row.nb_steps}",
                        (row.max_median_runtime_seconds, row.worst_absolute_error),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="Worst median runtime across strikes (seconds per valuation)",
           ylabel="Maximum absolute error across strikes (€)",
           title="Cross-strike worst-case Pareto frontier (observed grid)")
    ax.grid(which="both", alpha=0.20)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _aggregate_complete_strike_groups(
    baseline: list[Result], new_results: list[Result]
) -> tuple[list[Aggregate], list[Result]]:
    """Aggregate only configurations priced for all three strikes."""
    combined = baseline + new_results
    successful = [r for r in combined if r.status == "success"]
    failed = [r for r in combined if r.status != "success"]
    # Keep complete (model, N) groups, preserving any failed rows in new_cases.
    # Incomplete groups cannot support a cross-strike maximum.
    strike_sets: dict[tuple[str, int], set[float]] = {}
    for record in successful:
        key = (record.model, record.nb_steps)
        strike_sets.setdefault(key, set()).add(record.strike)
    allowed_keys = {
        key for key, strikes in strike_sets.items()
        if strikes == set(STRIKES)
        and sum(1 for record in successful
                if (record.model, record.nb_steps) == key) == len(STRIKES)
    }
    complete = aggregate_cross_strike(
        [record for record in successful
         if (record.model, record.nb_steps) in allowed_keys]
    )
    return complete, failed


def _phase2_output_files(output_dir: Path) -> dict[str, Path]:
    """Construct the paths for phase-two outputs only."""
    out_data = output_dir / "data"
    out_fig = output_dir / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)
    files = {
        "new_cases": out_data / "accuracy_cost_multistrike_new.csv",
        "timings": out_data / "accuracy_cost_multistrike_timings.csv",
        "aggregated": out_data / "accuracy_cost_multistrike_aggregate.csv",
        "pareto": out_data / "accuracy_cost_multistrike_pareto.csv",
        "targets": out_data / "accuracy_cost_multistrike_targets.csv",
        "runtime_fig": out_fig / "accuracy_cost_multistrike_error_runtime.png",
        "nodes_fig": out_fig / "accuracy_cost_multistrike_error_nodes.png",
        "pareto_fig": out_fig / "accuracy_cost_multistrike_pareto.png",
        "summary": output_dir / "accuracy_cost_multistrike_summary.txt",
    }
    return files


def _write_phase2_tables(files: dict[str, Path],
                         new_results: list[Result], raw: list[TimingObservation],
                         complete: list[Aggregate], front: list[Aggregate],
                         choices: list[dict]) -> None:
    """Persist raw measurements and aggregate tables without overwriting phase one."""
    write_csv(files["new_cases"], [asdict(r) for r in new_results],
              fieldnames=list(Result.__dataclass_fields__))
    write_csv(files["timings"], [asdict(r) for r in raw],
              fieldnames=list(TimingObservation.__dataclass_fields__))
    write_csv(files["aggregated"], [asdict(r) for r in complete],
              fieldnames=list(Aggregate.__dataclass_fields__))
    write_csv(files["pareto"], [asdict(r) for r in front],
              fieldnames=list(Aggregate.__dataclass_fields__))
    write_csv(files["targets"], choices)


def _plot_phase2_outputs(files: dict[str, Path], complete: list[Aggregate]) -> None:
    """Render runtime, allocation and Pareto figures from the same grid."""
    plot_curves(complete, "max_median_runtime_seconds", files["runtime_fig"],
                "Worst median runtime across strikes (seconds per valuation)")
    plot_curves(complete, "allocated_nodes_per_tree", files["nodes_fig"],
                "Allocated nodes per lattice")
    plot_frontier(complete, files["pareto_fig"])



def _phase2_summary_header(new_results: list[Result], failed: list[Result],
                           complete: list[Aggregate], repeats: int,
                           added_steps: tuple[int, ...]) -> list[str]:
    """Document scope, methodology and provenance of the timing experiment."""
    original_ns = ", ".join(str(x) for x in STEP_COUNTS)
    extra_ns = ", ".join(str(x) for x in added_steps) if added_steps else "none"
    lines = [
        "PHASE 2 — Cross-strike worst-case accuracy versus computational cost",
        "===================================================================",
        "",
        "Financial scope: European calls, no dividends, K=80/100/120.",
        "Black-Scholes analytic price reference; S0=100, r=0.02, sigma=0.2, T=1.",
        "Pruning threshold=1e-18.",
        "Metric: max absolute pricing error across the THREE strikes at same N.",
        "Cost per valuation: maximum of three separate median runtimes,",
        "not a measured joint / vectorized batch runtime.",
        "The sum of three medians estimates the cost to value all three contracts",
        "separately; it is not a measured batch runtime.",
        "",
        f"Phase-1 original steps: {original_ns}.",
        f"Newly measured steps: {extra_ns}.",
        # Exclude failed or incomplete observations from cross-strike comparisons.
        f"Phase-1 baseline: {BASELINE_CSV}",
        f"Phase-1 summary: {BASELINE_SUMMARY}",
        f"New timed repetitions per configuration: {repeats}.",
        "An unmeasured warmup and gc.collect() precede timed measurements.",
        "New runs timed end-to-end (constructor, build, recursive pricing).",
        f"Python={platform.python_version()}; OS={platform.platform()}",
        f"Git HEAD={git_head()}",
        "",
        f"New configurations: {len(new_results)}; success: {len(new_results) - len(failed)};"
        f" failures: {len(failed)}.",
        f"Complete (model, N, three-strike) groups: {len(complete)}.",
        "No interpolation between N points and no asymptotic convergence claims.",
        "Zero-error observations (if any) are excluded from log axes, not modified.",
        "The worst-case bound applies only to K=80/100/120, not all strikes.",
        "Baseline and new timings were measured in different runs; CPU-load drift",
        "can affect frontier membership near ties. Use the original raw timings",
        "and consider a joint rerun if cost differences are marginal.",
        # Select only observed feasible points when constructing the Pareto frontier.
        "All estimates conditional on this PC and sampled values of N.",
        "Call results only: do not claim to have remeasured put timings.",
        "No changes to src/ or original phase-1 CSVs.",
        "",
        "Best sampled configuration meeting a uniform error bound:",
    ]
    return lines


def _write_phase2_summary(files: dict[str, Path],
                         lines: list[str], choices: list[dict],
                         front: list[Aggregate], failed: list[Result]) -> None:
    """Append observed feasible targets, frontier and failure diagnostics."""
    for row in choices:
        if row["status"] == "reached":
            lines.append(
                f"  <= {row['target_eur']:.3g} EUR: {row['model']} "
                f"N={row['nb_steps']} worst error={row['worst_absolute_error']:.9g} "
                f"worst median latency={1000 * row['max_median_runtime_seconds']:.3f} ms"
            )
        else:
            lines.append(f"  <= {row['target_eur']:.3g} EUR: not reached")
    lines.extend(["", "Observed multi-strike Pareto frontier:"])
    for row in front:
        lines.append(
            f"  {row.model:22s} N={row.nb_steps:<3d} "
            f"worst_error={row.worst_absolute_error:.9g} "
            f"max_median_ms={1000 * row.max_median_runtime_seconds:.3f}"
        )
    if failed:
        # Retain the phase-one baseline and time only additional discretizations.
        lines.extend(["", "Failed new cases:"])
        for row in failed:
            lines.append(
                f"  {row.model} N={row.nb_steps} K={row.strike:g}: {row.failure_reason}"
            )
    files["summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_phase2(
    baseline: list[Result], new_results: list[Result],
    raw: list[TimingObservation], output_dir: Path, repeats: int,
    added_steps: tuple[int, ...]
) -> dict[str, Path]:
    """Separate outputs: never overwrite phase-1 or existing lattice analyses."""
    complete, failed = _aggregate_complete_strike_groups(baseline, new_results)
    if failed:
        print(f"WARNING: {len(failed)} failed new cases; incomplete groups excluded.")
    choices = target_choices(complete)
    front = pareto_multi(complete)
    files = _phase2_output_files(output_dir)
    _write_phase2_tables(files, new_results, raw, complete, front, choices)
    _plot_phase2_outputs(files, complete)
    lines = _phase2_summary_header(new_results, failed, complete, repeats, added_steps)
    _write_phase2_summary(files, lines, choices, front, failed)
    return files


def _parse_phase2_args(argv: list[str] | None) -> argparse.Namespace:
    """Validate the phase-two command line before any experiment starts."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="Smoke test: N=80 only, one repeat; separate outputs")
    group.add_argument("--baseline-only", action="store_true", help="Use phase-1 CSV only; no new timings")
    parser.add_argument("--repeats", type=int, default=REPEATS,
                        help="New timed runs per configuration (default: 5)")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    return args


def _phase2_run_configuration(args: argparse.Namespace) -> tuple[tuple[int, ...], int, Path]:
    """Select the same quick, baseline-only or full resolution grid."""
    if args.quick:
        extra_steps = (80,)
        repeats = 1
        output_dir = ROOT / "results" / "accuracy_cost_multistrike_quick"
    elif args.baseline_only:
        extra_steps = ()
        repeats = 0
        output_dir = ROOT / "results" / "accuracy_cost_multistrike_preview"
    else:
        extra_steps = ADDITIONAL_STEPS
        repeats = args.repeats
        output_dir = ROOT / "results"

    return extra_steps, repeats, output_dir


def main(argv: list[str] | None = None) -> int:
    # Aggregate error across all strikes before selecting an admissible configuration.
    args = _parse_phase2_args(argv)

    try:
        validate_source_metadata()
        baseline = load_baseline()
        extra_steps, repeats, output_dir = _phase2_run_configuration(args)

        print(f"Baseline: {len(baseline)} call configurations re-used.")
        print(f"New N values: {extra_steps if extra_steps else 'none (analysis-only)'}")
        new_results, raw = (
            measure_new_steps(extra_steps, repeats) if extra_steps else ([], [])
        )
        paths = save_phase2(baseline, new_results, raw, output_dir, repeats, extra_steps)
        print("\nSummary:", paths["summary"])
        print("Aggregate CSV:", paths["aggregated"])
        print("Figures directory:", paths["runtime_fig"].parent)
        failures = sum(r.status != "success" for r in new_results)
        return 1 if failures else 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
