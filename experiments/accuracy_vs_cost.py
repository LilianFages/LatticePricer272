"""Accuracy / computational-cost experiment for European vanilla options.

Run from the repository root:
    python -m experiments.accuracy_vs_cost
    python -m experiments.accuracy_vs_cost --quick

No existing pricing engine is modified. Timings are end-to-end, including
lattice initialization, construction and option backward induction.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.allocation_pruned_trinomial_tree import AllocationPrunedTrinomialTree
from src.binomial_tree import BinomialTree
from src.black_scholes import BlackScholesPricer
from src.option import CallOption, Option, PutOption
from src.recursive_pricer import RecursivePricer
from src.trinomial_tree import TrinomialTree


SPOT = 100.0
RATE = 0.02
VOLATILITY = 0.20
MATURITY = 1.0
PRUNING_THRESHOLD = 1e-18
STEP_COUNTS = (20, 40, 60, 100, 150, 200, 300)
STRIKES = (80.0, 100.0, 120.0)
OPTION_KINDS = ("call", "put")
MODELS = ("binomial", "trinomial_full", "trinomial_branch", "trinomial_allocation")
REPEATS = 5
ERROR_TARGETS = (0.10, 0.05, 0.01, 0.005)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_DATA_DIR = RESULTS / "data"
DEFAULT_FIGURES_DIR = RESULTS / "figures"

MODEL_LABELS = {
    "binomial": "Binomial",
    "trinomial_full": "Trinomial (no pruning)",
    "trinomial_branch": "Trinomial (branch pruning)",
    "trinomial_allocation": "Trinomial (allocation pruning)",
}


@dataclass(frozen=True)
class TimingObservation:
    nb_steps: int
    strike: float
    option_kind: str
    model: str
    repetition: int
    runtime_seconds: float
    price: float


@dataclass(frozen=True)
class Result:
    nb_steps: int
    strike: float
    option_kind: str
    model: str
    reference_price: float
    price: float | None
    absolute_error: float | None
    relative_error: float | None
    median_runtime_seconds: float | None
    min_runtime_seconds: float | None
    max_runtime_seconds: float | None
    allocated_nodes: int | None
    status: str
    failure_reason: str


def create_option(kind: str, strike: float) -> Option:
    """Create one *European* call or put."""
    if kind == "call":
        return CallOption(strike=strike)
    if kind == "put":
        return PutOption(strike=strike)
    raise ValueError(f"Unsupported option kind: {kind}")


def make_tree(model: str, nb_steps: int) -> (
    BinomialTree | TrinomialTree | AllocationPrunedTrinomialTree
):
    """Instantiate the existing models with identical market inputs."""
    common = {
        "spot": SPOT,
        "rate": RATE,
        "volatility": VOLATILITY,
        "maturity": MATURITY,
        "nb_steps": nb_steps,
    }
    if model == "binomial":
        return BinomialTree(**common)
    if model == "trinomial_full":
        return TrinomialTree(**common, pruning_threshold=0.0)
    if model == "trinomial_branch":
        return TrinomialTree(**common, pruning_threshold=PRUNING_THRESHOLD)
    if model == "trinomial_allocation":
        return AllocationPrunedTrinomialTree(
            **common, pruning_threshold=PRUNING_THRESHOLD
        )
    raise ValueError(f"Unsupported model: {model}")


def allocated_nodes(
    model: str,
    nb_steps: int,
    tree: BinomialTree | TrinomialTree | AllocationPrunedTrinomialTree,
) -> int:
    """Return physical allocations, not the count of reachable nodes.

    No-dividend binomial: sum_{j=0}^N (j+1).
    No-dividend full trinomial: sum_{j=0}^N (2*j+1).
    The branch-pruned trinomial keeps the same full column allocation.
    The allocation-pruned tree directly instruments its allocations.
    """
    if model == "binomial":
        return (nb_steps + 1) * (nb_steps + 2) // 2
    if model in ("trinomial_full", "trinomial_branch"):
        return (nb_steps + 1) ** 2
    if model == "trinomial_allocation":
        if not isinstance(tree, AllocationPrunedTrinomialTree):
            raise TypeError("Allocation-pruning tree expected.")
        return tree.allocated_node_count
    raise ValueError(f"Unsupported model: {model}")


def bs_reference(kind: str, strike: float) -> float:
    """Exact Black-Scholes benchmark within the no-dividend assumptions."""
    return BlackScholesPricer().price(
        option=create_option(kind, strike),
        spot=SPOT,
        rate=RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
    )


def measure_case(
    model: str,
    nb_steps: int,
    kind: str,
    strike: float,
    repeats: int,
    reference: float,
) -> tuple[Result, list[TimingObservation]]:
    """Warm up, measure independent runs, then aggregate by median.

    gc.collect() and plot/file operations are outside timed sections.
    A new tree is created for each timed repetition.
    """
    if repeats < 1:
        raise ValueError("At least one timed repetition is required.")

    option = create_option(kind, strike)
    observations: list[TimingObservation] = []
    runtimes: list[float] = []
    prices: list[float] = []
    node_counts: list[int] = []

    try:
        # Untimed warmup: imports, caches, and initial dispatch costs.
        warmup_tree = make_tree(model, nb_steps)
        RecursivePricer().price(warmup_tree, option)
        del warmup_tree
        gc.collect()

        for repetition in range(1, repeats + 1):
            gc.collect()
            started = time.perf_counter_ns()
            tree = make_tree(model, nb_steps)
            price = RecursivePricer().price(tree, option)
            elapsed = (time.perf_counter_ns() - started) / 1e9

            if not (math.isfinite(price) and elapsed > 0.0):
                raise ArithmeticError("Non-finite price or non-positive runtime.")

            counts = allocated_nodes(model, nb_steps, tree)
            runtimes.append(elapsed)
            prices.append(price)
            node_counts.append(counts)
            observations.append(
                TimingObservation(
                    nb_steps=nb_steps,
                    strike=strike,
                    option_kind=kind,
                    model=model,
                    repetition=repetition,
                    runtime_seconds=elapsed,
                    price=price,
                )
            )
            del tree

        if any(count != node_counts[0] for count in node_counts):
            raise ArithmeticError("Non-deterministic node allocation count.")
        if any(
            not math.isclose(price, prices[0], rel_tol=1e-12, abs_tol=1e-12)
            for price in prices[1:]
        ):
            raise ArithmeticError("Non-deterministic option prices across runs.")

        error = abs(prices[0] - reference)
        result = Result(
            nb_steps=nb_steps,
            strike=strike,
            option_kind=kind,
            model=model,
            reference_price=reference,
            price=prices[0],
            absolute_error=error,
            relative_error=error / reference if reference > 0 else None,
            median_runtime_seconds=statistics.median(runtimes),
            min_runtime_seconds=min(runtimes),
            max_runtime_seconds=max(runtimes),
            allocated_nodes=node_counts[0],
            status="success",
            failure_reason="",
        )
    except (ValueError, RuntimeError, OverflowError, RecursionError, ArithmeticError) as error:
        result = Result(
            nb_steps=nb_steps,
            strike=strike,
            option_kind=kind,
            model=model,
            reference_price=reference,
            price=None,
            absolute_error=None,
            relative_error=None,
            median_runtime_seconds=None,
            min_runtime_seconds=None,
            max_runtime_seconds=None,
            allocated_nodes=None,
            status="failed",
            failure_reason=f"{type(error).__name__}: {error}",
        )
    return result, observations


def run_experiment(
    steps: tuple[int, ...],
    strikes: tuple[float, ...],
    kinds: tuple[str, ...],
    repeats: int,
) -> tuple[list[Result], list[TimingObservation]]:
    """Execute the complete deterministic grid and print each status."""
    if any(n < 1 for n in steps) or len(set(steps)) != len(steps):
        raise ValueError("Steps must be distinct positive integers.")
    if len(set(strikes)) != len(strikes) or any(k <= 0 for k in strikes):
        raise ValueError("Strikes must be distinct positive numbers.")

    results: list[Result] = []
    raw: list[TimingObservation] = []
    total = len(steps) * len(strikes) * len(kinds) * len(MODELS)
    counter = 0

    for kind in kinds:
        for strike in strikes:
            reference = bs_reference(kind, strike)
            for nb_steps in steps:
                for model in MODELS:
                    counter += 1
                    result, timings = measure_case(
                        model, nb_steps, kind, strike, repeats, reference
                    )
                    results.append(result)
                    raw.extend(timings)
                    if result.status == "success":
                        assert result.absolute_error is not None
                        assert result.median_runtime_seconds is not None
                        print(
                            f"[{counter:>3}/{total}] {kind:4s} K={strike:>5g} "
                            f"N={nb_steps:>3} {model:22s} "
                            f"error={result.absolute_error:.5e} "
                            f"time={1e3 * result.median_runtime_seconds:.3f} ms",
                            flush=True,
                        )
                    else:
                        print(
                            f"[{counter:>3}/{total}] {kind:4s} K={strike:>5g} "
                            f"N={nb_steps:>3} {model:22s} "
                            f"FAILED: {result.failure_reason}",
                            flush=True,
                        )
    return results, raw


def contract_results(
    results: list[Result], kind: str = "call", strike: float = 100.0
) -> list[Result]:
    """Filter successful results for one financial contract."""
    return [
        result for result in results
        if result.status == "success"
        and result.option_kind == kind
        and result.strike == strike
    ]


def pareto_front(results: list[Result]) -> list[Result]:
    """Non-dominated accuracy/time points (minimize both axes).

    A point is dominated if another costs no more time and has no greater
    pricing error, with at least one strict improvement.
    """
    good = [
        row for row in results
        if row.status == "success"
        and row.median_runtime_seconds is not None
        and row.absolute_error is not None
    ]
    front = []
    for point in good:
        assert point.absolute_error is not None
        assert point.median_runtime_seconds is not None
        dominated = any(
            other is not point
            and other.median_runtime_seconds is not None
            and other.absolute_error is not None
            and other.median_runtime_seconds <= point.median_runtime_seconds
            and other.absolute_error <= point.absolute_error
            and (
                other.median_runtime_seconds < point.median_runtime_seconds
                or other.absolute_error < point.absolute_error
            )
            for other in good
        )
        if not dominated:
            front.append(point)
    return sorted(front, key=lambda row: float(row.median_runtime_seconds))


def fastest_at_error_target(
    results: list[Result], target: float
) -> Result | None:
    """Best sampled runtime satisfying the *absolute* error constraint."""
    candidates = [
        row for row in results
        if row.status == "success"
        and row.absolute_error is not None
        and row.median_runtime_seconds is not None
        and row.absolute_error <= target
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row.median_runtime_seconds))


def target_table(results: list[Result]) -> list[dict[str, str | int | float]]:
    """Generate sampled minimum-cost results, without interpolation."""
    rows: list[dict[str, str | int | float]] = []
    observed_kinds = tuple(kind for kind in OPTION_KINDS
                           if any(row.option_kind == kind for row in results))
    observed_strikes = tuple(strike for strike in STRIKES
                             if any(row.strike == strike for row in results))
    for kind in observed_kinds:
        for strike in observed_strikes:
            for model in MODELS:
                subset = [
                    row for row in contract_results(results, kind, strike)
                    if row.model == model
                ]
                for target in ERROR_TARGETS:
                    best = fastest_at_error_target(subset, target)
                    rows.append(
                        {
                            "option_kind": kind,
                            "strike": strike,
                            "model": model,
                            "absolute_error_target": target,
                            "status": "reached" if best else "not_reached",
                            "selected_steps": best.nb_steps if best else "",
                            "observed_absolute_error": (
                                best.absolute_error if best else ""
                            ),
                            "median_runtime_seconds": (
                                best.median_runtime_seconds if best else ""
                            ),
                        }
                    )
    return rows


def write_csv(
    file_path: Path, rows: list[dict], fieldnames: list[str] | None = None
) -> None:
    """Write a UTF-8 CSV with deterministic field order."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot infer columns for empty CSV: {file_path}")
    if fieldnames is None:
        fieldnames = list(rows[0])
    with file_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _positive_points(
    results: list[Result], x_attribute: str, y_attribute: str
) -> tuple[list[float], list[float]]:
    """Return strictly positive pairs for mathematically valid log axes."""
    x: list[float] = []
    y: list[float] = []
    for result in results:
        a = getattr(result, x_attribute)
        b = getattr(result, y_attribute)
        if a is not None and b is not None and a > 0 and b > 0:
            x.append(float(a))
            y.append(float(b))
    return x, y


def plot_comparison(
    results: list[Result],
    x_attribute: str,
    x_label: str,
    path: Path,
    title: str,
) -> None:
    """Plot one log-log relationship, with one series per model."""
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    omitted = 0
    for model in MODELS:
        subset = sorted(
            [row for row in results if row.model == model],
            key=lambda row: row.nb_steps,
        )
        x, y = _positive_points(subset, x_attribute, "absolute_error")
        omitted += sum(
            row.absolute_error == 0.0 for row in subset
            if row.status == "success"
        )
        if x:
            ax.loglog(x, y, "o-", markersize=5, label=MODEL_LABELS[model])
    ax.set(title=title, xlabel=x_label, ylabel="Absolute error vs Black-Scholes (€)")
    ax.grid(True, which="both", alpha=0.20)
    ax.legend(fontsize=9)
    if omitted:
        ax.text(
            0.015, 0.015,
            f"Exact zero errors excluded from log axes: {omitted}",
            transform=ax.transAxes, fontsize=8, va="bottom",
        )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_pareto(results: list[Result], path: Path) -> None:
    """Visualize all accuracy/cost choices and the sampled Pareto front."""
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for model in MODELS:
        subset = [row for row in results if row.model == model]
        x, y = _positive_points(
            subset, "median_runtime_seconds", "absolute_error"
        )
        if x:
            ax.scatter(x, y, s=55, label=MODEL_LABELS[model], alpha=0.8)

    front = pareto_front(results)
    fx, fy = _positive_points(
        front, "median_runtime_seconds", "absolute_error"
    )
    if fx:
        ax.plot(fx, fy, color="black", linestyle="--", linewidth=1.4,
                label="Observed Pareto frontier")
        for row in front:
            if row.absolute_error is not None and row.absolute_error > 0:
                ax.annotate(
                    str(row.nb_steps),
                    (float(row.median_runtime_seconds), float(row.absolute_error)),
                    xytext=(4, 5), textcoords="offset points", fontsize=7,
                )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(
        title="Observed precision/cost Pareto frontier — ATM European call",
        xlabel="Median end-to-end runtime (seconds)",
        ylabel="Absolute error vs Black-Scholes (€)",
    )
    ax.grid(True, which="both", alpha=0.20)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parity_deviations(results: list[Result]) -> list[float]:
    """Put-call parity residuals for matched successful lattice cases."""
    calls = {
        (row.model, row.nb_steps, row.strike): row.price
        for row in results if row.option_kind == "call" and row.status == "success"
    }
    residuals = []
    target_factor = math.exp(-RATE * MATURITY)
    for row in results:
        if row.option_kind != "put" or row.status != "success":
            continue
        call_price = calls.get((row.model, row.nb_steps, row.strike))
        if call_price is not None and row.price is not None:
            residuals.append(
                abs((call_price - row.price) - (SPOT - row.strike * target_factor))
            )
    return residuals


def git_head() -> str:
    """Return the current local commit; metadata only, no Git mutation."""
    try:
        command = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=ROOT, check=True, capture_output=True, text=True, timeout=3,
        )
        return command.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "not_available"


def save_outputs(
    results: list[Result], raw: list[TimingObservation],
    repeats: int, steps: tuple[int, ...], strikes: tuple[float, ...],
    kinds: tuple[str, ...], data_dir: Path, figure_dir: Path,
    summary_path: Path,
) -> None:
    """Export complete raw and aggregated tables, figures, and provenance."""
    write_csv(
        data_dir / "accuracy_cost_benchmark.csv",
        [asdict(row) for row in results],
    )
    write_csv(
        data_dir / "accuracy_cost_timings_raw.csv",
        [asdict(row) for row in raw],
        fieldnames=list(TimingObservation.__dataclass_fields__),
    )
    targets = target_table(results)
    write_csv(data_dir / "accuracy_cost_targets.csv", targets)

    for strike in strikes:
        if "call" not in kinds:
            break
        subset = contract_results(results, "call", strike)
        plot_comparison(
            subset,
            "median_runtime_seconds",
            "Median end-to-end runtime (seconds)",
            figure_dir / f"accuracy_cost_runtime_call_K{strike:g}.png",
            f"Accuracy vs runtime — European call, K={strike:g}",
        )

    if "call" in kinds and 100.0 in strikes:
        atm = contract_results(results, "call", 100.0)
        plot_comparison(
            atm,
            "allocated_nodes",
            "Allocated lattice nodes",
            figure_dir / "accuracy_cost_nodes_atm_call.png",
            "Accuracy vs allocated nodes — ATM European call",
        )
        plot_comparison(
            atm,
            "nb_steps",
            "Time steps N",
            figure_dir / "accuracy_cost_steps_atm_call.png",
            "Convergence vs steps — ATM European call",
        )
        plot_pareto(
            atm, figure_dir / "accuracy_cost_pareto_atm_call.png",
        )

    failed = [row for row in results if row.status != "success"]
    zero_errors = sum(
        row.absolute_error == 0.0 for row in results
        if row.status == "success"
    )
    parity = parity_deviations(results)
    atm_front = (
        pareto_front(contract_results(results, "call", 100.0))
        if "call" in kinds and 100.0 in strikes else []
    )
    lines = [
        "Accuracy versus computational cost — European vanilla options",
        "============================================================",
        "",
        f"Parameters: spot={SPOT}, rate={RATE}, sigma={VOLATILITY}, T={MATURITY}",
        f"Pruning threshold={PRUNING_THRESHOLD:.0e}",
        f"Strikes={strikes}; types={kinds}; N={steps}",
        f"Timed repetitions per configuration={repeats}",
        "Timing: median end-to-end (constructor + build + recursive pricing).",
        "An unmeasured warmup precedes each configuration.",
        "Explicit gc.collect() before, but not inside, the timed section.",
        "Timings exclude graphs, CSV, tracemalloc and explicit gc.collect().",
        "Automatic GC remains enabled and may occur during pricing.",
        f"Python={platform.python_version()}; OS={platform.platform()}",
        f"Git HEAD={git_head()}",
        "",
        f"Total configurations={len(results)}; successes={len(results) - len(failed)}; "
        f"failures={len(failed)}",
        f"Exactly zero absolute errors={zero_errors} (excluded from log axes)",
        f"Maximum put-call parity residual={max(parity):.12e}"
        if parity else "Put-call parity: no matched pairs measured.",
        "",
        "Interpretation:",
        "  Errors use *analytical* Black-Scholes for European no-dividend options.",
        "  Comparisons are conditional on the sampled time-step grid and this PC.",
        "  Pruning may alter numerical prices; its effect is measured, not assumed zero.",
        "  Empirical timing and convergence oscillations are not complexity proofs.",
        "  Hybrid omitted: without dividends it has binomial transitions.",
        "  Pareto frontier is limited to sampled points; no interpolation.",
        "",
        "Observed sampled Pareto frontier (ATM European call):",
    ]
    for row in atm_front:
        lines.append(
            f"  {row.model:22s} N={row.nb_steps:<4d} "
            f"time={row.median_runtime_seconds:.9f} s "
            f"error={row.absolute_error:.12e} €"
        )
    lines.extend(["", "Fastest sampled model attaining each ATM-call target:"])
    for target in ERROR_TARGETS:
        fastest = fastest_at_error_target(
            contract_results(results, "call", 100.0), target
        )
        if fastest is None:
            lines.append(f"  error <= {target:g} €: not reached")
        else:
            lines.append(
                f"  error <= {target:g} €: {fastest.model} "
                f"N={fastest.nb_steps} time={fastest.median_runtime_seconds:.9f} s "
                f"error={fastest.absolute_error:.12e} €"
            )
    if failed:
        lines.extend(["", "Failed configurations:"])
        for row in failed:
            lines.append(
                f"  {row.model} {row.option_kind} K={row.strike:g} "
                f"N={row.nb_steps}: {row.failure_reason}"
            )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a small smoke benchmark in a separate quick-output folder",
    )
    parser.add_argument(
        "--repeats", type=int, default=REPEATS,
        help=f"Timed runs for each configuration (default: {REPEATS})",
    )
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    if args.quick:
        steps = (20, 40)
        strikes = (100.0,)
        kinds = ("call",)
        data_dir = RESULTS / "accuracy_cost_quick" / "data"
        figure_dir = RESULTS / "accuracy_cost_quick" / "figures"
        summary_path = RESULTS / "accuracy_cost_quick" / "accuracy_cost_summary.txt"
    else:
        steps = STEP_COUNTS
        strikes = STRIKES
        kinds = OPTION_KINDS
        data_dir = DEFAULT_DATA_DIR
        figure_dir = DEFAULT_FIGURES_DIR
        summary_path = RESULTS / "accuracy_cost_summary.txt"

    print("Accuracy vs Computational Cost — Black-Scholes reference")
    print(f"Configurations={len(steps)*len(strikes)*len(kinds)*len(MODELS)} "
          f"timed runs each={args.repeats}", flush=True)
    print("Time measurements may vary with CPU load.\n", flush=True)

    results, raw = run_experiment(steps, strikes, kinds, args.repeats)
    save_outputs(
        results, raw, args.repeats, steps, strikes, kinds,
        data_dir, figure_dir, summary_path,
    )
    failures = sum(row.status != "success" for row in results)
    print(f"\nCompleted: {len(results)-failures}/{len(results)} configurations successful.")
    print("Aggregated CSV:", data_dir / "accuracy_cost_benchmark.csv")
    print("Raw timings:", data_dir / "accuracy_cost_timings_raw.csv")
    print("Target table:", data_dir / "accuracy_cost_targets.csv")
    print("Summary:", summary_path)
    print("Figures:", figure_dir)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
