"""Benchmark allocation pruning and estimate empirical power-law exponents.

Run from repository root:
    python -m experiments.allocation_pruning_benchmark

The experiment separates ordinary timing from tracemalloc measurements.
It does not change the financial pricing engine.
"""

import csv
import gc
import math
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Self

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.allocation_pruned_trinomial_tree import AllocationPrunedTrinomialTree
from src.dividend import CashDividend
from src.hybrid_tree import HybridTree
from src.node import Node, TrunkNode
from src.node import count_reachable_nodes as _count_priced_nodes
from src.option import CallOption
from src.recursive_pricer import RecursivePricer
from src.trinomial_tree import TrinomialTree


# Separate build-time measurements from peak-allocation measurements.
SPOT = 100.0
STRIKE = 100.0
RATE = 0.02
VOLATILITY = 0.20
MATURITY = 1.0
DIVIDEND_TIME = 0.5
DIVIDEND_AMOUNT = 3.0
PRUNING_THRESHOLD = 1e-18

STEP_COUNTS = [20, 40, 60, 100, 150, 200, 300]
RUNTIME_REPEATS = 5
FIT_MIN_STEPS = 40

MODEL_NAMES = {
    "trinomial_no_pruning": "Trinomial — no pruning",
    "trinomial_branch_pruning": "Trinomial — branch pruning",
    "trinomial_allocation_pruning": "Trinomial — allocation pruning",
    # Count reachable and allocated nodes using the actual lattice links.
    "hybrid": "Hybrid",
}

METRICS = {
    "total_seconds": ("End-to-end pricing time (s)", "runtime"),
    "build_seconds": ("Lattice build time (s)", "build_time"),
    "pricing_seconds": ("Backward pricing time (s)", "pricing_time"),
    "allocated_nodes": ("Allocated lattice nodes", "allocated_nodes"),
    "reachable_nodes": ("Reachable pricing nodes", "reachable_nodes"),
    "peak_python_mib": ("Peak traced Python memory (MiB)", "peak_memory"),
}

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "data"
FIGURES_DIR = ROOT / "results" / "figures"
SUMMARY_PATH = ROOT / "results" / "allocation_pruning_summary.txt"
BENCHMARK_CSV = DATA_DIR / "allocation_pruning_benchmark.csv"
FITS_CSV = DATA_DIR / "allocation_pruning_power_laws.csv"


def count_column_nodes(top: Node) -> int:
    """Count the nodes in one vertically linked lattice column."""
    count = 0
    current: Node | None = top

    while current is not None:
        count += 1
        current = current.lower_neighbor

    return count


def count_reachable_nodes(root: Node) -> int:
    """Retain the experiment API, delegate the shared pricing-graph traversal."""
    return _count_priced_nodes(root)

class TimedTrinomialTree(TrinomialTree):
    """Instrument the existing trinomial without changing its formulas."""

    def __init__(
        self: Self,
        spot: float,
        rate: float,
        volatility: float,
        maturity: float,
        nb_steps: int,
        dividends: list[CashDividend] | None = None,
        pruning_threshold: float = PRUNING_THRESHOLD
    ) -> None:
        super().__init__(
            spot, rate, volatility, maturity, nb_steps,
            dividends, pruning_threshold,
        )
        self.allocated_node_count = 1
        self.build_seconds = 0.0

    # Fit scaling exponents only over the declared discretization range.
    def _build_column(
        self: Self,
        trunk_price: float,
        top_exponent: int,
        bottom_exponent: int,
        previous_trunk: TrunkNode
    ) -> tuple[Node, TrunkNode]:
        result = super()._build_column(
            trunk_price, top_exponent, bottom_exponent, previous_trunk,
        )
        self.allocated_node_count += top_exponent - bottom_exponent + 1
        return result

    def build(self: Self) -> None:
        start = time.perf_counter()
        try:
            super().build()
        finally:
            self.build_seconds = time.perf_counter() - start


class TimedAllocationTree(AllocationPrunedTrinomialTree):
    """Time a tree whose allocated node count is already tracked."""

    def __init__(
        self: Self,
        spot: float,
        rate: float,
        volatility: float,
        maturity: float,
        nb_steps: int,
        dividends: list[CashDividend] | None = None,
        pruning_threshold: float = PRUNING_THRESHOLD
    ) -> None:
        super().__init__(
            spot, rate, volatility, maturity, nb_steps,
            dividends, pruning_threshold,
        )
        self.build_seconds = 0.0

    def build(self: Self) -> None:
        start = time.perf_counter()
        try:
            super().build()
        finally:
            self.build_seconds = time.perf_counter() - start


class TimedHybridTree(HybridTree):
    """Count node allocations in the hybrid builder and time its build."""

    def __init__(
        self: Self,
        spot: float,
        rate: float,
        volatility: float,
        maturity: float,
        nb_steps: int,
        dividends: list[CashDividend] | None = None
    ) -> None:
        super().__init__(
            spot, rate, volatility, maturity, nb_steps, dividends,
        )
        self.allocated_node_count = 1
        self.build_seconds = 0.0

    # Keep memory instrumentation outside the timed valuation loop.
    def _build_binomial_column(
        self: Self,
        current_top: Node
    ) -> Node:
        top = super()._build_binomial_column(current_top)
        self.allocated_node_count += count_column_nodes(top)
        return top

    def _build_bridge_column(
        self: Self,
        forward_price: float,
        top_exponent: int,
        bottom_exponent: int
    ) -> Node:
        top = super()._build_bridge_column(
            forward_price, top_exponent, bottom_exponent,
        )
        # Retain the same market and stopping rules across benchmark engines.
        self.allocated_node_count += count_column_nodes(top)
        return top

    def build(self: Self) -> None:
        start = time.perf_counter()
        try:
            super().build()
        finally:
            self.build_seconds = time.perf_counter() - start


BenchTree = TimedTrinomialTree | TimedAllocationTree | TimedHybridTree


def make_tree(model: str, steps: int) -> BenchTree:
    """Create a fresh tree for one measured repetition."""
    dividends = [
        CashDividend(time=DIVIDEND_TIME, amount=DIVIDEND_AMOUNT),
    ]
    common = {
        "spot": SPOT,
        "rate": RATE,
        "volatility": VOLATILITY,
        "maturity": MATURITY,
        "nb_steps": steps,
        "dividends": dividends,
    }

    if model == "trinomial_no_pruning":
        return TimedTrinomialTree(
            **common, pruning_threshold=0.0,
        )
    if model == "trinomial_branch_pruning":
        # Construct the selected pricing model under the common market parameters.
        return TimedTrinomialTree(
            **common, pruning_threshold=PRUNING_THRESHOLD,
        )
    if model == "trinomial_allocation_pruning":
        return TimedAllocationTree(
            **common, pruning_threshold=PRUNING_THRESHOLD,
        )
    if model == "hybrid":
        return TimedHybridTree(**common)

    raise ValueError(f"Unknown benchmark model: {model}")


def _timed_samples(
    model: str, steps: int
) -> dict | tuple[list[float], list[float], list[float], list[float]]:
    """Measure repeated runs without tracing allocations."""
    elapsed: list[float] = []
    builds: list[float] = []
    pricing: list[float] = []
    prices: list[float] = []
    for _ in range(RUNTIME_REPEATS):
        gc.collect()
        tree = make_tree(model, steps)
        pricer = RecursivePricer()
        option = CallOption(strike=STRIKE)
        start = time.perf_counter()
        try:
            price = pricer.price(tree, option)
        except (RuntimeError, ValueError, OverflowError) as exc:
            return {
                "nb_steps": steps,
                "model": model,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        total = time.perf_counter() - start
        elapsed.append(total)
        builds.append(tree.build_seconds)
        pricing.append(max(0.0, total - tree.build_seconds))
        prices.append(price)
        # Prevent objects from accumulating between repetitions.
        del tree, pricer, option
    gc.collect()
    return elapsed, builds, pricing, prices


def _memory_observation(model: str, steps: int) -> tuple[BenchTree, int]:
    """Measure peak traced memory in a separate, untimed pricing run."""
    tracemalloc.start()
    try:
        memory_tree = make_tree(model, steps)
        RecursivePricer().price(memory_tree, CallOption(strike=STRIKE))
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return memory_tree, peak_bytes


def _successful_timing_row(
    model: str,
    steps: int,
    samples: tuple[list[float], list[float], list[float], list[float]],
    memory_tree: BenchTree,
    peak_bytes: int
) -> dict:
    """Aggregate medians and observable allocation counts."""
    elapsed, builds, pricing, prices = samples
    allocated = memory_tree.allocated_node_count
    reachable = count_reachable_nodes(memory_tree.root)
    if reachable > allocated:
        raise AssertionError(
            f"{model} N={steps}: reachable ({reachable}) "
            f"exceeds allocated ({allocated})."
        )
    return {
        "nb_steps": steps,
        "model": model,
        "status": "success",
        "error": "",
        "price": statistics.median(prices),
        "total_seconds": statistics.median(elapsed),
        "build_seconds": statistics.median(builds),
        "pricing_seconds": statistics.median(pricing),
        "allocated_nodes": allocated,
        "reachable_nodes": reachable,
        "peak_python_mib": peak_bytes / (1024 ** 2),
        "pruned_nodes": getattr(memory_tree, "pruned_node_count", 0),
        "price_gap_to_branch": None,
        "relative_gap_to_branch": None,
    }


def run_timed_case(model: str, steps: int) -> dict:
    """Benchmark normal timings and independent traced allocations."""
    samples = _timed_samples(model, steps)
    if isinstance(samples, dict):
        return samples
    memory_tree, peak_bytes = _memory_observation(model, steps)
    return _successful_timing_row(model, steps, samples, memory_tree, peak_bytes)


def _report_benchmark_row(
    record: dict, steps: int, reference_price: float | None
) -> None:
    """Attach observed price gaps and display a pricing observation."""
    price = record.get("price")
    if price is not None and reference_price is not None:
        gap = price - reference_price
        record["price_gap_to_branch"] = gap
        record["relative_gap_to_branch"] = (
            abs(gap) / abs(reference_price) if reference_price != 0.0 else None
        )
    if record["status"] == "failed":
        print(f"  {record['model']:30s} FAILED: {record['error']}")
    else:
        print(
            f"  {record['model']:30s} "
            f"Price={record['price']:.8f}  "
            f"Nodes={record['allocated_nodes']:>7d}/"
            f"{record['reachable_nodes']:>7d}  "
            f"Total={record['total_seconds']:.4f}s  "
            f"Peak={record['peak_python_mib']:.2f}MiB"
        )


def run_benchmark() -> list[dict]:
    """Run each lattice configuration for every discretization level."""
    rows: list[dict] = []
    for steps in STEP_COUNTS:
        print(f"\nN = {steps}")
        step_rows = [run_timed_case(model, steps) for model in MODEL_NAMES]
        reference = next(
            row for row in step_rows
            if row["model"] == "trinomial_branch_pruning"
        )
        reference_price = reference.get("price")
        # Each depth compares contracts with one shared branch-pruning anchor.
        for record in step_rows:
            _report_benchmark_row(record, steps, reference_price)
            rows.append(record)
    return rows


def fit_power_law(xs: list[float], ys: list[float]) -> dict | None:
    """Fit log(y) = intercept + exponent * log(N), with ordinary LS."""
    if len(xs) < 3 or any(v <= 0.0 for v in xs + ys):
        return None

    log_x = [math.log(v) for v in xs]
    log_y = [math.log(v) for v in ys]

    x_bar = statistics.mean(log_x)
    y_bar = statistics.mean(log_y)
    xx = sum((x - x_bar) ** 2 for x in log_x)

    if xx == 0.0:
        return None

    exponent = sum(
        (x - x_bar) * (y - y_bar)
        for x, y in zip(log_x, log_y)
    ) / xx

    # Infer scaling exponents from valid positive values in the fit range.
    intercept = y_bar - exponent * x_bar
    fitted = [intercept + exponent * x for x in log_x]

    ss_total = sum((y - y_bar) ** 2 for y in log_y)
    ss_resid = sum(
        (actual - estimate) ** 2
        for actual, estimate in zip(log_y, fitted)
    )
    r_squared = (
        1.0 - ss_resid / ss_total
        if ss_total > 0.0 else 1.0
    )

    return {
        "exponent": exponent,
        "log_intercept": intercept,
        "r_squared": r_squared,
        "n_observations": len(xs),
        "min_steps": min(xs),
        "max_steps": max(xs),
    }


def compute_fits(rows: list[dict]) -> list[dict]:
    """Estimate empirical polynomial powers on N >= FIT_MIN_STEPS."""
    fits: list[dict] = []

    for model in MODEL_NAMES:
        for metric in METRICS:
            observations = [
                r for r in rows
                if r["model"] == model
                and r["status"] == "success"
                and r["nb_steps"] >= FIT_MIN_STEPS
                and r.get(metric) is not None
                and r[metric] > 0.0
            ]
            fit = fit_power_law(
                [float(r["nb_steps"]) for r in observations],
                [float(r[metric]) for r in observations],
            )

            if fit is not None:
                fits.append({
                    "model": model,
                    "metric": metric,
                    **fit,
                })

    return fits


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Save tabular results for reproducible analysis and Streamlit."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric_model(
    ax: Axes,
    rows: list[dict],
    model: str,
    label: str,
    metric: str,
    fit: dict | None
) -> None:
    """Draw sampled measurements and optional fitted scaling for one model."""
    observations = sorted(
        (
            row for row in rows
            if row["model"] == model
            and row["status"] == "success"
            and row.get(metric) is not None and row[metric] > 0.0
        ),
        key=lambda row: row["nb_steps"],
    )
    if not observations:
        return
    xs = [row["nb_steps"] for row in observations]
    ys = [row[metric] for row in observations]
    display_label = label
    if fit is not None:
        display_label += (
            f" (p={fit['exponent']:.2f}, R²={fit['r_squared']:.3f})"
        )
    line, = ax.plot(xs, ys, marker="o", label=display_label)
    if fit is not None:
        # Only the declared fitting range contributes to the fitted line.
        fit_xs = [x for x in xs if x >= FIT_MIN_STEPS]
        fit_ys = [
            math.exp(fit["log_intercept"]) * x ** fit["exponent"]
            for x in fit_xs
        ]
        ax.plot(
            fit_xs, fit_ys, "--", linewidth=1.3,
            alpha=0.65, color=line.get_color(),
        )


def save_plots(rows: list[dict], fits: list[dict]) -> None:
    """Create one log-log figure for each measured complexity metric."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fit_lookup = {(r["model"], r["metric"]): r for r in fits}
    for metric, (ylabel, short_name) in METRICS.items():
        fig, ax = plt.subplots(figsize=(9, 5.5))
        # Each model is drawn against its own observed fit, if available.
        for model, label in MODEL_NAMES.items():
            _plot_metric_model(
                ax, rows, model, label, metric, fit_lookup.get((model, metric))
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of time steps N")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Allocation pruning benchmark — {short_name} (log-log)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            FIGURES_DIR / f"allocation_pruning_{short_name}_loglog.png",
            dpi=200,
        )
        plt.close(fig)


def save_summary(rows: list[dict], fits: list[dict]) -> None:
    """Save a compact reproducible record of the comparison."""
    lines = [
        "Allocation pruning benchmark",
        "===========================",
        "",
        f"Spot={SPOT}; strike={STRIKE}; r={RATE}; sigma={VOLATILITY}; T={MATURITY}",
        f"Cash dividend={DIVIDEND_AMOUNT} at t={DIVIDEND_TIME}",
        f"Pruning threshold={PRUNING_THRESHOLD:.0e}",
        f"Time repetitions={RUNTIME_REPEATS}; median reported",
        f"Power fits use N >= {FIT_MIN_STEPS}",
        "",
        "Time measurements exclude tracemalloc overhead.",
        "Peak Python memory comes from a separate traced pricing run.",
        "Power exponents are empirical estimates, not asymptotic proofs.",
        "Trinomial no-pruning may fail on inadmissible tail probabilities.",
        "",
        "EMPIRICAL POWER LAW FITS: metric ~ C * N^p",
    ]

    # Count reachable and allocated nodes using the actual lattice links.
    for fit in fits:
        lines.append(
            f"{fit['model']:30s} {fit['metric']:20s} "
            f"p={fit['exponent']:.4f}, R2={fit['r_squared']:.4f} "
            f"(N={fit['min_steps']:.0f}..{fit['max_steps']:.0f}, "
            f"{fit['n_observations']} points)"
        )

    lines.extend(["", "PRICING GAPS TO BRANCH-PRUNED TRINOMIAL"])
    for row in rows:
        if row["status"] != "success":
            continue
        gap = row.get("price_gap_to_branch")
        if gap is not None:
            lines.append(
                f"N={row['nb_steps']:>3} "
                f"{row['model']:30s} gap={gap:+.10e}"
            )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run all benchmarks and export figures, data and regressions."""
    rows = run_benchmark()
    fits = compute_fits(rows)

    save_csv(
        BENCHMARK_CSV, rows,
        [
            "nb_steps", "model", "status", "error", "price",
            "total_seconds", "build_seconds", "pricing_seconds",
            "allocated_nodes", "reachable_nodes", "peak_python_mib",
            "pruned_nodes", "price_gap_to_branch",
            "relative_gap_to_branch",
        ],
    )
    save_csv(
        FITS_CSV, fits,
        [
            "model", "metric", "exponent", "log_intercept",
            # Fit scaling exponents only over the declared discretization range.
            "r_squared", "n_observations", "min_steps", "max_steps",
        ],
    )
    save_plots(rows, fits)
    save_summary(rows, fits)

    print("\nPOWER-LAW FITS (empirical):")
    for fit in fits:
        if fit["metric"] in ("total_seconds", "allocated_nodes"):
            print(
                f"  {fit['model']:30s} {fit['metric']:16s} "
                f"p={fit['exponent']:.3f} R²={fit['r_squared']:.3f}"
            )

    print("\nOutputs:")
    print(" ", BENCHMARK_CSV.relative_to(ROOT))
    print(" ", FITS_CSV.relative_to(ROOT))
    print(" ", SUMMARY_PATH.relative_to(ROOT))
    print(" ", FIGURES_DIR.relative_to(ROOT) / "allocation_pruning_*_loglog.png")


if __name__ == "__main__":
    main()
