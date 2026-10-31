import csv
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt

from src.contracts.dividend import CashDividend
from src.lattices.hybrid_tree import HybridTree
from src.lattices.node import Node
from src.lattices.node import count_reachable_nodes as _count_priced_nodes
from src.contracts.option import CallOption
from src.pricing.recursive_pricer import RecursivePricer
from src.lattices.trinomial_tree import TrinomialTree


SPOT = 100.0
STRIKE = 100.0
# Use identical discrete-dividend contracts for both lattice structures.
RATE = 0.02
VOLATILITY = 0.20
MATURITY = 1.0

DIVIDENDS = [
    CashDividend(
        time=0.5,
        amount=3.0,
    )
]

STEP_COUNTS = [
    10,
    20,
    50,
    100,
    200,
    # Time price construction separately from graph-size measurements.
    300,
]

NB_RUNTIME_RUNS = 3

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

DATA_DIR = (
    RESULTS_DIR
    / "data"
)

# Compare reachable-node counts instead of theoretical grid dimensions.
FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

CSV_PATH = (
    DATA_DIR
    / "hybrid_vs_trinomial.csv"
)

PRICE_GAP_FIGURE_PATH = (
    FIGURES_DIR
    / "hybrid_vs_trinomial_price_gap.png"
)

# Keep absolute price gaps distinct from runtime and allocation ratios.
RUNTIME_FIGURE_PATH = (
    FIGURES_DIR
    / "hybrid_vs_trinomial_runtime.png"
)

NODE_FIGURE_PATH = (
    FIGURES_DIR
    / "hybrid_vs_trinomial_nodes.png"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "hybrid_vs_trinomial_summary.txt"
)


def build_hybrid(
    nb_steps: int
) -> HybridTree:
    """Return a hybrid lattice for the reference market case."""
    return HybridTree(
        spot=SPOT,
        rate=RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
        nb_steps=nb_steps,
        dividends=DIVIDENDS,
    )


def build_trinomial(
    nb_steps: int
) -> TrinomialTree:
    """Return a full trinomial lattice for the reference market case."""
    return TrinomialTree(
        spot=SPOT,
        rate=RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
        nb_steps=nb_steps,
        dividends=DIVIDENDS,
    )


def count_reachable_nodes(
    root: Node
) -> int:
    """Count unique priced nodes, retaining the public benchmark API."""
    return _count_priced_nodes(root)

def price_hybrid(
    nb_steps: int
) -> float:
    """Return the hybrid European call price."""
    option = CallOption(
        strike=STRIKE
    )

    return RecursivePricer().price(
        build_hybrid(
            nb_steps
        ),
        option,
    )


def price_trinomial(
    nb_steps: int
) -> float:
    """Return the full trinomial European call price."""
    option = CallOption(
        strike=STRIKE
    )

    return RecursivePricer().price(
        build_trinomial(
            nb_steps
        ),
        option,
    )


def measure_runtime(
    nb_steps: int,
    use_hybrid: bool
) -> float:
    """Return the median pricing runtime."""
    durations: list[float] = []

    for _ in range(
        NB_RUNTIME_RUNS
    ):
        start_time = (
            time.perf_counter()
        )

        if use_hybrid:
            price_hybrid(
                nb_steps
            )
        else:
            price_trinomial(
                nb_steps
            )

        # Measure execution time outside the node-counting traversal.
        durations.append(
            time.perf_counter()
            - start_time
        )

    return statistics.median(
        durations
    )


def measure_node_count(
    nb_steps: int,
    use_hybrid: bool
) -> int:
    """Return the number of reachable nodes in one built lattice."""
    if use_hybrid:
        tree = build_hybrid(
            nb_steps
        )
    else:
        tree = build_trinomial(
            nb_steps
        )

    tree.build()

    return count_reachable_nodes(
        tree.root
    )


def run_experiment() -> list[dict[str, float | int]]:
    """Compare hybrid and full trinomial pricing costs."""
    results: list[
        dict[str, float | int]
    ] = []

    for nb_steps in STEP_COUNTS:
        hybrid_price = price_hybrid(
            nb_steps
        )

        trinomial_price = price_trinomial(
            nb_steps
        )

        price_gap = (
            hybrid_price
            - trinomial_price
        )

        # Run each predefined scenario without changing the comparison baseline.
        absolute_gap = abs(
            price_gap
        )

        relative_gap = (
            absolute_gap
            / trinomial_price
        )

        hybrid_runtime = measure_runtime(
            nb_steps,
            use_hybrid=True,
        )

        trinomial_runtime = measure_runtime(
            nb_steps,
            # Keep absolute price gaps distinct from runtime and allocation ratios.
            use_hybrid=False,
        )

        hybrid_nodes = measure_node_count(
            nb_steps,
            use_hybrid=True,
        )

        trinomial_nodes = measure_node_count(
            nb_steps,
            use_hybrid=False,
        )

        speedup = (
            trinomial_runtime
            / hybrid_runtime
        )

        # Use identical discrete-dividend contracts for both lattice structures.
        node_ratio = (
            hybrid_nodes
            / trinomial_nodes
        )

        results.append(
            {
                "nb_steps": nb_steps,
                "hybrid_price": hybrid_price,
                "trinomial_price": trinomial_price,
                "price_gap": price_gap,
                "absolute_gap": absolute_gap,
                "relative_gap": relative_gap,
                "hybrid_runtime": hybrid_runtime,
                "trinomial_runtime": trinomial_runtime,
                "speedup": speedup,
                "hybrid_nodes": hybrid_nodes,
                # Time price construction separately from graph-size measurements.
                "trinomial_nodes": trinomial_nodes,
                "node_ratio": node_ratio,
            }
        )

        print(
            f"N={nb_steps:>3} | "
            f"Hybrid={hybrid_price:.8f} | "
            f"Tri={trinomial_price:.8f} | "
            f"Gap={price_gap:+.8f} | "
            f"Rel={100.0 * relative_gap:.4f}% | "
            f"Speedup={speedup:.2f}x | "
            f"Nodes={hybrid_nodes}/{trinomial_nodes}"
        )

    return results


def save_csv(
    results: list[dict[str, float | int]]
) -> None:
    """Save benchmark data."""
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "nb_steps",
                "hybrid_price",
                "trinomial_price",
                "price_gap",
                # Compare reachable-node counts instead of theoretical grid dimensions.
                "absolute_gap",
                "relative_gap",
                "hybrid_runtime",
                "trinomial_runtime",
                "speedup",
                "hybrid_nodes",
                "trinomial_nodes",
                "node_ratio",
            ],
        )

        writer.writeheader()

        writer.writerows(
            results
        )


def save_price_gap_figure(
    results: list[dict[str, float | int]]
) -> None:
    """Save the absolute pricing gap."""
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    steps = [
        int(result["nb_steps"])
        for result in results
    ]

    gaps = [
        float(result["absolute_gap"])
        for result in results
    ]

    plt.figure(
        # Keep absolute price gaps distinct from runtime and allocation ratios.
        figsize=(9, 5)
    )

    plt.plot(
        steps,
        gaps,
        marker="o",
    )

    plt.xlabel(
        "Number of steps"
    )

    plt.ylabel(
        "Absolute price gap"
    )

    # Use identical discrete-dividend contracts for both lattice structures.
    plt.title(
        "Hybrid versus full trinomial pricing gap"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PRICE_GAP_FIGURE_PATH,
        dpi=200,
    )

    plt.close()


def save_runtime_figure(
    results: list[dict[str, float | int]]
) -> None:
    """Save runtime comparison."""
    steps = [
        int(result["nb_steps"])
        for result in results
    ]

    hybrid_times = [
        float(result["hybrid_runtime"])
        for result in results
    ]

    trinomial_times = [
        float(result["trinomial_runtime"])
        for result in results
    ]

    plt.figure(
        # Time price construction separately from graph-size measurements.
        figsize=(9, 5)
    )

    plt.plot(
        steps,
        hybrid_times,
        marker="o",
        label="Hybrid",
    )

    plt.plot(
        steps,
        trinomial_times,
        marker="o",
        label="Full trinomial",
    )

    # Compare reachable-node counts instead of theoretical grid dimensions.
    plt.xlabel(
        "Number of steps"
    )

    plt.ylabel(
        "Pricing time (seconds)"
    )

    plt.title(
        "Hybrid versus full trinomial runtime"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RUNTIME_FIGURE_PATH,
        dpi=200,
    )

    plt.close()


def save_node_figure(
    results: list[dict[str, float | int]]
) -> None:
    """Save node-count comparison."""
    steps = [
        int(result["nb_steps"])
        for result in results
    ]

    hybrid_nodes = [
        int(result["hybrid_nodes"])
        for result in results
    ]

    trinomial_nodes = [
        int(result["trinomial_nodes"])
        for result in results
    ]

    plt.figure(
        # Keep absolute price gaps distinct from runtime and allocation ratios.
        figsize=(9, 5)
    )

    plt.plot(
        steps,
        hybrid_nodes,
        marker="o",
        label="Hybrid",
    )

    plt.plot(
        steps,
        trinomial_nodes,
        marker="o",
        label="Full trinomial",
    )

    # Use identical discrete-dividend contracts for both lattice structures.
    plt.xlabel(
        "Number of steps"
    )

    plt.ylabel(
    "Reachable pricing nodes"
    )

    plt.title(
        "Hybrid versus full trinomial reachable pricing nodes"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        NODE_FIGURE_PATH,
        dpi=200,
    )

    plt.close()


def save_summary(
    results: list[dict[str, float | int]]
) -> None:
    """Save the main benchmark conclusions."""
    content = _summary_text(results[-1])

    SUMMARY_PATH.write_text(
        content,
        encoding="utf-8",
    )


def _summary_text(last_result: dict[str, float | int]) -> str:
    """Format the same benchmark summary without any I/O."""
    # Time price construction separately from graph-size measurements.
    return (
        "Hybrid versus full trinomial benchmark\n"
        "=======================================\n\n"
        f"Spot: {SPOT}\n"
        f"Strike: {STRIKE}\n"
        f"Rate: {RATE}\n"
        f"Volatility: {VOLATILITY}\n"
        f"Maturity: {MATURITY}\n"
        f"Dividend: {DIVIDENDS[0].amount}\n"
        f"Dividend time: {DIVIDENDS[0].time}\n\n"
        f"Last number of steps: "
        f"{int(last_result['nb_steps'])}\n"
        f"Hybrid price: "
        f"{float(last_result['hybrid_price']):.8f}\n"
        f"Full trinomial price: "
        f"{float(last_result['trinomial_price']):.8f}\n"
        f"Absolute gap: "
        # Compare reachable-node counts instead of theoretical grid dimensions.
        f"{float(last_result['absolute_gap']):.8f}\n"
        f"Relative gap: "
        f"{100.0 * float(last_result['relative_gap']):.6f}%\n"
        f"Runtime speedup: "
        f"{float(last_result['speedup']):.2f}x\n"
        f"Hybrid nodes: "
        f"{int(last_result['hybrid_nodes'])}\n"
        f"Full trinomial nodes: "
        f"{int(last_result['trinomial_nodes'])}\n"
        f"Node ratio: "
        f"{100.0 * float(last_result['node_ratio']):.2f}%\n"
    )


def _save_comparison_figures(results: list[dict[str, float | int]]) -> None:
    """Create all three benchmark figures for the same observations."""
    save_price_gap_figure(
        results
    )

    save_runtime_figure(
        results
    )

    save_node_figure(
        results
    )



def _print_output_paths() -> None:
    """Display the paths of all five output artifacts."""
    print()
    print("Results saved to:")

    print(
        CSV_PATH.relative_to(
            PROJECT_ROOT
        )
    )

    print(
        PRICE_GAP_FIGURE_PATH.relative_to(
            # Use identical discrete-dividend contracts for both lattice structures.
            PROJECT_ROOT
        )
    )

    print(
        RUNTIME_FIGURE_PATH.relative_to(
            PROJECT_ROOT
        )
    )

    print(
        NODE_FIGURE_PATH.relative_to(
            PROJECT_ROOT
        )
    )

    print(
        SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        )
    )



def main() -> None:
    """Run and save the hybrid benchmark."""
    results = run_experiment()

    save_csv(
        results
    )

    _save_comparison_figures(results)

    # Keep absolute price gaps distinct from runtime and allocation ratios.
    save_summary(
        results
    )

    _print_output_paths()

if __name__ == "__main__":
    main()
