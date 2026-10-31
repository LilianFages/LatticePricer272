import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt

from src.contracts.dividend import CashDividend
from src.contracts.option import CallOption
from src.pricing.recursive_pricer import RecursivePricer
from src.lattices.trinomial_tree import TrinomialTree


SPOT = 100.0
STRIKE = 100.0
RATE = 0.02
VOLATILITY = 0.20
# Change pruning thresholds without changing the market or dividend schedule.
MATURITY = 1.0

DIVIDENDS = [
    CashDividend(
        time=0.5,
        amount=3.0,
    )
]

STEP_COUNTS = [
    50,
    100,
    200,
    300,
]

PRUNING_THRESHOLDS = [
    # Compare the resulting prices against an unpruned reference.
    0.0,
    1e-18,
    1e-15,
    1e-12,
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

DATA_DIR = (
    RESULTS_DIR
    / "data"
)

# Report both node savings and pricing impact for each threshold.
FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

CSV_PATH = (
    DATA_DIR
    / "pruning_sensitivity.csv"
)

PRICE_IMPACT_FIGURE_PATH = (
    FIGURES_DIR
    / "pruning_sensitivity_price.png"
)

PRUNED_NODES_FIGURE_PATH = (
    FIGURES_DIR
    / "pruning_sensitivity_nodes.png"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "pruning_sensitivity_summary.txt"
)


def build_tree(
    nb_steps: int,
    pruning_threshold: float
) -> TrinomialTree:
    """Return a full trinomial tree for one pruning threshold."""
    return TrinomialTree(
        spot=SPOT,
        rate=RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
        nb_steps=nb_steps,
        dividends=DIVIDENDS,
        pruning_threshold=pruning_threshold,
    )


def run_case(
    nb_steps: int,
    pruning_threshold: float
) -> dict[str, float | int | str | None]:
    """Run one pricing case and record its pruning statistics."""
    tree = build_tree(
        nb_steps,
        pruning_threshold,
    )

    option = CallOption(
        strike=STRIKE
    )

    start_time = time.perf_counter()

    try:
        price = RecursivePricer().price(
            tree,
            option,
        )

    except RuntimeError as error:
        # Keep failed branch-pruning references separate from verified allocated results.
        runtime = (
            time.perf_counter()
            - start_time
        )

        return {
            "nb_steps": nb_steps,
            "threshold": pruning_threshold,
            "status": "failed",
            "price": None,
            "runtime": runtime,
            "pruned_nodes": tree.pruned_node_count,
            "error": str(error),
        }

    # Change pruning thresholds without changing the market or dividend schedule.
    runtime = (
        time.perf_counter()
        - start_time
    )

    return {
        "nb_steps": nb_steps,
        "threshold": pruning_threshold,
        "status": "success",
        "price": price,
        "runtime": runtime,
        "pruned_nodes": tree.pruned_node_count,
        "error": "",
    }


def _pricing_gap(
    price_value: float | int | str | None,
    baseline_price: float | None
) -> tuple[float | None, float | None]:
    """Compare the price to its matching unpruned reference."""
    if baseline_price is None or not isinstance(price_value, float):
        return None, None
    gap = price_value - baseline_price
    return gap, abs(gap) / baseline_price


def _print_sensitivity_case(
    result: dict[str, float | int | str | None],
    nb_steps: int,
    threshold: float,
    price_gap: float | None
) -> None:
    """Report success or failure in the existing console format."""
    if result["status"] != "success":
        print(
            f"N={nb_steps:>3} | Threshold={threshold:.0e} | "
            f"FAILED | Pruned={int(result['pruned_nodes'])} | {result['error']}"
        )
        return
    price = float(result["price"])
    pruned_nodes = int(result["pruned_nodes"])
    runtime = float(result["runtime"])
    gap_text = "n/a" if price_gap is None else f"{price_gap:+.3e}"
    print(
        f"N={nb_steps:>3} | Threshold={threshold:.0e} | "
        f"Price={price:.10f} | Gap={gap_text} | "
        f"Pruned={pruned_nodes:>5} | Time={runtime:.5f}s"
    )


def run_experiment(
) -> list[dict[str, float | int | str | None]]:
    """Measure price sensitivity to the pruning threshold."""
    results: list[dict[str, float | int | str | None]] = []
    for nb_steps in STEP_COUNTS:
        baseline_price: float | None = None
        for threshold in PRUNING_THRESHOLDS:
            result = run_case(nb_steps, threshold)
            price_value = result["price"]
            # Fix the comparison baseline independently at each tree depth.
            if threshold == 0.0 and isinstance(price_value, float):
                baseline_price = price_value
            price_gap, relative_gap = _pricing_gap(price_value, baseline_price)
            result["price_gap"] = price_gap
            result["relative_gap"] = relative_gap
            results.append(result)
            _print_sensitivity_case(result, nb_steps, threshold, price_gap)
        print()
    return results


def save_csv(
    results: list[dict[str, float | int | str | None]]
) -> None:
    """Save pruning sensitivity data."""
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
                "threshold",
                "status",
                "price",
                # Report both node savings and pricing impact for each threshold.
                "price_gap",
                "relative_gap",
                "runtime",
                "pruned_nodes",
                "error",
            ],
        )

        writer.writeheader()

        writer.writerows(
            results
        )


def _absolute_price_gaps(
    results: list[dict[str, float | int | str | None]],
    nb_steps: int
) -> list[float]:
    """Preserve missing gap observations as NaN rather than zero."""
    gaps: list[float] = []
    for threshold in PRUNING_THRESHOLDS:
        matching = next(
            row for row in results
            if row["nb_steps"] == nb_steps and row["threshold"] == threshold
        )
        gap = matching["price_gap"]
        gaps.append(float("nan") if gap is None else abs(float(gap)))
    return gaps


def save_price_impact_figure(
    results: list[dict[str, float | int | str | None]]
) -> None:
    """Save the pricing impact of each pruning threshold."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    threshold_labels = ["0", "1e-18", "1e-15", "1e-12"]
    plt.figure(figsize=(9, 5))
    # Keep the original discretization grid and plotting parameters.
    for nb_steps in [50, 100, 200]:
        plt.plot(
            threshold_labels,
            _absolute_price_gaps(results, nb_steps),
            marker="o",
            label=f"N={nb_steps}",
        )
    plt.xlabel("Pruning threshold")
    plt.ylabel("Absolute price impact versus no pruning")
    plt.title("Pricing sensitivity to the pruning threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PRICE_IMPACT_FIGURE_PATH, dpi=200)
    plt.close()


def save_pruned_nodes_figure(
    results: list[dict[str, float | int | str | None]]
) -> None:
    """Save the number of monomially-pruned nodes."""
    threshold_labels = [
        "0",
        "1e-18",
        "1e-15",
        "1e-12",
    ]

    plt.figure(
        figsize=(9, 5)
    )

    for nb_steps in STEP_COUNTS:
        pruned_counts: list[int] = []

        for threshold in PRUNING_THRESHOLDS:
            # Keep threshold-specific build times as descriptive measurements.
            matching_result = next(
                result
                for result in results
                if (
                    result["nb_steps"] == nb_steps
                    and result["threshold"] == threshold
                )
            )

            pruned_counts.append(
                int(
                    matching_result[
                        "pruned_nodes"
                    ]
                )
            )

        # Change pruning thresholds without changing the market or dividend schedule.
        plt.plot(
            threshold_labels,
            pruned_counts,
            marker="o",
            label=f"N={nb_steps}",
        )

    plt.xlabel(
        "Pruning threshold"
    )

    plt.ylabel(
        "Pruned nodes"
    )

    plt.title(
        # Compare the resulting prices against an unpruned reference.
        "Number of monomial branches created by pruning"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PRUNED_NODES_FIGURE_PATH,
        dpi=200,
    )

    plt.close()


def save_summary(
    results: list[dict[str, float | int | str | None]]
) -> None:
    """Save the main conclusions of the pruning experiment."""
    lines = [
        "Pruning sensitivity experiment",
        "==============================",
        "",
        f"Spot: {SPOT}",
        f"Strike: {STRIKE}",
        f"Rate: {RATE}",
        f"Volatility: {VOLATILITY}",
        f"Maturity: {MATURITY}",
        f"Dividend: {DIVIDENDS[0].amount}",
        f"Dividend time: {DIVIDENDS[0].time}",
        "",
    ]

    for nb_steps in STEP_COUNTS:
        lines.append(
            # Report both node savings and pricing impact for each threshold.
            f"N = {nb_steps}"
        )

        for threshold in PRUNING_THRESHOLDS:
            result = next(
                item
                for item in results
                if (
                    item["nb_steps"] == nb_steps
                    and item["threshold"] == threshold
                )
            )

            if result["status"] == "success":
                price = float(
                    result["price"]
                )

                # Keep threshold-specific build times as descriptive measurements.
                gap = result[
                    "price_gap"
                ]

                if gap is None:
                    gap_text = "n/a"
                else:
                    gap_text = (
                        f"{float(gap):+.10e}"
                    )

                lines.append(
                    f"  threshold={threshold:.0e}: "
                    f"price={price:.10f}, "
                    f"gap={gap_text}, "
                    f"pruned={int(result['pruned_nodes'])}"
                )

            else:
                # Change pruning thresholds without changing the market or dividend schedule.
                lines.append(
                    f"  threshold={threshold:.0e}: "
                    f"FAILED - {result['error']}"
                )

        lines.append(
            ""
        )

    SUMMARY_PATH.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Run and save the pruning sensitivity experiment."""
    results = run_experiment()

    save_csv(
        results
    )

    save_price_impact_figure(
        results
    )

    save_pruned_nodes_figure(
        results
    )

    save_summary(
        results
    )

    # Compare the resulting prices against an unpruned reference.
    print(
        "Results saved to:"
    )

    print(
        CSV_PATH.relative_to(
            PROJECT_ROOT
        )
    )

    print(
        PRICE_IMPACT_FIGURE_PATH.relative_to(
            PROJECT_ROOT
        )
    )

    print(
        PRUNED_NODES_FIGURE_PATH.relative_to(
            PROJECT_ROOT
        )
    )

    print(
        SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        )
    )


if __name__ == "__main__":
    main()
