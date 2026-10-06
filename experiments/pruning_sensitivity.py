import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt

from src.dividend import CashDividend
from src.option import CallOption
from src.recursive_pricer import RecursivePricer
from src.trinomial_tree import TrinomialTree


SPOT = 100.0
STRIKE = 100.0
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
    50,
    100,
    200,
    300,
]

PRUNING_THRESHOLDS = [
    0.0,
    1e-18,
    1e-15,
    1e-12,
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

DATA_DIR = (
    RESULTS_DIR
    / "data"
)

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


def run_experiment(
) -> list[dict[str, float | int | str | None]]:
    """Measure price sensitivity to the pruning threshold."""
    results: list[
        dict[str, float | int | str | None]
    ] = []

    for nb_steps in STEP_COUNTS:
        baseline_price: float | None = None

        for threshold in PRUNING_THRESHOLDS:
            result = run_case(
                nb_steps,
                threshold,
            )

            price_value = result["price"]

            if (
                threshold == 0.0
                and isinstance(
                    price_value,
                    float,
                )
            ):
                baseline_price = (
                    price_value
                )

            if (
                baseline_price is not None
                and isinstance(
                    price_value,
                    float,
                )
            ):
                price_gap = (
                    price_value
                    - baseline_price
                )

                relative_gap = (
                    abs(price_gap)
                    / baseline_price
                )

            else:
                price_gap = None
                relative_gap = None

            result["price_gap"] = (
                price_gap
            )

            result["relative_gap"] = (
                relative_gap
            )

            results.append(
                result
            )

            if result["status"] == "success":
                price = float(
                    result["price"]
                )

                pruned_nodes = int(
                    result["pruned_nodes"]
                )

                runtime = float(
                    result["runtime"]
                )

                if price_gap is None:
                    gap_text = "n/a"
                else:
                    gap_text = (
                        f"{price_gap:+.3e}"
                    )

                print(
                    f"N={nb_steps:>3} | "
                    f"Threshold={threshold:.0e} | "
                    f"Price={price:.10f} | "
                    f"Gap={gap_text} | "
                    f"Pruned={pruned_nodes:>5} | "
                    f"Time={runtime:.5f}s"
                )

            else:
                print(
                    f"N={nb_steps:>3} | "
                    f"Threshold={threshold:.0e} | "
                    f"FAILED | "
                    f"Pruned={int(result['pruned_nodes'])} | "
                    f"{result['error']}"
                )

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


def save_price_impact_figure(
    results: list[dict[str, float | int | str | None]]
) -> None:
    """Save the pricing impact of each pruning threshold."""
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_labels = [
        "0",
        "1e-18",
        "1e-15",
        "1e-12",
    ]

    plt.figure(
        figsize=(9, 5)
    )

    for nb_steps in [
        50,
        100,
        200,
    ]:
        gaps: list[float] = []

        for threshold in PRUNING_THRESHOLDS:
            matching_result = next(
                result
                for result in results
                if (
                    result["nb_steps"] == nb_steps
                    and result["threshold"] == threshold
                )
            )

            gap = matching_result[
                "price_gap"
            ]

            if gap is None:
                gaps.append(
                    float("nan")
                )
            else:
                gaps.append(
                    abs(
                        float(gap)
                    )
                )

        plt.plot(
            threshold_labels,
            gaps,
            marker="o",
            label=f"N={nb_steps}",
        )

    plt.xlabel(
        "Pruning threshold"
    )

    plt.ylabel(
        "Absolute price impact versus no pruning"
    )

    plt.title(
        "Pricing sensitivity to the pruning threshold"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PRICE_IMPACT_FIGURE_PATH,
        dpi=200,
    )

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