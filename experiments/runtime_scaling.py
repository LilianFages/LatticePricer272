import csv
import gc
import math
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt

from src.backward_pricer import BackwardPricer
from src.binomial_tree import BinomialTree
from src.option import CallOption
from src.recursive_pricer import RecursivePricer


SPOT = 100.0
STRIKE = 100.0
RATE = 0.02
VOLATILITY = 0.20
MATURITY = 1.0

STEP_COUNTS = [
    10,
    20,
    50,
    100,
    200,
    400,
    600,
    800,
    900,
]

NB_RUNS = 5
FIT_MIN_STEPS = 50

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
    / "runtime_scaling.csv"
)

FIGURE_PATH = (
    FIGURES_DIR
    / "runtime_scaling_loglog.png"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "runtime_scaling_summary.txt"
)


def build_tree(
    nb_steps: int,
) -> BinomialTree:
    """Return a fresh binomial tree."""
    return BinomialTree(
        spot=SPOT,
        rate=RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
        nb_steps=nb_steps,
    )


def measure_recursive(
    nb_steps: int,
) -> float:
    """Measure one recursive pricing run."""
    tree = build_tree(
        nb_steps
    )

    option = CallOption(
        strike=STRIKE
    )

    start = time.perf_counter()

    RecursivePricer().price(
        tree,
        option,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    del tree
    gc.collect()

    return elapsed


def measure_backward(
    nb_steps: int,
) -> float:
    """Measure one backward pricing run."""
    tree = build_tree(
        nb_steps
    )

    option = CallOption(
        strike=STRIKE
    )

    start = time.perf_counter()

    BackwardPricer().price(
        tree,
        option,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    del tree
    gc.collect()

    return elapsed


def median_runtime(
    nb_steps: int,
    method: str,
) -> float:
    """Return the median runtime over several runs."""
    times: list[float] = []

    for _ in range(NB_RUNS):
        if method == "recursive":
            elapsed = measure_recursive(
                nb_steps
            )
        elif method == "backward":
            elapsed = measure_backward(
                nb_steps
            )
        else:
            raise ValueError(
                "Unknown pricing method."
            )

        times.append(
            elapsed
        )

    return statistics.median(
        times
    )


def run_experiment() -> list[dict[str, float | int]]:
    """Measure runtime scaling of both pricers."""
    results: list[
        dict[str, float | int]
    ] = []

    for nb_steps in STEP_COUNTS:
        recursive_time = median_runtime(
            nb_steps,
            "recursive",
        )

        backward_time = median_runtime(
            nb_steps,
            "backward",
        )

        speed_ratio = (
            recursive_time
            / backward_time
        )

        results.append(
            {
                "nb_steps": nb_steps,
                "recursive_seconds": recursive_time,
                "backward_seconds": backward_time,
                "recursive_backward_ratio": speed_ratio,
            }
        )

        print(
            f"N={nb_steps:>3} | "
            f"Recursive={recursive_time:.6f} s | "
            f"Backward={backward_time:.6f} s | "
            f"Ratio={speed_ratio:.3f}"
        )

    return results


def estimate_order(
    results: list[dict[str, float | int]],
    key: str,
) -> tuple[float, float]:
    """Estimate runtime = C * N^q."""
    selected = [
        result
        for result in results
        if int(result["nb_steps"])
        >= FIT_MIN_STEPS
    ]

    x_values = [
        math.log(
            int(result["nb_steps"])
        )
        for result in selected
    ]

    y_values = [
        math.log(
            float(result[key])
        )
        for result in selected
    ]

    mean_x = sum(
        x_values
    ) / len(x_values)

    mean_y = sum(
        y_values
    ) / len(y_values)

    numerator = sum(
        (x - mean_x)
        * (y - mean_y)
        for x, y in zip(
            x_values,
            y_values,
        )
    )

    denominator = sum(
        (x - mean_x) ** 2
        for x in x_values
    )

    slope = (
        numerator
        / denominator
    )

    intercept = (
        mean_y
        - slope * mean_x
    )

    return (
        slope,
        math.exp(intercept),
    )


def save_csv(
    results: list[dict[str, float | int]],
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
                "recursive_seconds",
                "backward_seconds",
                "recursive_backward_ratio",
            ],
        )

        writer.writeheader()
        writer.writerows(results)


def save_figure(
    results: list[dict[str, float | int]],
) -> None:
    """Save the log-log runtime graph."""
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    steps = [
        int(result["nb_steps"])
        for result in results
    ]

    recursive_times = [
        float(
            result["recursive_seconds"]
        )
        for result in results
    ]

    backward_times = [
        float(
            result["backward_seconds"]
        )
        for result in results
    ]

    plt.figure(
        figsize=(9, 5)
    )

    plt.loglog(
        steps,
        recursive_times,
        marker="o",
        label="Recursive",
    )

    plt.loglog(
        steps,
        backward_times,
        marker="o",
        label="Backward",
    )

    plt.xlabel(
        "Number of steps N"
    )

    plt.ylabel(
        "Runtime (seconds)"
    )

    plt.title(
        "Pricing runtime versus number of steps"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH,
        dpi=200,
    )

    plt.close()


def save_summary(
    recursive_order: float,
    backward_order: float,
) -> None:
    """Save runtime scaling results."""
    content = (
        "Runtime scaling experiment\n"
        "==========================\n\n"
        f"Runs per point: {NB_RUNS}\n"
        f"Fit from N >= {FIT_MIN_STEPS}\n\n"
        "Model:\n"
        "runtime = C * N^q\n\n"
        f"Recursive exponent q: "
        f"{recursive_order:.6f}\n"
        f"Backward exponent q: "
        f"{backward_order:.6f}\n"
    )

    SUMMARY_PATH.write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    """Run and save the runtime benchmark."""
    print(
        f"Median over {NB_RUNS} runs per point."
    )

    print()

    results = run_experiment()

    recursive_order, _ = estimate_order(
        results,
        "recursive_seconds",
    )

    backward_order, _ = estimate_order(
        results,
        "backward_seconds",
    )

    print()
    print(
        f"Recursive runtime exponent q: "
        f"{recursive_order:.6f}"
    )

    print(
        f"Backward runtime exponent q: "
        f"{backward_order:.6f}"
    )

    save_csv(
        results
    )

    save_figure(
        results
    )

    save_summary(
        recursive_order,
        backward_order,
    )

    print()
    print("Results saved to:")

    print(
        CSV_PATH.relative_to(
            PROJECT_ROOT
        )
    )

    print(
        FIGURE_PATH.relative_to(
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