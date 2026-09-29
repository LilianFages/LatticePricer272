import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

from src.backward_pricer import BackwardPricer
from src.binomial_tree import BinomialTree
from src.black_scholes import BlackScholesPricer
from src.option import CallOption


SPOT = 100.0
RATE = 0.02
VOLATILITY = 0.20
MATURITY = 1.0

STEP_COUNTS = [
    20,
    30,
    50,
    75,
    100,
    150,
    200,
    300,
    400,
    500,
]

STRIKES = [
    90.0 + 0.25 * index
    for index in range(81)
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
    / "convergence_order.csv"
)

FIGURE_PATH = (
    FIGURES_DIR
    / "convergence_order.png"
)

SCALED_FIGURE_PATH = (
    FIGURES_DIR
    / "scaled_convergence_error.png"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "convergence_order_summary.txt"
)


def black_scholes_price(
    strike: float,
) -> float:
    """Return the Black-Scholes reference price."""
    option = CallOption(
        strike=strike
    )

    return BlackScholesPricer().price(
        option=option,
        spot=SPOT,
        rate=RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
    )


def tree_price(
    nb_steps: int,
    strike: float,
) -> float:
    """Return the backward lattice price."""
    tree = BinomialTree(
        spot=SPOT,
        rate=RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
        nb_steps=nb_steps,
    )

    option = CallOption(
        strike=strike
    )

    return BackwardPricer().price(
        tree,
        option,
    )


def compute_max_error(
    nb_steps: int,
) -> tuple[float, float]:
    """Return the maximum pricing error across strikes."""
    max_error = 0.0
    max_error_strike = STRIKES[0]

    for strike in STRIKES:
        tree_value = tree_price(
            nb_steps,
            strike,
        )

        black_value = black_scholes_price(
            strike
        )

        absolute_error = abs(
            tree_value
            - black_value
        )

        if absolute_error > max_error:
            max_error = absolute_error
            max_error_strike = strike

    return (
        max_error,
        max_error_strike,
    )


def run_experiment() -> list[dict[str, float | int]]:
    """Measure the convergence error envelope."""
    results: list[
        dict[str, float | int]
    ] = []

    for nb_steps in STEP_COUNTS:
        max_error, max_error_strike = (
            compute_max_error(
                nb_steps
            )
        )

        scaled_error = (
            nb_steps
            * max_error
        )

        results.append(
            {
                "nb_steps": nb_steps,
                "max_absolute_error": max_error,
                "max_error_strike": max_error_strike,
                "n_times_max_error": scaled_error,
            }
        )

        print(
            f"N={nb_steps:>3} | "
            f"Max error={max_error:.8f} | "
            f"K={max_error_strike:.2f} | "
            f"N*Error={scaled_error:.8f}"
        )

    return results


def estimate_convergence_order(
    results: list[dict[str, float | int]],
) -> tuple[float, float, float]:
    """Estimate E_N = C * N^(-p)."""
    log_steps = [
        math.log(
            int(result["nb_steps"])
        )
        for result in results
    ]

    log_errors = [
        math.log(
            float(
                result["max_absolute_error"]
            )
        )
        for result in results
    ]

    mean_x = (
        sum(log_steps)
        / len(log_steps)
    )

    mean_y = (
        sum(log_errors)
        / len(log_errors)
    )

    numerator = sum(
        (x - mean_x)
        * (y - mean_y)
        for x, y in zip(
            log_steps,
            log_errors,
        )
    )

    denominator = sum(
        (x - mean_x) ** 2
        for x in log_steps
    )

    slope = (
        numerator
        / denominator
    )

    intercept = (
        mean_y
        - slope * mean_x
    )

    order = -slope

    predicted_values = [
        intercept
        + slope * x
        for x in log_steps
    ]

    total_variation = sum(
        (y - mean_y) ** 2
        for y in log_errors
    )

    residual_variation = sum(
        (y - predicted) ** 2
        for y, predicted in zip(
            log_errors,
            predicted_values,
        )
    )

    r_squared = (
        1.0
        - residual_variation
        / total_variation
    )

    constant = math.exp(
        intercept
    )

    return (
        order,
        constant,
        r_squared,
    )


def save_csv(
    results: list[dict[str, float | int]],
) -> None:
    """Save convergence-order data."""
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
                "max_absolute_error",
                "max_error_strike",
                "n_times_max_error",
            ],
        )

        writer.writeheader()
        writer.writerows(results)


def save_log_log_figure(
    results: list[dict[str, float | int]],
    order: float,
    constant: float,
) -> None:
    """Save the log-log convergence graph."""
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    steps = [
        int(result["nb_steps"])
        for result in results
    ]

    errors = [
        float(
            result["max_absolute_error"]
        )
        for result in results
    ]

    fitted_errors = [
        constant
        * nb_steps ** (-order)
        for nb_steps in steps
    ]

    plt.figure(
        figsize=(9, 5)
    )

    plt.loglog(
        steps,
        errors,
        marker="o",
        label="Maximum observed error",
    )

    plt.loglog(
        steps,
        fitted_errors,
        linestyle="--",
        label=f"Fit: C N^(-{order:.3f})",
    )

    plt.xlabel(
        "Number of steps N"
    )

    plt.ylabel(
        "Maximum absolute pricing error"
    )

    plt.title(
        "Binomial empirical convergence order"
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


def save_scaled_error_figure(
    results: list[dict[str, float | int]],
) -> None:
    """Save N times the maximum pricing error."""
    steps = [
        int(result["nb_steps"])
        for result in results
    ]

    scaled_errors = [
        float(
            result["n_times_max_error"]
        )
        for result in results
    ]

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        steps,
        scaled_errors,
        marker="o",
    )

    plt.xlabel(
        "Number of steps N"
    )

    plt.ylabel(
        "N × maximum absolute error"
    )

    plt.title(
        "First-order convergence diagnostic"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        SCALED_FIGURE_PATH,
        dpi=200,
    )

    plt.close()


def save_summary(
    order: float,
    constant: float,
    r_squared: float,
) -> None:
    """Save the estimated convergence order."""
    content = (
        "Binomial convergence order experiment\n"
        "=====================================\n\n"
        f"Spot: {SPOT}\n"
        f"Rate: {RATE}\n"
        f"Volatility: {VOLATILITY}\n"
        f"Maturity: {MATURITY}\n"
        f"Strike range: "
        f"{STRIKES[0]} to {STRIKES[-1]}\n\n"
        "Error measure:\n"
        "E_N = max_K |Tree(N, K) - BlackScholes(K)|\n\n"
        "Model:\n"
        "E_N = C * N^(-p)\n\n"
        f"Estimated p: {order:.6f}\n"
        f"Estimated C: {constant:.6f}\n"
        f"Log-log R²: {r_squared:.6f}\n\n"
        "Theoretical first-order target: p = 1\n"
    )

    SUMMARY_PATH.write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    """Run and save the convergence-order experiment."""
    results = run_experiment()

    order, constant, r_squared = (
        estimate_convergence_order(
            results
        )
    )

    print()
    print(
        f"Estimated convergence order p: "
        f"{order:.6f}"
    )

    print(
        f"Estimated constant C: "
        f"{constant:.6f}"
    )

    print(
        f"Log-log R²: "
        f"{r_squared:.6f}"
    )

    save_csv(
        results
    )

    save_log_log_figure(
        results,
        order,
        constant,
    )

    save_scaled_error_figure(
        results
    )

    save_summary(
        order,
        constant,
        r_squared,
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
        SCALED_FIGURE_PATH.relative_to(
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