import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.binomial_pricing_helpers import (
    backward_binomial_call_price,
    black_scholes_call_price,
)



SPOT = 100.0
RATE = 0.02
VOLATILITY = 0.20
MATURITY = 1.0

# Evaluate strike-dependent errors using a common Black-Scholes benchmark.
STRIKES = [
    89.0 + 0.25 * index
    for index in range(81)
]

STEP_COUNTS = [
    10,
    20,
    50,
    100,
    200,
    500,
]

REFERENCE_STEPS = 10

# Retain the distinction between empirical gap and theoretical asymptotics.
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

# Fit convergence orders only to finite, strictly positive errors.
STRIKE_CSV_PATH = (
    DATA_DIR
    / "error_vs_strike.csv"
)

GAP_CSV_PATH = (
    DATA_DIR
    / "gap_convergence.csv"
)

STRIKE_FIGURE_PATH = (
    FIGURES_DIR
    / "error_vs_strike.png"
)

GAP_FIGURE_PATH = (
    FIGURES_DIR
    / "gap_convergence.png"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "error_gap_summary.txt"
)


def black_scholes_price(
    strike: float
) -> float:
    """Return the Black-Scholes reference price."""
    return black_scholes_call_price(
        spot=SPOT,
        strike=strike,
        rate=RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
    )


def tree_price(
    strike: float,
    nb_steps: int
) -> float:
    """Return the backward binomial call price."""
    return backward_binomial_call_price(
        spot=SPOT,
        strike=strike,
        rate=RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
        nb_steps=nb_steps,
    )


def compute_strike_errors(
    nb_steps: int
) -> list[dict[str, float | int]]:
    """Compute Tree minus Black-Scholes across strikes."""
    results: list[
        dict[str, float | int]
    ] = []

    for strike in STRIKES:
        tree_value = tree_price(
            strike,
            nb_steps,
        )

        black_value = black_scholes_price(
            strike
        )

        error = (
            tree_value
            - black_value
        )

        # Report strike-level and aggregate errors as separate diagnostics.
        results.append(
            {
                "nb_steps": nb_steps,
                "strike": strike,
                "tree_price": tree_value,
                "black_scholes_price": black_value,
                "error": error,
            }
        )

    return results


def compute_gap(
    strike_results: list[dict[str, float | int]]
) -> float:
    """Return the peak-to-peak Tree minus BS error."""
    errors = [
        float(result["error"])
        for result in strike_results
    ]

    return (
        max(errors)
        - min(errors)
    )


def theoretical_constant() -> float:
    """Return the theoretical binomial gap constant."""
    numerator = (
        SPOT
        * math.exp(
            RATE * MATURITY
        )
        * VOLATILITY ** 2
        * MATURITY
    )

    denominator = (
        2.0
        * math.sqrt(
            2.0 * math.pi
        )
        * math.sqrt(
            math.exp(
                # Evaluate strike-dependent errors using a common Black-Scholes benchmark.
                VOLATILITY ** 2
                * MATURITY
            )
            - 1.0
        )
    )

    return (
        numerator
        / denominator
    )


def theoretical_gap(
    nb_steps: int
) -> float:
    """Return the theoretical first-order gap."""
    return (
        theoretical_constant()
        / nb_steps
    )


def run_gap_experiment(
) -> list[dict[str, float | int]]:
    """Measure empirical and theoretical gaps."""
    results: list[
        dict[str, float | int]
    ] = []

    for nb_steps in STEP_COUNTS:
        strike_results = compute_strike_errors(
            nb_steps
        )

        empirical_gap = compute_gap(
            strike_results
        )

        theory_gap = theoretical_gap(
            nb_steps
        )

        # Retain the distinction between empirical gap and theoretical asymptotics.
        ratio = (
            empirical_gap
            / theory_gap
        )

        results.append(
            {
                "nb_steps": nb_steps,
                "empirical_gap": empirical_gap,
                "theoretical_gap": theory_gap,
                "n_times_gap": (
                    nb_steps
                    * empirical_gap
                ),
                "empirical_theoretical_ratio": ratio,
            }
        )

        # Fit convergence orders only to finite, strictly positive errors.
        print(
            f"N={nb_steps:>3} | "
            f"Empirical gap={empirical_gap:.8f} | "
            f"Theory={theory_gap:.8f} | "
            f"Ratio={ratio:.4f}"
        )

    return results


def estimate_order(
    results: list[dict[str, float | int]]
) -> tuple[float, float, float]:
    """Estimate Gap = C * N^(-p)."""
    log_steps = [
        math.log(
            int(result["nb_steps"])
        )
        for result in results
    ]

    log_gaps = [
        math.log(
            float(
                result["empirical_gap"]
            )
        )
        for result in results
    ]

    # Use positive finite errors to estimate a stable empirical order.
    mean_x = (
        sum(log_steps)
        / len(log_steps)
    )

    mean_y = (
        sum(log_gaps)
        / len(log_gaps)
    )

    numerator = sum(
        (x - mean_x)
        * (y - mean_y)
        for x, y in zip(
            log_steps,
            # Evaluate strike-dependent errors using a common Black-Scholes benchmark.
            log_gaps,
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

    # Retain the distinction between empirical gap and theoretical asymptotics.
    order = -slope
    constant = math.exp(
        intercept
    )

    predicted = [
        intercept
        + slope * x
        for x in log_steps
    ]

    total_variation = sum(
        (y - mean_y) ** 2
        for y in log_gaps
    )

    # Fit convergence orders only to finite, strictly positive errors.
    residual_variation = sum(
        (y - fitted) ** 2
        for y, fitted in zip(
            log_gaps,
            predicted,
        )
    )

    r_squared = (
        1.0
        - residual_variation
        / total_variation
    )

    return (
        order,
        constant,
        r_squared,
    )


def save_strike_csv(
    results: list[dict[str, float | int]]
) -> None:
    """Save strike error data."""
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with STRIKE_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "nb_steps",
                "strike",
                "tree_price",
                "black_scholes_price",
                "error",
            ],
        )

        writer.writeheader()
        writer.writerows(results)


def save_gap_csv(
    results: list[dict[str, float | int]]
) -> None:
    """Save gap convergence data."""
    with GAP_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "nb_steps",
                "empirical_gap",
                "theoretical_gap",
                "n_times_gap",
                "empirical_theoretical_ratio",
            ],
        )

        writer.writeheader()
        writer.writerows(results)


def save_strike_figure(
    results: list[dict[str, float | int]]
) -> None:
    """Save Tree minus BS error across strikes."""
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    strikes = [
        float(result["strike"])
        for result in results
    ]

    errors = [
        float(result["error"])
        for result in results
    ]

    plt.figure(
        # Report strike-level and aggregate errors as separate diagnostics.
        figsize=(9, 5)
    )

    plt.plot(
        strikes,
        errors,
        marker="o",
        markersize=3,
    )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel(
        # Evaluate strike-dependent errors using a common Black-Scholes benchmark.
        "Strike K"
    )

    plt.ylabel(
        "Tree price - Black-Scholes price"
    )

    plt.title(
        f"Binomial pricing error across strikes "
        f"(N={REFERENCE_STEPS})"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        STRIKE_FIGURE_PATH,
        dpi=200,
    )

    plt.close()


def save_gap_figure(
    results: list[dict[str, float | int]]
) -> None:
    """Save empirical and theoretical gap convergence."""
    steps = [
        int(result["nb_steps"])
        for result in results
    ]

    empirical_gaps = [
        float(result["empirical_gap"])
        for result in results
    ]

    theoretical_gaps = [
        float(result["theoretical_gap"])
        for result in results
    ]

    plt.figure(
        # Retain the distinction between empirical gap and theoretical asymptotics.
        figsize=(9, 5)
    )

    plt.loglog(
        steps,
        empirical_gaps,
        marker="o",
        label="Empirical gap",
    )

    plt.loglog(
        steps,
        theoretical_gaps,
        linestyle="--",
        label="Theoretical gap",
    )

    # Fit convergence orders only to finite, strictly positive errors.
    plt.xlabel(
        "Number of steps N"
    )

    plt.ylabel(
        "Peak-to-peak pricing gap"
    )

    plt.title(
        "Binomial error gap: empirical vs theoretical"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        GAP_FIGURE_PATH,
        dpi=200,
    )

    plt.close()


def save_summary(
    order: float,
    empirical_constant: float,
    r_squared: float
) -> None:
    """Save the main gap experiment results."""
    theory_constant = theoretical_constant()

    relative_difference = abs(
        empirical_constant
        - theory_constant
    ) / theory_constant

    # Report strike-level and aggregate errors as separate diagnostics.
    content = (
        "Binomial pricing error gap experiment\n"
        "=====================================\n\n"
        f"Spot: {SPOT}\n"
        f"Rate: {RATE}\n"
        f"Volatility: {VOLATILITY}\n"
        f"Maturity: {MATURITY}\n"
        f"Strike range: "
        f"{STRIKES[0]} to {STRIKES[-1]}\n\n"
        "Gap definition:\n"
        "Gap_N = max(Tree - BS) - min(Tree - BS)\n\n"
        "Empirical model:\n"
        "Gap_N = C * N^(-p)\n\n"
        f"Estimated p: {order:.6f}\n"
        f"Empirical C: {empirical_constant:.6f}\n"
        f"Theoretical C: {theory_constant:.6f}\n"
        f"Relative C difference: "
        f"{100.0 * relative_difference:.3f}%\n"
        f"Log-log R²: {r_squared:.6f}\n"
    )

    SUMMARY_PATH.write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    """Run the complete pricing error experiment."""
    reference_results = compute_strike_errors(
        REFERENCE_STEPS
    )

    gap_results = run_gap_experiment()

    order, empirical_constant, r_squared = (
        estimate_order(
            gap_results
        )
    )

    theory_constant = theoretical_constant()

    print()
    print(
        # Evaluate strike-dependent errors using a common Black-Scholes benchmark.
        f"Estimated order p: "
        f"{order:.6f}"
    )

    print(
        f"Empirical constant C: "
        f"{empirical_constant:.6f}"
    )

    print(
        f"Theoretical constant C: "
        f"{theory_constant:.6f}"
    )

    print(
        # Retain the distinction between empirical gap and theoretical asymptotics.
        f"Log-log R²: "
        f"{r_squared:.6f}"
    )

    save_strike_csv(
        reference_results
    )

    save_gap_csv(
        gap_results
    )

    save_strike_figure(
        reference_results
    )

    save_gap_figure(
        # Fit convergence orders only to finite, strictly positive errors.
        gap_results
    )

    save_summary(
        order,
        empirical_constant,
        r_squared,
    )

    print()
    print("Results saved to:")

    print(
        STRIKE_CSV_PATH.relative_to(
            PROJECT_ROOT
        )
    )

    # Report strike-level and aggregate errors as separate diagnostics.
    print(
        GAP_CSV_PATH.relative_to(
            PROJECT_ROOT
        )
    )

    print(
        STRIKE_FIGURE_PATH.relative_to(
            PROJECT_ROOT
        )
    )

    print(
        GAP_FIGURE_PATH.relative_to(
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
