import csv
from pathlib import Path

import matplotlib.pyplot as plt

from src.binomial_tree import BinomialTree
from src.black_scholes import BlackScholesPricer
from src.option import CallOption
from src.recursive_pricer import RecursivePricer
from src.trinomial_tree import TrinomialTree


SPOT = 100.0
STRIKE = 100.0
RATE = 0.02
VOLATILITY = 0.20
MATURITY = 1.0

STEP_COUNTS = [
    1,
    2,
    3,
    5,
    10,
    20,
    50,
    100,
    200,
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
    / "trinomial_convergence.csv"
)

PRICE_FIGURE_PATH = (
    FIGURES_DIR
    / "trinomial_convergence_prices.png"
)

ERROR_FIGURE_PATH = (
    FIGURES_DIR
    / "trinomial_convergence_error.png"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "trinomial_convergence_summary.txt"
)


def black_scholes_price() -> float:
    """Return the Black-Scholes reference price."""
    option = CallOption(
        strike=STRIKE
    )

    return BlackScholesPricer().price(
        option=option,
        spot=SPOT,
        rate=RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
    )


def binomial_price(
    nb_steps: int
) -> float:
    """Return the recursive binomial price."""
    tree = BinomialTree(
        spot=SPOT,
        rate=RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
        nb_steps=nb_steps,
    )

    option = CallOption(
        strike=STRIKE
    )

    return RecursivePricer().price(
        tree,
        option,
    )


def trinomial_price(
    nb_steps: int
) -> float:
    """Return the recursive trinomial price."""
    tree = TrinomialTree(
        spot=SPOT,
        rate=RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
        nb_steps=nb_steps,
    )

    option = CallOption(
        strike=STRIKE
    )

    return RecursivePricer().price(
        tree,
        option,
    )


def run_experiment() -> list[dict[str, float | int]]:
    """Compare both lattices against Black-Scholes."""
    reference_price = (
        black_scholes_price()
    )

    results: list[
        dict[str, float | int]
    ] = []

    for nb_steps in STEP_COUNTS:
        bin_price = binomial_price(
            nb_steps
        )

        tri_price = trinomial_price(
            nb_steps
        )

        bin_error = (
            bin_price
            - reference_price
        )

        tri_error = (
            tri_price
            - reference_price
        )

        results.append(
            {
                "nb_steps": nb_steps,
                "binomial_price": bin_price,
                "trinomial_price": tri_price,
                "black_scholes_price": reference_price,
                "binomial_error": bin_error,
                "trinomial_error": tri_error,
                "binomial_absolute_error": abs(bin_error),
                "trinomial_absolute_error": abs(tri_error),
            }
        )

        print(
            f"N={nb_steps:>3} | "
            f"Bin={bin_price:.8f} | "
            f"Tri={tri_price:.8f} | "
            f"BS={reference_price:.8f} | "
            f"Bin error={bin_error:+.8f} | "
            f"Tri error={tri_error:+.8f}"
        )

    return results


def save_csv(
    results: list[dict[str, float | int]]
) -> None:
    """Save convergence data."""
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
                "binomial_price",
                "trinomial_price",
                "black_scholes_price",
                "binomial_error",
                "trinomial_error",
                "binomial_absolute_error",
                "trinomial_absolute_error",
            ],
        )

        writer.writeheader()
        writer.writerows(
            results
        )


def save_price_figure(
    results: list[dict[str, float | int]]
) -> None:
    """Save binomial, trinomial and Black-Scholes prices."""
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    steps = [
        int(result["nb_steps"])
        for result in results
    ]

    binomial_prices = [
        float(result["binomial_price"])
        for result in results
    ]

    trinomial_prices = [
        float(result["trinomial_price"])
        for result in results
    ]

    bs_prices = [
        float(result["black_scholes_price"])
        for result in results
    ]

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        steps,
        binomial_prices,
        marker="o",
        label="Binomial",
    )

    plt.plot(
        steps,
        trinomial_prices,
        marker="o",
        label="Trinomial",
    )

    plt.plot(
        steps,
        bs_prices,
        linestyle="--",
        label="Black-Scholes",
    )

    plt.xlabel(
        "Number of steps"
    )

    plt.ylabel(
        "Call price"
    )

    plt.title(
        "Binomial and trinomial convergence to Black-Scholes"
    )

    plt.legend()
    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PRICE_FIGURE_PATH,
        dpi=200,
    )

    plt.close()


def save_error_figure(
    results: list[dict[str, float | int]]
) -> None:
    """Save absolute convergence errors."""
    steps = [
        int(result["nb_steps"])
        for result in results
    ]

    binomial_errors = [
        float(result["binomial_absolute_error"])
        for result in results
    ]

    trinomial_errors = [
        float(result["trinomial_absolute_error"])
        for result in results
    ]

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        steps,
        binomial_errors,
        marker="o",
        label="Binomial",
    )

    plt.plot(
        steps,
        trinomial_errors,
        marker="o",
        label="Trinomial",
    )

    plt.xlabel(
        "Number of steps"
    )

    plt.ylabel(
        "Absolute pricing error"
    )

    plt.title(
        "Lattice error versus Black-Scholes"
    )

    plt.legend()
    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        ERROR_FIGURE_PATH,
        dpi=200,
    )

    plt.close()


def save_summary(
    results: list[dict[str, float | int]]
) -> None:
    """Save the main convergence results."""
    first_result = results[0]
    last_result = results[-1]

    content = (
        "Binomial vs trinomial convergence experiment\n"
        "============================================\n\n"
        f"Spot: {SPOT}\n"
        f"Strike: {STRIKE}\n"
        f"Rate: {RATE}\n"
        f"Volatility: {VOLATILITY}\n"
        f"Maturity: {MATURITY}\n\n"
        f"Black-Scholes price: "
        f"{float(first_result['black_scholes_price']):.8f}\n\n"
        f"Binomial absolute error at N="
        f"{int(last_result['nb_steps'])}: "
        f"{float(last_result['binomial_absolute_error']):.8f}\n"
        f"Trinomial absolute error at N="
        f"{int(last_result['nb_steps'])}: "
        f"{float(last_result['trinomial_absolute_error']):.8f}\n"
    )

    SUMMARY_PATH.write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    """Run and save the convergence experiment."""
    results = run_experiment()

    save_csv(
        results
    )

    save_price_figure(
        results
    )

    save_error_figure(
        results
    )

    save_summary(
        results
    )

    print()
    print("Results saved to:")

    print(
        CSV_PATH.relative_to(
            PROJECT_ROOT
        )
    )

    print(
        PRICE_FIGURE_PATH.relative_to(
            PROJECT_ROOT
        )
    )

    print(
        ERROR_FIGURE_PATH.relative_to(
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