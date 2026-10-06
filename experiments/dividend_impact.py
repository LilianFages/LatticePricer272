import csv
from pathlib import Path

import matplotlib.pyplot as plt

from src.dividend import CashDividend
from src.option import CallOption, PutOption
from src.recursive_pricer import RecursivePricer
from src.trinomial_tree import TrinomialTree


SPOT = 100.0
STRIKE = 100.0
RATE = 0.0
VOLATILITY = 0.20
MATURITY = 1.0
DIVIDEND_TIME = 0.5
NB_STEPS = 100

DIVIDEND_AMOUNTS = [
    0.0,
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
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
    / "dividend_impact.csv"
)

FIGURE_PATH = (
    FIGURES_DIR
    / "dividend_impact.png"
)

PARITY_FIGURE_PATH = (
    FIGURES_DIR
    / "dividend_put_call_parity.png"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "dividend_impact_summary.txt"
)


def option_price(
    dividend_amount: float,
    is_call: bool
) -> float:
    """Return a trinomial option price for one dividend amount."""
    dividends: list[CashDividend] = []

    if dividend_amount > 0.0:
        dividends.append(
            CashDividend(
                time=DIVIDEND_TIME,
                amount=dividend_amount,
            )
        )

    tree = TrinomialTree(
        spot=SPOT,
        rate=RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
        nb_steps=NB_STEPS,
        dividends=dividends,
    )

    if is_call:
        option = CallOption(
            strike=STRIKE
        )
    else:
        option = PutOption(
            strike=STRIKE
        )

    return RecursivePricer().price(
        tree,
        option,
    )


def run_experiment() -> list[dict[str, float]]:
    """Measure the impact of discrete dividends on call and put prices."""
    results: list[
        dict[str, float]
    ] = []

    for dividend_amount in DIVIDEND_AMOUNTS:
        call_price = option_price(
            dividend_amount,
            is_call=True,
        )

        put_price = option_price(
            dividend_amount,
            is_call=False,
        )

        call_minus_put = (
            call_price
            - put_price
        )

        theoretical_parity = (
            SPOT
            - dividend_amount
            - STRIKE
        )

        parity_error = (
            call_minus_put
            - theoretical_parity
        )

        results.append(
            {
                "dividend": dividend_amount,
                "call_price": call_price,
                "put_price": put_price,
                "call_minus_put": call_minus_put,
                "theoretical_parity": theoretical_parity,
                "parity_error": parity_error,
            }
        )

        print(
            f"D={dividend_amount:>5.2f} | "
            f"Call={call_price:.8f} | "
            f"Put={put_price:.8f} | "
            f"C-P={call_minus_put:+.8f} | "
            f"Theory={theoretical_parity:+.8f} | "
            f"Parity error={parity_error:+.8e}"
        )

    return results


def save_csv(
    results: list[dict[str, float]]
) -> None:
    """Save dividend impact data."""
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
                "dividend",
                "call_price",
                "put_price",
                "call_minus_put",
                "theoretical_parity",
                "parity_error",
            ],
        )

        writer.writeheader()

        writer.writerows(
            results
        )


def save_price_figure(
    results: list[dict[str, float]]
) -> None:
    """Save option prices as functions of the dividend."""
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dividends = [
        result["dividend"]
        for result in results
    ]

    call_prices = [
        result["call_price"]
        for result in results
    ]

    put_prices = [
        result["put_price"]
        for result in results
    ]

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        dividends,
        call_prices,
        marker="o",
        label="Call",
    )

    plt.plot(
        dividends,
        put_prices,
        marker="o",
        label="Put",
    )

    plt.xlabel(
        "Cash dividend"
    )

    plt.ylabel(
        "Option price"
    )

    plt.title(
        "Impact of a discrete dividend at zero interest rate"
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


def save_parity_figure(
    results: list[dict[str, float]]
) -> None:
    """Compare observed and theoretical put-call parity."""
    dividends = [
        result["dividend"]
        for result in results
    ]

    observed_values = [
        result["call_minus_put"]
        for result in results
    ]

    theoretical_values = [
        result["theoretical_parity"]
        for result in results
    ]

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        dividends,
        observed_values,
        marker="o",
        label="Trinomial C - P",
    )

    plt.plot(
        dividends,
        theoretical_values,
        linestyle="--",
        label="Theoretical parity",
    )

    plt.xlabel(
        "Cash dividend"
    )

    plt.ylabel(
        "Call price - Put price"
    )

    plt.title(
        "Put-call parity with a discrete dividend"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PARITY_FIGURE_PATH,
        dpi=200,
    )

    plt.close()


def save_summary(
    results: list[dict[str, float]]
) -> None:
    """Save the main conclusions of the experiment."""
    first_result = results[0]
    last_result = results[-1]

    max_parity_error = max(
        abs(result["parity_error"])
        for result in results
    )

    content = (
        "Discrete dividend impact experiment\n"
        "===================================\n\n"
        f"Spot: {SPOT}\n"
        f"Strike: {STRIKE}\n"
        f"Rate: {RATE}\n"
        f"Volatility: {VOLATILITY}\n"
        f"Maturity: {MATURITY}\n"
        f"Dividend time: {DIVIDEND_TIME}\n"
        f"Number of steps: {NB_STEPS}\n\n"
        f"Call price without dividend: "
        f"{first_result['call_price']:.8f}\n"
        f"Call price with D="
        f"{last_result['dividend']:.2f}: "
        f"{last_result['call_price']:.8f}\n\n"
        f"Put price without dividend: "
        f"{first_result['put_price']:.8f}\n"
        f"Put price with D="
        f"{last_result['dividend']:.2f}: "
        f"{last_result['put_price']:.8f}\n\n"
        f"Maximum put-call parity error: "
        f"{max_parity_error:.8e}\n"
    )

    SUMMARY_PATH.write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    """Run and save the dividend impact experiment."""
    results = run_experiment()

    save_csv(
        results
    )

    save_price_figure(
        results
    )

    save_parity_figure(
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
        FIGURE_PATH.relative_to(
            PROJECT_ROOT
        )
    )

    print(
        PARITY_FIGURE_PATH.relative_to(
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