import csv
from pathlib import Path

from src.backward_pricer import BackwardPricer
from src.binomial_tree import BinomialTree
from src.option import CallOption, PutOption


SPOT = 100.0
STRIKE = 100.0
VOLATILITY = 0.20
MATURITY = 1.0
NB_STEPS = 200

RATES = [
    -0.10,
    -0.075,
    -0.05,
    -0.025,
    0.00,
    0.025,
    0.05,
    0.075,
    0.10,
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

CSV_PATH = (
    DATA_DIR
    / "american_vs_european.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "american_vs_european_summary.txt"
)


def price_option(
    rate: float,
    option_type: str,
    is_american: bool,
) -> float:
    """Price one option configuration."""
    tree = BinomialTree(
        spot=SPOT,
        rate=rate,
        volatility=VOLATILITY,
        maturity=MATURITY,
        nb_steps=NB_STEPS,
    )

    if option_type == "call":
        option = CallOption(
            strike=STRIKE,
            is_american=is_american,
        )

    elif option_type == "put":
        option = PutOption(
            strike=STRIKE,
            is_american=is_american,
        )

    else:
        raise ValueError(
            "Option type must be call or put."
        )

    return BackwardPricer().price(
        tree,
        option,
    )


def run_experiment() -> list[dict[str, object]]:
    """Run American vs European pricing across rates."""
    results: list[dict[str, object]] = []

    for rate in RATES:
        european_call = price_option(
            rate,
            "call",
            False,
        )

        american_call = price_option(
            rate,
            "call",
            True,
        )

        european_put = price_option(
            rate,
            "put",
            False,
        )

        american_put = price_option(
            rate,
            "put",
            True,
        )

        call_difference = (
            american_call
            - european_call
        )

        put_difference = (
            american_put
            - european_put
        )

        results.append(
            {
                "rate": rate,
                "european_call": european_call,
                "american_call": american_call,
                "call_difference": call_difference,
                "european_put": european_put,
                "american_put": american_put,
                "put_difference": put_difference,
            }
        )

        print(
            f"r={rate:+.3f} | "
            f"Call Am-Eu={call_difference:.6f} | "
            f"Put Am-Eu={put_difference:.6f}"
        )

    return results


def save_csv(
    results: list[dict[str, object]],
) -> None:
    """Save raw results to CSV."""
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
                "rate",
                "european_call",
                "american_call",
                "call_difference",
                "european_put",
                "american_put",
                "put_difference",
            ],
        )

        writer.writeheader()
        writer.writerows(results)


def save_summary(
    results: list[dict[str, object]],
) -> None:
    """Save a readable summary of the experiment."""
    lines = [
        "American vs European pricing experiment",
        "=======================================",
        "",
        f"Spot: {SPOT}",
        f"Strike: {STRIKE}",
        f"Volatility: {VOLATILITY}",
        f"Maturity: {MATURITY}",
        f"Number of steps: {NB_STEPS}",
        "",
        "rate | call Am-Eu | put Am-Eu",
        "--------------------------------",
    ]

    for result in results:
        rate = float(result["rate"])

        call_difference = float(
            result["call_difference"]
        )

        put_difference = float(
            result["put_difference"]
        )

        lines.append(
            f"{rate:+.3f} | "
            f"{call_difference:.6f} | "
            f"{put_difference:.6f}"
        )

    SUMMARY_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Run and save the experiment."""
    results = run_experiment()

    save_csv(
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
        SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        )
    )


if __name__ == "__main__":
    main()