import csv
from pathlib import Path

from src.backward_pricer import BackwardPricer
from src.binomial_error import BinomialErrorEstimator
from src.binomial_tree import BinomialTree
from src.black_scholes import BlackScholesPricer
from src.option import CallOption


SPOT = 100.0
RATE = 0.02
VOLATILITY = 0.20
MATURITY = 1.0

TARGET_PRECISIONS = [
    0.005,
    0.0025,
    0.001,
    0.0005,
    0.0001,
]

STRIKES = [
    89.0 + 0.25 * index
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

CSV_PATH = (
    DATA_DIR
    / "precision_target.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "precision_target_summary.txt"
)


def black_scholes_price(
    strike: float,
) -> float:
    """Return the Black-Scholes call price."""
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
    strike: float,
    nb_steps: int,
) -> float:
    """Return the binomial call price."""
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


def empirical_gap(
    nb_steps: int,
) -> float:
    """Return the peak-to-peak pricing error."""
    minimum_error: float | None = None
    maximum_error: float | None = None

    for strike in STRIKES:
        error = (
            tree_price(
                strike,
                nb_steps,
            )
            - black_scholes_price(
                strike
            )
        )

        if (
            minimum_error is None
            or error < minimum_error
        ):
            minimum_error = error

        if (
            maximum_error is None
            or error > maximum_error
        ):
            maximum_error = error

    if (
        minimum_error is None
        or maximum_error is None
    ):
        raise RuntimeError(
            "No strike error was calculated."
        )

    return (
        maximum_error
        - minimum_error
    )


def run_experiment() -> list[dict[str, object]]:
    """Test theoretical step recommendations."""
    estimator = BinomialErrorEstimator()

    results: list[
        dict[str, object]
    ] = []

    for target in TARGET_PRECISIONS:
        nb_steps = (
            estimator.steps_for_relative_precision(
                target_precision=target,
                rate=RATE,
                volatility=VOLATILITY,
                maturity=MATURITY,
            )
        )

        theoretical_relative_gap = (
            estimator.relative_gap(
                nb_steps=nb_steps,
                rate=RATE,
                volatility=VOLATILITY,
                maturity=MATURITY,
            )
        )

        measured_gap = empirical_gap(
            nb_steps
        )

        empirical_relative_gap = (
            measured_gap
            / SPOT
        )

        target_met = (
            empirical_relative_gap
            <= target
        )

        ratio = (
            empirical_relative_gap
            / target
        )

        results.append(
            {
                "target_precision": target,
                "recommended_steps": nb_steps,
                "theoretical_relative_gap": (
                    theoretical_relative_gap
                ),
                "empirical_relative_gap": (
                    empirical_relative_gap
                ),
                "empirical_target_ratio": ratio,
                "target_met": target_met,
            }
        )

        status = (
            "OK"
            if target_met
            else "FAILED"
        )

        print(
            f"Target={100.0 * target:.3f}% | "
            f"N={nb_steps:>3} | "
            f"Empirical="
            f"{100.0 * empirical_relative_gap:.4f}% | "
            f"Ratio={ratio:.4f} | "
            f"{status}"
        )

    return results


def save_csv(
    results: list[dict[str, object]],
) -> None:
    """Save raw precision results."""
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
                "target_precision",
                "recommended_steps",
                "theoretical_relative_gap",
                "empirical_relative_gap",
                "empirical_target_ratio",
                "target_met",
            ],
        )

        writer.writeheader()
        writer.writerows(results)


def save_summary(
    results: list[dict[str, object]],
) -> None:
    """Save a readable experiment summary."""
    lines = [
        "User-defined precision experiment",
        "=================================",
        "",
        f"Spot: {SPOT}",
        f"Rate: {RATE}",
        f"Volatility: {VOLATILITY}",
        f"Maturity: {MATURITY}",
        "",
        "Target | Steps | Empirical gap | Status",
        "---------------------------------------",
    ]

    for result in results:
        target = float(
            result["target_precision"]
        )

        nb_steps = int(
            result["recommended_steps"]
        )

        empirical = float(
            result["empirical_relative_gap"]
        )

        status = (
            "OK"
            if bool(result["target_met"])
            else "FAILED"
        )

        lines.append(
            f"{100.0 * target:.3f}% | "
            f"{nb_steps} | "
            f"{100.0 * empirical:.4f}% | "
            f"{status}"
        )

    SUMMARY_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Run and save the precision experiment."""
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