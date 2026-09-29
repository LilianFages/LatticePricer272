import csv
import gc
import platform
import sys
import time
from pathlib import Path

from src.backward_pricer import BackwardPricer
from src.binomial_tree import BinomialTree
from src.option import CallOption
from src.recursive_pricer import RecursivePricer


SPOT = 100.0
RATE = 0.02
VOLATILITY = 0.20
MATURITY = 1.0
STRIKE = 100.0

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
    / "recursion_limit.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "recursion_limit_summary.txt"
)


def create_results_directories() -> None:
    """Create result directories when needed."""
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        RESULTS_DIR
        / "figures"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )


def run_recursive(
    nb_steps: int,
) -> tuple[bool, float, float | None]:
    """Run recursive pricing for a given number of steps."""
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

    start_time = time.perf_counter()

    try:
        price = RecursivePricer().price(
            tree,
            option,
        )

        success = True

    except RecursionError:
        price = None
        success = False

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    del tree
    gc.collect()

    return (
        success,
        elapsed_time,
        price,
    )


def add_recursive_result(
    results: list[dict[str, object]],
    nb_steps: int,
) -> bool:
    """Run and store one recursive experiment."""
    success, elapsed_time, price = (
        run_recursive(nb_steps)
    )

    status = (
        "OK"
        if success
        else "RecursionError"
    )

    results.append(
        {
            "method": "recursive",
            "nb_steps": nb_steps,
            "status": status,
            "price": price,
            "elapsed_seconds": elapsed_time,
        }
    )

    print(
        f"N={nb_steps}: "
        f"{status} "
        f"({elapsed_time:.3f} s)"
    )

    return success


def find_recursion_limit(
    results: list[dict[str, object]],
) -> tuple[int, int]:
    """Find the last successful and first failing step count."""
    python_limit = sys.getrecursionlimit()

    low = max(
        python_limit // 2,
        1,
    )

    high = (
        python_limit
        + 100
    )

    low_success = add_recursive_result(
        results,
        low,
    )

    if not low_success:
        raise RuntimeError(
            "Initial lower bound already fails."
        )

    high_success = add_recursive_result(
        results,
        high,
    )

    if high_success:
        raise RuntimeError(
            "Initial upper bound still succeeds."
        )

    while high - low > 1:
        middle = (
            low + high
        ) // 2

        success = add_recursive_result(
            results,
            middle,
        )

        if success:
            low = middle
        else:
            high = middle

    return low, high


def run_backward(
    nb_steps: int,
) -> tuple[float, float]:
    """Run backward pricing for a given number of steps."""
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

    start_time = time.perf_counter()

    price = BackwardPricer().price(
        tree,
        option,
    )

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    return price, elapsed_time


def add_backward_result(
    results: list[dict[str, object]],
    nb_steps: int,
) -> None:
    """Run and store the backward comparison."""
    price, elapsed_time = (
        run_backward(nb_steps)
    )

    results.append(
        {
            "method": "backward",
            "nb_steps": nb_steps,
            "status": "OK",
            "price": price,
            "elapsed_seconds": elapsed_time,
        }
    )

    print(
        f"Backward N={nb_steps}: "
        f"OK, "
        f"price={price:.6f}, "
        f"time={elapsed_time:.3f} s"
    )


def save_csv(
    results: list[dict[str, object]],
) -> None:
    """Save raw experiment results to CSV."""
    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "method",
                "nb_steps",
                "status",
                "price",
                "elapsed_seconds",
            ],
        )

        writer.writeheader()
        writer.writerows(results)


def save_summary(
    last_success: int,
    first_failure: int,
) -> None:
    """Save the main recursion-limit result."""
    content = (
        "Recursive pricing limit experiment\n"
        "==================================\n\n"
        f"Python version: {platform.python_version()}\n"
        f"Python recursion limit: {sys.getrecursionlimit()}\n\n"
        f"Spot: {SPOT}\n"
        f"Strike: {STRIKE}\n"
        f"Rate: {RATE}\n"
        f"Volatility: {VOLATILITY}\n"
        f"Maturity: {MATURITY}\n\n"
        f"Largest successful recursive N: {last_success}\n"
        f"First recursive RecursionError N: {first_failure}\n"
        f"Backward pricing at N={first_failure}: OK\n"
    )

    SUMMARY_PATH.write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    """Run the recursion-limit experiment."""
    create_results_directories()

    results: list[dict[str, object]] = []

    print(
        "Python version:",
        platform.python_version(),
    )

    print(
        "Python recursion limit:",
        sys.getrecursionlimit(),
    )

    print()

    last_success, first_failure = (
        find_recursion_limit(
            results
        )
    )

    print()
    print(
        "Largest recursive N:",
        last_success,
    )

    print(
        "First RecursionError N:",
        first_failure,
    )

    print()

    add_backward_result(
        results,
        first_failure,
    )

    save_csv(
        results
    )

    save_summary(
        last_success,
        first_failure,
    )

    print()
    print(
        "Results saved to:"
    )

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