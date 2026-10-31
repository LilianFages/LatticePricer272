"""Time profiles of put values and optimal-exercise probabilities.

Reproduces the two experiments from the course slides:
  (1) Average option value by dates, expressed in pricing-date numeraire.
  (2) Exercise probability by dates (exercise-region node mass).

A supplementary first-exercise (stopping-time) distribution avoids counting
paths more than once.  Uses existing src/ objects without modifying them.

Run from repository root:
    python -m experiments.dividends.option_value_exercise_profiles
    python -m experiments.dividends.option_value_exercise_profiles --steps 150
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.lattices.allocation_pruned_trinomial_tree import AllocationPrunedTrinomialTree
from src.contracts.dividend import CashDividend
from src.lattices.node import Node
from src.contracts.option import PutOption
from src.pricing.recursive_pricer import RecursivePricer


# Aggregate recombining transition mass before the exercise decision.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "results" / "data"
FIGURES_DIR = ROOT / "results" / "figures"
SUMMARY_PATH = ROOT / "results" / "option_by_dates_summary.txt"
DATA_PATH = DATA_DIR / "option_by_dates_profiles.csv"

SPOT = 100.0
STRIKE = 100.0
RATE = 0.02
VOLATILITY = 0.20
MATURITY = 1.0
DIVIDEND_TIME = 0.5
DIVIDEND_AMOUNT = 3.0
PRUNING_THRESHOLD = 1e-18
EXERCISE_TOLERANCE = 1e-10
MASS_TOLERANCE = 2e-10


# Distinguish the exercise region from the first-exercise distribution.
@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    spot: float
    is_american: bool
    dividends: tuple[tuple[float, float], ...] = ()


@dataclass
class Result:
    scenario: Scenario
    initial_price: float
    rows: list[dict[str, float | int | str]]
    forward_at_maturity: float
    total_first_exercise_probability: float
    expiry_without_exercise_probability: float


def make_scenarios() -> list[Scenario]:
    """Three schemes with a common *initial terminal forward*.

    For a known discrete cash dividend D at time t_d,
    F_0(T) = S_0 exp(rT) - D exp(r(T-t_d)).
    Thus an ex-dividend stock's matched no-dividend spot equals
    S_0 - D exp(-rt_d), not S_0 itself.
    """
    adjusted_spot = SPOT - DIVIDEND_AMOUNT * math.exp(-RATE * DIVIDEND_TIME)
    if adjusted_spot <= 0.0:
        raise ValueError("Adjusted spot must remain positive.")
    return [
        Scenario(
            key="american_cash_dividend",
            label="American put — cash dividend",
            spot=SPOT,
            is_american=True,
            dividends=((DIVIDEND_TIME, DIVIDEND_AMOUNT),),
        ),
        Scenario(
            key="american_no_dividend",
            label="American put — no dividend",
            spot=adjusted_spot,
            is_american=True,
        ),
        Scenario(
            key="european_no_dividend",
            label="European put — no dividend",
            spot=adjusted_spot,
            is_american=False,
        ),
    ]


def transitions(node: Node) -> list[tuple[Node, float]]:
    """Return all forward branches; duplicate successors are intentional.

    Probability-based pruning can connect up, mid and down to one node.
    Summing these weights is essential; deduplicating would lose mass.
    """
    pairs = (
        (node.next_up, node.up_probability),
        (node.next_mid, node.mid_probability),
        (node.next_down, node.down_probability),
    )
    result: list[tuple[Node, float]] = []
    for child, weight in pairs:
        if child is None:
            if weight not in (None, 0.0):
                raise AssertionError("Nonzero probability without successor.")
            continue
        if weight is None or not math.isfinite(weight) or weight < -1e-12:
            raise AssertionError("Invalid probability at a linked node.")
        if weight > 0.0:
            result.append((child, float(weight)))
    total = sum(weight for _, weight in result)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=5e-10):
        raise AssertionError(f"Outgoing probability mass = {total:.16g}.")
    return result


def propagate(states: dict[Node, float]) -> dict[Node, float]:
    """Advance an unstopped risk-neutral distribution by one tree step."""
    next_states: dict[Node, float] = defaultdict(float)
    for node, mass in states.items():
        for child, weight in transitions(node):
            next_states[child] += mass * weight
    return dict(next_states)


def exercise_decision(node: Node, option: PutOption, dt: float, rate: float,
                      is_maturity: bool) -> bool:
    """True if the node lies in the exercise set, using strict preference.

    At maturity exercise has positive payoff; at earlier dates, only the
    American option can exercise, and only if payoff strictly exceeds hold.
    """
    intrinsic = option.payoff(node.price)
    if intrinsic <= 0.0:
        return False
    if is_maturity:
        return True
    if not option.is_american:
        return False
    hold = math.exp(-rate * dt) * sum(
        weight * _value(child) for child, weight in transitions(node)
    )
    scale = max(1.0, abs(intrinsic), abs(hold))
    return intrinsic > hold + EXERCISE_TOLERANCE * scale


def _value(node: Node) -> float:
    if node.option_value is None or not math.isfinite(node.option_value):
        raise AssertionError("Missing option value: run RecursivePricer first.")
    return float(node.option_value)


def _exercise_partition(
    all_states: dict[Node, float],
    alive_states: dict[Node, float],
    option: PutOption,
    dt: float,
    rate: float,
    is_maturity: bool
) -> tuple[float, float, dict[Node, float]]:
    """Measure exercise-region mass and surviving first-exercise paths."""
    exercise_set = {
        node for node in all_states
        if exercise_decision(node, option, dt, rate, is_maturity)
    }
    region_mass = sum(all_states[node] for node in exercise_set)
    first_mass = sum(alive_states.get(node, 0.0) for node in exercise_set)
    remaining = {
        node: prob for node, prob in alive_states.items()
        if node not in exercise_set
    }
    return region_mass, first_mass, remaining


def _profile_row(
    scenario: Scenario,
    step: int,
    dt: float,
    rate: float,
    mean_value: float,
    mean_stock: float,
    all_mass: float,
    region_mass: float,
    first_mass: float,
    exercised_total: float,
    remaining_mass: float
) -> dict[str, float | int | str]:
    """Record exactly the original professor and stopping-time measures."""
    t = step * dt
    discount = math.exp(-rate * t)
    return {
        "scenario": scenario.key,
        "step": step,
        "time": t,
        "mean_option_value": mean_value,
        "discounted_mean_option_value": discount * mean_value,
        "mean_stock_price": mean_stock,
        "unconditional_probability_mass": all_mass,
        "exercise_region_probability": region_mass,
        "first_exercise_probability": first_mass,
        "cumulative_first_exercise_probability": exercised_total,
        "survival_probability_after_step": remaining_mass,
    }


# Unconditional and surviving paths must remain separate at every date.
def _profile_step(
    all_states: dict[Node, float],
    alive_states: dict[Node, float],
    option: PutOption,
    step: int,
    nb_steps: int,
    dt: float,
    rate: float,
    scenario: Scenario,
    exercised_total: float
) -> tuple[dict[str, float | int | str], dict[Node, float], float, float]:
    """Compute one unconditional and stopping-time probability observation."""
    all_mass = sum(all_states.values())
    if not math.isclose(all_mass, 1.0, abs_tol=MASS_TOLERANCE, rel_tol=0):
        raise AssertionError(f"Lost unconditional mass at step {step}: {all_mass}.")
    mean_value = sum(prob * _value(node) for node, prob in all_states.items())
    mean_stock = sum(prob * node.price for node, prob in all_states.items())
    # Exercise-region mass uses all paths; stopping mass uses surviving paths.
    region_mass, first_mass, remaining = _exercise_partition(
        all_states, alive_states, option, dt, rate, step == nb_steps
    )
    exercised_total += first_mass
    row = _profile_row(
        scenario, step, dt, rate, mean_value, mean_stock, all_mass,
        region_mass, first_mass, exercised_total, sum(remaining.values())
    )
    return row, remaining, exercised_total, mean_stock


def _validated_profile_result(
    root: Node,
    scenario: Scenario,
    rows: list[dict[str, float | int | str]],
    final_forward: float,
    exercised_total: float,
    expiry_without_exercise: float
) -> Result:
    """Verify normalized stopping mass and the initial discounted price."""
    if not math.isclose(exercised_total + expiry_without_exercise,
                        1.0, abs_tol=MASS_TOLERANCE, rel_tol=0):
        raise AssertionError("First-exercise plus worthless expiry must sum to one.")
    initial_price = _value(root)
    if not math.isclose(float(rows[0]["discounted_mean_option_value"]),
                        initial_price, abs_tol=1e-12, rel_tol=1e-12):
        raise AssertionError("Wrong initial option value.")
    return Result(scenario, initial_price, rows, final_forward,
                  exercised_total, expiry_without_exercise)


def profile_from_priced_tree(
    root: Node,
    option: PutOption,
    nb_steps: int,
    maturity: float,
    rate: float,
    scenario: Scenario
) -> Result:
    """Compute datewise discounted option value and exercise probabilities.

    Unconditional paths measure the professor's exercise-region mass.
    Surviving paths measure a separate first-exercise distribution.
    """
    dt = maturity / nb_steps
    all_states: dict[Node, float] = {root: 1.0}
    alive_states: dict[Node, float] = {root: 1.0}
    rows: list[dict[str, float | int | str]] = []
    exercised_total = 0.0
    expiry_without_exercise = 0.0
    final_forward = float("nan")
    # Propagate all paths and non-exercised paths under the same transitions.
    for step in range(nb_steps + 1):
        row, remaining, exercised_total, mean_stock = _profile_step(
            all_states, alive_states, option, step, nb_steps,
            dt, rate, scenario, exercised_total
        )
        rows.append(row)
        if step == nb_steps:
            expiry_without_exercise = sum(remaining.values())
            final_forward = mean_stock
        else:
            all_states = propagate(all_states)
            alive_states = propagate(remaining)

    return _validated_profile_result(
        root, scenario, rows, final_forward,
        exercised_total, expiry_without_exercise
    )


# Price each scenario first, then analyze its resulting exercise policy.
def run_scenario(scenario: Scenario, nb_steps: int) -> Result:
    dividends = [CashDividend(time=t, amount=d) for t, d in scenario.dividends]
    option = PutOption(strike=STRIKE, is_american=scenario.is_american)
    tree = AllocationPrunedTrinomialTree(
        spot=scenario.spot,
        rate=RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
        nb_steps=nb_steps,
        # Evaluate one contract without blending its exercise convention with others.
        dividends=dividends,
        pruning_threshold=PRUNING_THRESHOLD,
    )
    RecursivePricer().price(tree, option)
    return profile_from_priced_tree(
        tree.root, option, nb_steps, MATURITY, RATE, scenario
    )


def run_experiment(nb_steps: int = 100) -> list[Result]:
    if nb_steps < 2 or nb_steps % 2 != 0:
        raise ValueError("Steps must be even and at least 2 (dividend at T/2).")
    results = [run_scenario(scenario, nb_steps) for scenario in make_scenarios()]
    forwards = [res.forward_at_maturity for res in results]
    if max(forwards) - min(forwards) > 1e-7:
        raise AssertionError(f"Forward mismatch: {forwards}")
    euro = next(res for res in results if res.scenario.key == "european_no_dividend")
    # Run each predefined scenario without changing the comparison baseline.
    max_martingale_deviation = max(
        abs(float(row["discounted_mean_option_value"]) - euro.initial_price)
        for row in euro.rows
    )
    if max_martingale_deviation > 1e-7:
        raise AssertionError(f"European martingale residual {max_martingale_deviation}")
    return results


def plot_profiles(results: list[Result], path: Path, key: str,
                  title: str, ylabel: str, *, percent: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for res in results:
        xs = [float(row["time"]) for row in res.rows]
        ys = [float(row[key]) * (100 if percent else 1) for row in res.rows]
        ax.plot(xs, ys, label=res.scenario.label, linewidth=1.8)
    ax.axvline(DIVIDEND_TIME, linestyle=":", color="0.45", linewidth=1.1,
               # Express datewise option values in a consistent pricing-date numeraire.
               label="Cash-dividend date")
    ax.set(title=title, xlabel="Time (years)", ylabel=ylabel, xlim=(0, MATURITY))
    if percent:
        ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def save_csv(results: list[Result]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = [row for res in results for row in res.rows]
    with DATA_PATH.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        # A European option must carry zero early-exercise probability.
        writer.writerows(rows)


def save_summary(results: list[Result], nb_steps: int) -> None:
    s = make_scenarios()
    matched_spot = s[1].spot
    lines = [
        "OPTION VALUE AND EXERCISE PROBABILITY BY DATES",
        "==============================================",
        "Experiment reproducing professor's three-slide definitions.",
        "",
        f"S0 cash-dividend scheme={SPOT}; no-dividend adjusted spot={matched_spot:.12f}",
        f"K={STRIKE}, r={RATE}, sigma={VOLATILITY}, T={MATURITY}, N={nb_steps}",
        f"Cash dividend D={DIVIDEND_AMOUNT} at t={DIVIDEND_TIME}",
        f"Pruning threshold={PRUNING_THRESHOLD}",
        "A discrete cash dividend of 3 EUR is NOT a continuous 3% dividend yield.",
        "Matched INITIAL TERMINAL FORWARD, not matched initial spot or dynamics.",
        # Preserve the option stopping rule for each simulated scenario.
        "",
        "Professor metric 1: exp(-r*t) * sum(reach_prob(node) * option_value(node)).",
        "Professor metric 2: sum(reach_prob(node) over exercise-region nodes).",
        "In the second metric, earlier exercised paths are NOT removed.",
        "Supplement: first-exercise probability removes already-exercised paths.",
        "Exercise ties treated as continuation; terminal exercise iff payoff > 0.",
        "Exercise-region values at different dates must NOT be added to 100%.",
        "",
    ]
    for res in results:
        x = res.rows
        err = max(abs(float(z["discounted_mean_option_value"]) - res.initial_price)
                  for z in x)
        early = res.total_first_exercise_probability - float(x[-1]["first_exercise_probability"])
        lines.extend([
            f"SCENARIO: {res.scenario.label}",
            f"  Initial put premium            = {res.initial_price:.12f} EUR",
            # Aggregate recombining transition mass before the exercise decision.
            f"  Mean terminal forward          = {res.forward_at_maturity:.12f} EUR",
            f"  PV average at maturity         = {float(x[-1]['discounted_mean_option_value']):.12f} EUR",
            f"  Max |PV mean - initial price| = {err:.12e} EUR",
            f"  Exercise first time (early)   = {early:.10%}",
            f"  Exercise first time (all)     = {res.total_first_exercise_probability:.10%}",
            f"  Expiry without exercise       = {res.expiry_without_exercise_probability:.10%}",
            "",
        ])
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    results = run_experiment(args.steps)
    save_csv(results)
    # Distinguish the exercise region from the first-exercise distribution.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plots = [
        ("discounted_mean_option_value", "option_by_dates_average_pv.png",
         "Average option value by dates — discounted to t=0", "Average discounted put value (EUR)", False),
        ("exercise_region_probability", "option_by_dates_exercise_region.png",
         "Exercise probability by dates — node-region mass", "Unconditional probability in exercise region (%)", True),
        ("first_exercise_probability", "option_by_dates_first_exercise.png",
         "Probability of first exercise by dates — stopping distribution",
         "First exercise probability at date (%)", True),
    ]
    for key, name, title, ylabel, percent in plots:
        plot_profiles(results, FIGURES_DIR / name, key, title, ylabel, percent=percent)
    save_summary(results, args.steps)
    for res in results:
        print(f"{res.scenario.label:34s} | "
              f"V0={res.initial_price:.8f} EUR | "
              f"forward={res.forward_at_maturity:.8f} | "
              f"first exercise={res.total_first_exercise_probability:.3%}")
    print(f"CSV: {DATA_PATH.relative_to(ROOT)}")
    print(f"Summary: {SUMMARY_PATH.relative_to(ROOT)}")
    print("Figures: results/figures/option_by_dates_*.png")


if __name__ == "__main__":
    main()
