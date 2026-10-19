"""Validate allocation pruning independently and against branch pruning.

Run at repository root:
    python -m experiments.allocation_pruning_robustness

The experiment compares two approximations of the SAME trinomial model.
It does not estimate pricing error against an exact financial benchmark.
When the branch-pruned reference fails on a zero-mass extreme cash-dividend
node, the allocation-pruned run is still checked and reported independently.
Such a case is NOT counted as a successful price comparison.
Single-run runtimes are diagnostic only, not a rigorous speed benchmark.
"""

import csv
import math
from itertools import product
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.allocation_pruned_trinomial_tree import AllocationPrunedTrinomialTree
from src.dividend import CashDividend
from src.node import Node, TrunkNode
from src.option import CallOption, Option, PutOption
from src.recursive_pricer import RecursivePricer
from src.trinomial_tree import TrinomialTree


# Compare allocation and branch pruning on identical scenario inputs.
SPOT = 100.0
RATE = 0.02
VOLATILITY = 0.20
MATURITY = 1.0
PRUNING_THRESHOLD = 1e-18
STEP_COUNTS = (40, 100, 200)
STRIKES = (80.0, 100.0, 120.0)
OPTION_KINDS = ("call", "put")
EXERCISE_STYLES = (False, True)
DIVIDEND_SCENARIOS = {
    "no_dividend": (),
    "single_dividend": ((0.50, 3.0),),
    "two_dividends": ((0.25, 2.0), (0.75, 3.0)),
}

PRICE_ABS_TOL = 1e-9
PRICE_REL_TOL = 1e-10
# Check probability-mass conservation independently of price agreement.
MASS_ABS_TOL = 1e-10
LOCAL_PROB_TOL = 1e-12

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "data" / "allocation_pruning_robustness.csv"
SUMMARY_PATH = ROOT / "results" / "allocation_pruning_robustness_summary.txt"
COVERAGE_FIGURE = ROOT / "results" / "figures" / "allocation_pruning_robustness_coverage.png"
SAVINGS_FIGURE = ROOT / "results" / "figures" / "allocation_pruning_robustness_savings.png"


@dataclass(frozen=True)
class ValidationCase:
    scenario: str
    nb_steps: int
    strike: float
    kind: str
    is_american: bool


def cases() -> list[ValidationCase]:
    """Build a deterministic matrix of 108 financial configurations."""
    return [
        ValidationCase(scenario, steps, strike, kind, american)
        for scenario in DIVIDEND_SCENARIOS
        for steps in STEP_COUNTS
        for strike in STRIKES
        for kind in OPTION_KINDS
        for american in EXERCISE_STYLES
    ]


def make_dividends(scenario: str) -> list[CashDividend]:
    """Create fresh dividend objects for every tree construction."""
    return [
        CashDividend(time=time_, amount=amount)
        for time_, amount in DIVIDEND_SCENARIOS[scenario]
    ]


def make_option(case: ValidationCase) -> Option:
    """Create a vanilla option using the project's existing classes."""
    option_class = CallOption if case.kind == "call" else PutOption
    return option_class(strike=case.strike, is_american=case.is_american)


class CountingTrinomialTree(TrinomialTree):
    """Instrument full-column allocations without changing pricing logic."""

    def __init__(
        self: Self,
        spot: float,
        rate: float,
        volatility: float,
        maturity: float,
        nb_steps: int,
        dividends: list[CashDividend] | None = None,
        pruning_threshold: float = PRUNING_THRESHOLD,
    ) -> None:
        super().__init__(
            spot, rate, volatility, maturity, nb_steps,
            dividends, pruning_threshold,
        )
        self.allocated_node_count: int = 1

    # Treat the documented extreme-tail reference failure as a missing comparison.
    def _build_column(
        self: Self,
        trunk_price: float,
        top_exponent: int,
        bottom_exponent: int,
        previous_trunk: TrunkNode,
    ) -> tuple[Node, TrunkNode]:
        built = super()._build_column(
            trunk_price, top_exponent, bottom_exponent, previous_trunk,
        )
        self.allocated_node_count += top_exponent - bottom_exponent + 1
        return built


def make_tree(case: ValidationCase, optimized: bool) -> TrinomialTree:
    """Instantiate the unmodified engine or its allocation-pruned variant."""
    tree_type = AllocationPrunedTrinomialTree if optimized else CountingTrinomialTree
    return tree_type(
        spot=SPOT,
        rate=RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
        nb_steps=case.nb_steps,
        dividends=make_dividends(case.scenario),
        pruning_threshold=PRUNING_THRESHOLD,
    )


def _validated_successor_frontier(
    frontier: dict[int, Node], date_index: int,
) -> dict[int, Node]:
    """Check each transition row and deduplicate recombining successors."""
    next_frontier: dict[int, Node] = {}
    for node in frontier.values():
        transitions = (
            (node.next_up, node.up_probability),
            (node.next_mid, node.mid_probability),
            (node.next_down, node.down_probability),
        )
        row_sum = 0.0
        for successor, probability in transitions:
            if successor is None or probability is None:
                raise RuntimeError(f"Incomplete transition at date {date_index}.")
            if not math.isfinite(probability) or not (
                -LOCAL_PROB_TOL <= probability <= 1.0 + LOCAL_PROB_TOL
            ):
                raise RuntimeError(f"Invalid local probability at date {date_index}.")
            row_sum += probability
            next_frontier[id(successor)] = successor

        if abs(row_sum - 1.0) > LOCAL_PROB_TOL:
            raise RuntimeError(f"Transition probabilities do not sum to 1 at date {date_index}.")

    return next_frontier


def maximum_probability_mass_error(tree: TrinomialTree) -> float:
    """Inspect probability normalization at every reachable date.

    Deduplicate common successor nodes, including a monomial branch whose
    next_up, next_mid and next_down may all be the very same Node object.
    """
    frontier: dict[int, Node] = {id(tree.root): tree.root}
    largest_error = 0.0

    for date_index in range(tree.nb_steps + 1):
        total_mass = math.fsum(node.reach_probability for node in frontier.values())
        if not math.isfinite(total_mass):
            raise RuntimeError(f"Nonfinite probability mass at date {date_index}.")
        largest_error = max(largest_error, abs(total_mass - 1.0))

        if date_index == tree.nb_steps:
            break

        next_frontier = _validated_successor_frontier(frontier, date_index)
        frontier = next_frontier

    return largest_error


def price_one(case: ValidationCase, optimized: bool) -> dict[str, float | int]:
    """Perform one independent build-and-price with structural checks."""
    tree = make_tree(case, optimized)
    start = time.perf_counter()
    price = RecursivePricer().price(tree, make_option(case))
    elapsed = time.perf_counter() - start

    if not math.isfinite(price) or price < 0.0:
        raise RuntimeError("Option price is not finite or is negative.")

    mass_error = maximum_probability_mass_error(tree)
    if mass_error > MASS_ABS_TOL:
        raise RuntimeError(f"Probability mass drift {mass_error:.3e} exceeds tolerance.")

    return {
        "price": price,
        "runtime_seconds": elapsed,
        "allocated_nodes": tree.allocated_node_count,
        "mass_error": mass_error,
        "pruned_nodes": tree.pruned_node_count,
    }



def _empty_validation_result(case: ValidationCase) -> dict[str, object]:
    """Initialize the full, stable validation-report schema."""
    result: dict[str, object] = {
        "scenario": case.scenario,
        "nb_steps": case.nb_steps,
        "strike": case.strike,
        "kind": case.kind,
        "exercise": "american" if case.is_american else "european",
        "status": "failed",
        "branch_status": "failed",
        "allocation_status": "failed",
        "branch_price": "",
        "allocation_price": "",
        "absolute_gap": "",
        "relative_gap": "",
        "branch_nodes": "",
        "allocation_nodes": "",
        "node_savings_pct": "",
        "branch_runtime_seconds": "",
        # Do not interpret one-shot timing measurements as performance confidence bounds.
        "allocation_runtime_seconds": "",
        "branch_mass_error": "",
        "allocation_mass_error": "",
        "branch_error": "",
        "allocation_error": "",
        "error": "",
    }

    return result


def _collect_pricer_outcomes(
    case: ValidationCase, result: dict[str, object],
) -> dict[str, dict[str, float | int]]:
    """Price each engine independently and record its own failure."""
    outcomes: dict[str, dict[str, float | int]] = {}
    for name, optimized in (("branch", False), ("allocation", True)):
        try:
            outcome = price_one(case, optimized=optimized)
        except Exception as exc:
            result[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
        else:
            outcomes[name] = outcome
            result[f"{name}_status"] = "success"
            result[f"{name}_price"] = outcome["price"]
            result[f"{name}_nodes"] = outcome["allocated_nodes"]
            result[f"{name}_runtime_seconds"] = outcome["runtime_seconds"]
            result[f"{name}_mass_error"] = outcome["mass_error"]

    return outcomes


def _classify_missing_branch_reference(
    case: ValidationCase, result: dict[str, object],
) -> None:
    """Distinguish the diagnosed tail limitation from unexpected failures."""
    # Only the diagnosed extreme-tail limitation is an expected
    # reference failure. Do not silently accept unrelated failures.
    known_tail_limitation = (
        case.scenario == "two_dividends"
        and case.nb_steps == 200
        and result["branch_error"]
        == "RuntimeError: Expected stock price must remain positive."
    )
    if known_tail_limitation:
        result["status"] = "allocation_only"
        result["error"] = "Known branch tail limitation; no pricing comparison"
    else:
        result["error"] = (
            "Unexpected branch reference failure: "
            + str(result["branch_error"])
        )



def run_case(case: ValidationCase) -> dict[str, object]:
    """Evaluate the two pricers independently, recording all outcomes.

    status='passed': both prices available and matching, no extra allocations.
    status='allocation_only': allocation succeeds, but branch is unavailable.
    status='failed': allocation fails or an available comparison disagrees.
    """
    result = _empty_validation_result(case)

    outcomes = _collect_pricer_outcomes(case, result)

    allocation = outcomes.get("allocation")
    branch = outcomes.get("branch")

    if allocation is None:
        result["error"] = (
            "Allocation pricing failed: " + str(result["allocation_error"])
        )
        return result

    if branch is None:
        _classify_missing_branch_reference(case, result)
        return result

    # Keep failed branch-pruning references separate from verified allocated results.
    branch_price = float(branch["price"])
    allocation_price = float(allocation["price"])
    absolute_gap = abs(allocation_price - branch_price)
    branch_nodes = int(branch["allocated_nodes"])
    allocation_nodes = int(allocation["allocated_nodes"])
    same_price = math.isclose(
        allocation_price, branch_price,
        rel_tol=PRICE_REL_TOL, abs_tol=PRICE_ABS_TOL,
    )
    fewer_nodes = allocation_nodes <= branch_nodes
    result.update({
        "status": "passed" if same_price and fewer_nodes else "failed",
        "absolute_gap": absolute_gap,
        "relative_gap": absolute_gap / max(abs(branch_price), 1e-15),
        "node_savings_pct": 100.0 * (1.0 - allocation_nodes / branch_nodes),
        "error": "" if same_price and fewer_nodes
                 else "Price gap or node-allocation criterion violated",
    })
    return result



def _check_strike_monotonicity(
    index: dict, price_key: str, engine: str, scenario: str,
    steps: int, kind: str, issues: list[str],
) -> None:
    """Check that calls fall and puts rise as strike increases."""
    for exercise in ("european", "american"):
        prices = [
            float(index[(scenario, steps, strike, kind, exercise)][price_key])
            for strike in STRIKES
            if (scenario, steps, strike, kind, exercise) in index
        ]
        if len(prices) != len(STRIKES):
            continue
        for lower, higher in zip(prices, prices[1:]):
            violation = (
                lower + PRICE_ABS_TOL < higher if kind == "call"
                else lower > higher + PRICE_ABS_TOL
            )
            if violation:
                issues.append(
                    f"Strike monotonicity: {engine}, "
                    f"{scenario}, N={steps}, {kind}, {exercise}"
                )


def _check_american_dominance(
    index: dict, price_key: str, engine: str, scenario: str,
    steps: int, kind: str, issues: list[str],
) -> None:
    """American put/call value must dominate the matching European value."""
    for strike in STRIKES:
        euro = index.get((scenario, steps, strike, kind, "european"))
        amer = index.get((scenario, steps, strike, kind, "american"))
        if euro is not None and amer is not None:
            if float(amer[price_key]) + PRICE_ABS_TOL < float(euro[price_key]):
                issues.append(
                    f"American dominance: {engine}, {scenario}, "
                    f"N={steps}, {kind}, K={strike}"
                )



def check_financial_invariants(rows: list[dict[str, object]]) -> list[str]:
    """Check exercise dominance and strike monotonicity per available engine."""
    issues: list[str] = []
    for engine in ("branch", "allocation"):
        price_key = f"{engine}_price"
        status_key = f"{engine}_status"
        index = {
            (r["scenario"], r["nb_steps"], r["strike"], r["kind"], r["exercise"]): r
            for r in rows if r[status_key] == "success"
        }

        for scenario, steps, kind in product(DIVIDEND_SCENARIOS, STEP_COUNTS, OPTION_KINDS):
            _check_american_dominance(index, price_key, engine, scenario, steps, kind, issues)
            _check_strike_monotonicity(index, price_key, engine, scenario, steps, kind, issues)
    return issues


def save_csv(rows: list[dict[str, object]]) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



def _coverage_counts(rows: list[dict[str, object]], scenarios: list[str]
) -> tuple[list[int], list[int], list[int]]:
    """Summarize independently validated, unpaired and failed scenarios."""
    comparable = [
        sum(r["scenario"] == scenario and r["status"] == "passed" for r in rows)
        for scenario in scenarios
    ]
    allocation_only = [
        sum(r["scenario"] == scenario and r["status"] == "allocation_only" for r in rows)
        for scenario in scenarios
    ]
    failed = [
        sum(r["scenario"] == scenario and r["status"] == "failed" for r in rows)
        for scenario in scenarios
    ]
    return comparable, allocation_only, failed


def _save_coverage_figure(rows: list[dict[str, object]]) -> None:
    """Plot comparison coverage and node savings for comparable cases only."""
    COVERAGE_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    scenarios = list(DIVIDEND_SCENARIOS)
    pretty = ["No dividend", "One cash dividend", "Two cash dividends"]
    comparable, allocation_only, failed = _coverage_counts(rows, scenarios)
    x = list(range(len(scenarios)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width for i in x], comparable, width, label="Compared: equivalent prices")
    ax.bar(x, allocation_only, width, label="Allocation only: reference unavailable")
    ax.bar([i + width for i in x], failed, width, label="Failed validation")
    ax.set_xticks(x, pretty)
    ax.set_ylabel("Number of option configurations")
    ax.set_title("Allocation pruning: independent validation coverage")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(COVERAGE_FIGURE, dpi=200)
    plt.close(fig)



def _save_savings_figure(rows: list[dict[str, object]]) -> None:
    """Plot node savings only for matched price comparisons."""
    scenarios = list(DIVIDEND_SCENARIOS)
    pretty = ["No dividend", "One cash dividend", "Two cash dividends"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for scenario, label in zip(scenarios, pretty):
        xs: list[int] = []
        savings: list[float] = []
        for steps in STEP_COUNTS:
            match = next((
                r for r in rows
                if r["status"] == "passed"
                and r["scenario"] == scenario
                and r["nb_steps"] == steps
                and r["strike"] == 100.0
                and r["kind"] == "call"
                and r["exercise"] == "european"
            ), None)
            if match is not None:
                # Compare allocation and branch pruning on identical scenario inputs.
                xs.append(steps)
                savings.append(float(match["node_savings_pct"]))
        ax.plot(xs, savings, marker="o", linewidth=1.8, label=label)
    ax.set_xlabel("Number of time steps N")
    ax.set_ylabel("Allocated-node reduction (%)")
    ax.set_title("Allocation savings relative to branch pruning")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(SAVINGS_FIGURE, dpi=200)
    plt.close(fig)



def save_figures(rows: list[dict[str, object]]) -> None:
    """Write coverage and savings charts in their original order."""
    _save_coverage_figure(rows)
    _save_savings_figure(rows)


def save_summary(rows: list[dict[str, object]], issues: list[str]) -> None:
    comparable = [r for r in rows if r["status"] == "passed"]
    allocation_only = [r for r in rows if r["status"] == "allocation_only"]
    failed = [r for r in rows if r["status"] == "failed"]
    max_gap = max((float(r["absolute_gap"]) for r in comparable), default=math.nan)
    lines = [
        "Allocation pruning - independent and cross-option validation (v2)",
        "===============================================================",
        f"Spot={SPOT}; rate={RATE}; volatility={VOLATILITY}; maturity={MATURITY}",
        f"Threshold={PRUNING_THRESHOLD}; N={STEP_COUNTS}; strikes={STRIKES}",
        f"Absolute price tolerance={PRICE_ABS_TOL}; relative tolerance={PRICE_REL_TOL}",
        f"Configurations={len(rows)}; comparable={len(comparable)}; "
        f"allocation_only={len(allocation_only)}; failed={len(failed)}",
        # Check probability-mass conservation independently of price agreement.
        f"Allocation engine success={sum(r['allocation_status'] == 'success' for r in rows)}",
        f"Branch engine success={sum(r['branch_status'] == 'success' for r in rows)}",
        f"Maximum |price difference| among comparable cases={max_gap:.12e}",
        f"Financial invariant violations={len(issues)}",
        "",
        "IMPORTANT: A case marked allocation_only has NOT been validated against",
        "the branch price, because that numerical reference failed to build.",
        "Its independent price, local transition probabilities, and probability",
        "mass are checked, as are available financial invariants.",
        "Single-run timings are not formal speed benchmarks.",
        "",
        "Unavailable branch references (allocation still succeeded):",
    ]
    lines.extend(
        f"  {r['scenario']} N={r['nb_steps']} K={r['strike']} "
        f"{r['kind']} {r['exercise']}: allocation_price={float(r['allocation_price']):.12f} "
        f"branch_error={r['branch_error']}"
        for r in allocation_only
    )
    # Treat the documented extreme-tail reference failure as a missing comparison.
    if not allocation_only:
        lines.append("  None")
    lines.append("Failed cases:")
    lines.extend(
        f"  {r['scenario']} N={r['nb_steps']} K={r['strike']} "
        f"{r['kind']} {r['exercise']}: {r['error']}"
        for r in failed
    )
    if not failed:
        lines.append("  None")
    lines.append("Financial invariant violations:")
    lines.extend(f"  {issue}" for issue in issues)
    if not issues:
        lines.append("  None")
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log_pricing_case(index: int, total: int, case: ValidationCase,
                      result: dict[str, object]) -> None:
    """Report each independent engine result without changing validation."""
    print(
        f"[{index:3}/{total}] {case.scenario:<17} N={case.nb_steps:3} "
        f"K={case.strike:5.0f} {case.kind:4} "
        f"{'American' if case.is_american else 'European':8} "
        f"-> {result['status']}"
    )
    if result["status"] == "allocation_only":
        print(f"    Branch unavailable: {result['branch_error']}")
        print(f"    Allocation price: {float(result['allocation_price']):.12f}")
    elif result["status"] == "failed":
        print(f"    {result['error']}")


def _log_validation_summary(rows: list[dict[str, object]], issues: list[str]) -> int:
    """Report validated comparisons and return the count of failed cases."""
    comparable = sum(r["status"] == "passed" for r in rows)
    independent = sum(r["status"] == "allocation_only" for r in rows)
    failures = sum(r["status"] == "failed" for r in rows)
    print(f"\nComparison validated: {comparable}/{len(rows)} cases.")
    print(f"Allocation-only: {independent}/{len(rows)} cases (no branch reference).")
    print(f"Failed: {failures}/{len(rows)} cases.")
    print(f"Financial invariant violations: {len(issues)}")
    print(f"Results: {DATA_PATH.relative_to(ROOT)}")
    print(f"Summary: {SUMMARY_PATH.relative_to(ROOT)}")
    print(f"Figures: {COVERAGE_FIGURE.relative_to(ROOT)}, {SAVINGS_FIGURE.relative_to(ROOT)}")
    return failures


def main() -> None:
    all_cases = cases()
    rows: list[dict[str, object]] = []
    for index, case in enumerate(all_cases, start=1):
        # Do not interpret one-shot timing measurements as performance confidence bounds.
        result = run_case(case)
        rows.append(result)
        _log_pricing_case(index, len(all_cases), case, result)
    issues = check_financial_invariants(rows)
    save_csv(rows)
    save_figures(rows)
    save_summary(rows, issues)
    failures = _log_validation_summary(rows, issues)
    if failures or issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
