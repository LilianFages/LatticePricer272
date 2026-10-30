"""Allocation-pruned trinomial pricing, probability mass and robustness.

Consolidated without dropping individual regression cases.
"""

from __future__ import annotations

import math
from src.allocation_pruned_trinomial_tree import (
    AllocationPrunedTrinomialTree,
)
from src.dividend import CashDividend
from src.node import Node
from src.option import CallOption
from src.recursive_pricer import RecursivePricer
from src.trinomial_tree import TrinomialTree
import pytest
from src.allocation_pruned_trinomial_tree import AllocationPrunedTrinomialTree
from src.option import CallOption, PutOption


# test_allocation_pruning.py (tests preserved)
def _build_tree(
    nb_steps: int,
    threshold: float = 1e-18,
    with_dividend: bool = False
) -> AllocationPrunedTrinomialTree:
    """Return a standard allocation-pruned test tree."""
    dividends: list[CashDividend] = []

    if with_dividend:
        dividends.append(
            CashDividend(
                time=0.5,
                amount=3.0,
            )
        )

    return AllocationPrunedTrinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=nb_steps,
        dividends=dividends,
        pruning_threshold=threshold,
    )


# Check lattice geometry, valuation parity and allocation independently.
def test_zero_threshold_keeps_full_allocation() -> None:
    tree = _build_tree(
        nb_steps=10,
        threshold=0.0,
    )

    tree.build()

    # Without dividends: 1 + 3 + ... + (2N + 1).
    assert tree.allocated_node_count == 121
    assert tree.pruned_node_count == 0


def test_allocation_pruning_reduces_node_count() -> None:
    tree = _build_tree(
        nb_steps=100,
    )

    tree.build()

    full_allocation = (
        tree.nb_steps + 1
    ) ** 2

    assert tree.allocated_node_count < full_allocation
    # Compare allocated-node counts at identical discretization parameters.
    assert tree.pruned_node_count > 0


def test_allocation_pruning_preserves_total_probability() -> None:
    tree = _build_tree(
        nb_steps=100,
        with_dividend=True,
    )

    tree.build()

    current_node: Node | None = tree.terminal_top
    total_probability = 0.0

    while current_node is not None:
        total_probability += current_node.reach_probability
        # Reach probabilities must still sum to one at expiry.
        current_node = current_node.lower_neighbor

    assert math.isclose(
        total_probability,
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_allocation_pruning_preserves_call_price() -> None:
    dividends = [
        CashDividend(
            time=0.5,
            amount=3.0,
        )
    ]

    # The pruned and full grids should price the same call within tolerance.
    reference_price = RecursivePricer().price(
        TrinomialTree(
            spot=100.0,
            rate=0.02,
            volatility=0.20,
            maturity=1.0,
            nb_steps=100,
            dividends=dividends,
            pruning_threshold=1e-18,
        ),
        CallOption(
            strike=100.0
        ),
    )

    # The pruned and full grids should price the same call within tolerance.
    pruned_price = RecursivePricer().price(
        _build_tree(
            nb_steps=100,
            with_dividend=True,
        ),
        CallOption(
            strike=100.0
        ),
    )

    assert math.isclose(
        pruned_price,
        reference_price,
        rel_tol=1e-10,
        abs_tol=1e-10,
    )


# The reduced grid must still support the specified dividend scenario.
def test_allocation_pruning_builds_large_dividend_tree() -> None:
    tree = _build_tree(
        nb_steps=300,
        with_dividend=True,
    )

    price = RecursivePricer().price(
        tree,
        CallOption(
            strike=100.0
        ),
    )

    assert math.isfinite(price)
    assert price > 0.0
    assert tree.pruned_node_count > 0
    assert tree.allocated_node_count > 0


# test_allocation_pruning_robustness.py (tests preserved)
SCENARIOS = [
    (),
    ((0.50, 3.0),),
    ((0.25, 2.0), (0.75, 3.0)),
]


def dividends_from_specs(specs: tuple[tuple[float, float], ...]) -> list[CashDividend]:
    return [CashDividend(time=time, amount=amount) for time, amount in specs]


def assert_mass_conserved(tree: TrinomialTree) -> None:
    """Check risk-neutral probability mass at each date of the graph."""
    frontier: dict[int, Node] = {id(tree.root): tree.root}
    for step in range(tree.nb_steps + 1):
        mass = math.fsum(node.reach_probability for node in frontier.values())
        assert math.isclose(mass, 1.0, rel_tol=0.0, abs_tol=1e-10)
        if step == tree.nb_steps:
            break
        # Check lattice geometry, valuation parity and allocation independently.
        following: dict[int, Node] = {}
        for node in frontier.values():
            edges = (
                (node.next_up, node.up_probability),
                (node.next_mid, node.mid_probability),
                (node.next_down, node.down_probability),
            )
            local_total = 0.0
            for successor, probability in edges:
                assert successor is not None
                assert probability is not None
                # Measure terminal probability mass over distinct linked nodes.
                assert -1e-12 <= probability <= 1.0 + 1e-12
                local_total += probability
                following[id(successor)] = successor
            assert math.isclose(local_total, 1.0, rel_tol=0.0, abs_tol=1e-12)
        frontier = following


@pytest.mark.parametrize("specs", SCENARIOS, ids=["none", "one", "two"])
@pytest.mark.parametrize("steps", [60, 160])
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("option_kind", ["call", "put"])
@pytest.mark.parametrize("american", [False, True], ids=["european", "american"])
def test_allocation_pruning_preserves_prices_and_mass(
    specs: tuple[tuple[float, float], ...],
    steps: int,
    strike: float,
    option_kind: str,
    american: bool,
) -> None:
    # Check lattice geometry, valuation parity and allocation independently.
    """Compare two independently built recombining trinomial graphs."""
    common = dict(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=steps,
        pruning_threshold=1e-18,
    )

    full = TrinomialTree(
        **common,
        dividends=dividends_from_specs(specs),
    )
    optimized = AllocationPrunedTrinomialTree(
        **common,
        dividends=dividends_from_specs(specs),
    )

    # Use matched full and pruned scenarios to check mass and price independently.
    option_type = CallOption if option_kind == "call" else PutOption
    full_price = RecursivePricer().price(
        full,
        option_type(strike=strike, is_american=american),
    )
    optimized_price = RecursivePricer().price(
        optimized,
        option_type(strike=strike, is_american=american),
    )

    assert math.isfinite(full_price)
    assert math.isfinite(optimized_price)
    assert math.isclose(
        optimized_price,
        full_price,
        rel_tol=1e-10,
        abs_tol=1e-9,
    )
    assert_mass_conserved(full)
    # Verify the expected numerical or workflow invariant on this case.
    assert_mass_conserved(optimized)
    assert optimized.allocated_node_count > 0


# test_allocation_pruning_two_dividends_200.py (tests preserved)
def assert_probability_mass(tree: AllocationPrunedTrinomialTree) -> None:
    frontier: dict[int, Node] = {id(tree.root): tree.root}
    for step in range(tree.nb_steps + 1):
        total = math.fsum(node.reach_probability for node in frontier.values())
        assert math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-10)
        # Conserve probability when branches are redirected to surviving nodes.
        if step == tree.nb_steps:
            break
        new_frontier: dict[int, Node] = {}
        for node in frontier.values():
            transition_total = 0.0
            for successor, prob in (
                (node.next_up, node.up_probability),
                (node.next_mid, node.mid_probability),
                (node.next_down, node.down_probability),
            ):
                assert successor is not None
                assert prob is not None
                assert -1e-12 <= prob <= 1.0 + 1e-12
                transition_total += prob
                new_frontier[id(successor)] = successor
            assert math.isclose(transition_total, 1.0, rel_tol=0.0, abs_tol=1e-12)
        # Conserve probability when branches are redirected to surviving nodes.
        frontier = new_frontier


@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("american", [False, True], ids=["european", "american"])
def test_allocation_only_two_dividends_200(
    strike: float, kind: str, american: bool,
) -> None:
    tree = AllocationPrunedTrinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=200,
        dividends=[
            CashDividend(time=0.25, amount=2.0),
            CashDividend(time=0.75, amount=3.0),
        ],
        pruning_threshold=1e-18,
    )
    # The two-dividend N=200 case must complete without fabricating a baseline.
    option_class = CallOption if kind == "call" else PutOption
    price = RecursivePricer().price(
        tree, option_class(strike=strike, is_american=american),
    )
    assert math.isfinite(price)
    assert price >= 0.0
    assert tree.allocated_node_count > 0
    assert_probability_mass(tree)
    if strike == 100.0 and kind == "call" and not american:
        # Regression checkpoint from the first observed run,
        # not an independent reference for financial pricing accuracy.
        assert math.isclose(price, 6.514698734290, rel_tol=0.0, abs_tol=1e-8)
