"""Regression tests for allocation pruning across exercise and cash dividends.

Run at repository root:
    python -m pytest tests/test_allocation_pruning_robustness.py -q
"""

import math

import pytest

from src.allocation_pruned_trinomial_tree import AllocationPrunedTrinomialTree
from src.dividend import CashDividend
from src.node import Node
from src.option import CallOption, PutOption
from src.recursive_pricer import RecursivePricer
from src.trinomial_tree import TrinomialTree


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
    assert_mass_conserved(optimized)
    assert optimized.allocated_node_count > 0
