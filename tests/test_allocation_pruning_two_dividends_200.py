"""Independently validate allocation pruning with two cash dividends at N=200.

These tests deliberately do not claim equivalence to a branch reference,
which may fail on zero-reach extreme nodes in this configuration.
"""

import math

import pytest

from src.allocation_pruned_trinomial_tree import AllocationPrunedTrinomialTree
from src.dividend import CashDividend
from src.node import Node
from src.option import CallOption, PutOption
from src.recursive_pricer import RecursivePricer


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
