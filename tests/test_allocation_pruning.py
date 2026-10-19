
import math

from src.allocation_pruned_trinomial_tree import (
    AllocationPrunedTrinomialTree,
)
from src.dividend import CashDividend
from src.node import Node
from src.option import CallOption
from src.recursive_pricer import RecursivePricer
from src.trinomial_tree import TrinomialTree


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
