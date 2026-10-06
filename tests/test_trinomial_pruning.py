import math

from src.dividend import CashDividend
from src.node import Node
from src.trinomial_tree import TrinomialTree


def _terminal_top(
    tree: TrinomialTree
) -> Node:
    """Return the upper reachable terminal node."""
    current_node: Node = tree.root

    for _ in range(
        tree.nb_steps
    ):
        assert current_node.next_up is not None

        current_node = (
            current_node.next_up
        )

    return current_node


def test_root_reach_probability_is_one() -> None:
    tree = TrinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=2,
    )

    assert math.isclose(
        tree.root.reach_probability,
        1.0,
        rel_tol=1e-12,
    )


def test_terminal_reach_probabilities_sum_to_one() -> None:
    tree = TrinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=5,
        pruning_threshold=0.0,
    )

    tree.build()

    current_node: Node | None = (
        _terminal_top(
            tree
        )
    )

    total_probability = 0.0

    while current_node is not None:
        total_probability += (
            current_node.reach_probability
        )

        current_node = (
            current_node.lower_neighbor
        )

    assert math.isclose(
        total_probability,
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_low_probability_node_uses_monomial_branching() -> None:
    tree = TrinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=3,
        pruning_threshold=0.20,
    )

    tree.build()

    upper_node = tree.root.next_up

    assert upper_node is not None

    assert (
        upper_node.reach_probability
        < tree.pruning_threshold
    )

    assert upper_node.next_mid is not None

    assert (
        upper_node.next_up
        is upper_node.next_mid
    )

    assert (
        upper_node.next_down
        is upper_node.next_mid
    )

    assert upper_node.up_probability == 0.0
    assert upper_node.mid_probability == 1.0
    assert upper_node.down_probability == 0.0

    assert tree.pruned_node_count > 0


def test_pruning_preserves_total_probability() -> None:
    tree = TrinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=5,
        pruning_threshold=0.20,
    )

    tree.build()

    current_node: Node | None = (
        _terminal_top(
            tree
        )
    )

    total_probability = 0.0

    while current_node is not None:
        total_probability += (
            current_node.reach_probability
        )

        current_node = (
            current_node.lower_neighbor
        )

    assert math.isclose(
        total_probability,
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_pruning_allows_large_dividend_tree_to_build() -> None:
    tree = TrinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=300,
        dividends=[
            CashDividend(
                time=0.5,
                amount=3.0,
            )
        ],
        pruning_threshold=1e-18,
    )

    tree.build()

    assert (
        tree.pruned_node_count
        > 0
    )