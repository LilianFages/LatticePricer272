"""Binomial lattice construction and discretization-error guarantees.

Consolidated without dropping individual regression cases.
"""

from __future__ import annotations

import math
import pytest
from src.lattices.binomial_tree import BinomialTree
from src.lattices.node import Node
from src.analytics.accuracy_metrics import BinomialErrorEstimator


# test_binomial_tree.py (tests preserved)
# Centralize the standard market parameters used throughout the tests.
def _build_tree(
    nb_steps: int,
    spot: float = 100.0,
    rate: float = 0.02,
    volatility: float = 0.20,
    maturity: float = 1.0
) -> BinomialTree:
    return BinomialTree(
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        nb_steps=nb_steps,
    )

# Return the highest node of the terminal column of a forward-built tree.
def _terminal_top(tree: BinomialTree) -> Node:
    tree.build()
    current_node: Node = tree.root
    for _ in range(tree.nb_steps):
        assert current_node.next_up is not None
        current_node = current_node.next_up
    return current_node

# Count nodes by following the vertical links of one lattice column.
def _count_column_nodes(top_node: Node) -> int:
    node_count = 0
    current_node: Node | None = top_node
    while current_node is not None:
        node_count += 1
        current_node = current_node.lower_neighbor
    return node_count

# A one-step binomial tree contains two direct successors from the root.
def test_one_step_tree_creation() -> None:
    tree = _build_tree(nb_steps=1)
    tree.build()
    assert tree.root.price == 100.0
    assert tree.root.next_up is not None
    assert tree.root.next_down is not None
    assert tree.root.next_mid is None
    # The upper and lower nodes belong to the same vertical column.
    assert tree.root.next_up.lower_neighbor is tree.root.next_down

# Risk-neutral binomial probabilities must form a complete distribution.
def test_binomial_probabilities_sum_to_one() -> None:
    tree = _build_tree(nb_steps=1)
    probability_sum = tree.up_probability + tree.down_probability
    assert math.isclose(
        probability_sum,
        1.0,
        rel_tol=1e-12,
    )

# The risk-neutral expected stock price must equal the one-step forward.
def test_expected_value_matches_forward() -> None:
    tree = _build_tree(nb_steps=1)
    tree.build()
    assert tree.root.next_up is not None
    assert tree.root.next_down is not None
    expected_price = (
        tree.up_probability * tree.root.next_up.price
        + tree.down_probability * tree.root.next_down.price
    )
    # Under the risk-neutral measure, E[S(t + dt)] = S(t) exp(r dt).
    forward_price = tree.spot * math.exp(tree.rate * tree.dt)
    assert math.isclose(
        expected_price,
        forward_price,
        rel_tol=1e-12,
    )

# Up-down and down-up paths must reach the same recombining node.
def test_two_step_tree_recombines() -> None:
    tree = _build_tree(nb_steps=2)
    tree.build()
    assert tree.root.next_up is not None
    assert tree.root.next_down is not None
    middle_from_up = tree.root.next_up.next_down
    middle_from_down = tree.root.next_down.next_up
    assert middle_from_up is not None
    assert middle_from_down is not None
    assert middle_from_up is middle_from_down

# A three-step tree must preserve recombination across every adjacent branch.
def test_three_step_tree_structure() -> None:
    tree = _build_tree(nb_steps=3)
    tree.build()
    first_up = tree.root.next_up
    first_down = tree.root.next_down
    assert first_up is not None
    assert first_down is not None
    second_top = first_up.next_up
    second_middle = first_up.next_down
    second_bottom = first_down.next_down
    assert second_top is not None
    assert second_middle is not None
    assert second_bottom is not None
    # Adjacent paths must share the same nodes after recombination.
    assert second_middle is first_down.next_up
    assert second_top.next_down is second_middle.next_up
    assert second_middle.next_down is second_bottom.next_up
    third_top = second_top.next_up
    assert third_top is not None
    assert _count_column_nodes(third_top) == 4

# An up-down path over two periods must return to the two-step forward level.
def test_up_down_price_matches_two_step_forward() -> None:
    tree = _build_tree(nb_steps=2)
    tree.build()
    first_up = tree.root.next_up
    assert first_up is not None
    assert first_up.next_down is not None
    middle_price = first_up.next_down.price
    # The middle node after two steps lies on the forward price.
    expected_price = tree.spot * math.exp(
        tree.rate * tree.maturity
    )
    assert math.isclose(
        middle_price,
        expected_price,
        rel_tol=1e-12,
    )

# A binomial terminal column contains exactly N + 1 nodes.
def test_terminal_column_has_nb_steps_plus_one_nodes() -> None:
    tree = _build_tree(nb_steps=5)
    terminal_top = _terminal_top(tree)
    node_count = _count_column_nodes(terminal_top)
    assert node_count == tree.nb_steps + 1

# Consecutive terminal prices must follow the geometric lattice spacing.
def test_terminal_prices_are_geometrically_spaced() -> None:
    tree = _build_tree(nb_steps=4)
    upper_node = _terminal_top(tree)
    while upper_node.lower_neighbor is not None:
        lower_node = upper_node.lower_neighbor
        # Two adjacent terminal nodes differ by one up/down spacing.
        price_ratio = upper_node.price / lower_node.price
        assert math.isclose(
            price_ratio,
            tree.alpha ** 2,
            rel_tol=1e-12,
        )
        upper_node = lower_node

# Spot must remain strictly positive when creating the lattice.
def test_invalid_spot_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="Spot must be strictly positive",
    ):
        _build_tree(
            nb_steps=10,
            spot=0.0,
        )

# Volatility may be zero but cannot be negative.
def test_negative_volatility_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="Volatility cannot be negative",
    ):
        _build_tree(
            nb_steps=10,
            volatility=-0.20,
        )

# The pricing horizon must be strictly positive.
def test_invalid_maturity_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="Maturity must be strictly positive",
    ):
        _build_tree(
            nb_steps=10,
            maturity=0.0,
        )

# A lattice requires at least one discrete time step.
def test_invalid_number_of_steps_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="Number of steps must be a positive integer",
    ):
        _build_tree(nb_steps=0)


# test_binomial_error.py (tests preserved)
# The theoretical relative gap must decrease exactly as 1 / N.
def test_relative_gap_is_inverse_to_number_of_steps() -> None:
    estimator = BinomialErrorEstimator()

    gap_100 = estimator.relative_gap(
        nb_steps=100,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
    )
    gap_200 = estimator.relative_gap(
        nb_steps=200,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
    )

    assert math.isclose(
        gap_100,
        2.0 * gap_200,
        rel_tol=1e-12,
    )


# The reference market case requires 41 steps for a 0.1% target.
def test_steps_for_one_tenth_percent_precision() -> None:
    estimator = BinomialErrorEstimator()

    nb_steps = estimator.steps_for_relative_precision(
        target_precision=0.001,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
    )

    assert nb_steps == 41


# The recommended step count must be the smallest one meeting the target.
def test_recommended_steps_meet_target_precision() -> None:
    estimator = BinomialErrorEstimator()
    target_precision = 0.001

    nb_steps = estimator.steps_for_relative_precision(
        target_precision=target_precision,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
    )
    achieved_precision = estimator.relative_gap(
        nb_steps=nb_steps,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
    )

    assert achieved_precision <= target_precision

    previous_precision = estimator.relative_gap(
        nb_steps=nb_steps - 1,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
    )

    assert previous_precision > target_precision


# With zero volatility, the theoretical discretization gap vanishes.
def test_zero_volatility_requires_one_step() -> None:
    estimator = BinomialErrorEstimator()

    nb_steps = estimator.steps_for_relative_precision(
        target_precision=0.001,
        rate=0.02,
        volatility=0.0,
        maturity=1.0,
    )

    assert nb_steps == 1


# A non-positive target precision is not mathematically admissible.
def test_invalid_target_precision_raises_error() -> None:
    estimator = BinomialErrorEstimator()

    with pytest.raises(
        ValueError,
        match="Target precision must be strictly positive",
    ):
        estimator.steps_for_relative_precision(
            target_precision=0.0,
            rate=0.02,
            volatility=0.20,
            maturity=1.0,
        )
