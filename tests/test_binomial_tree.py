import math

import pytest

from src.binomial_tree import BinomialTree
from src.node import Node, TrunkNode


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

# The root starts the central trunk of the lattice.
def test_root_is_trunk_node() -> None:
    tree = _build_tree(nb_steps=4)
    assert isinstance(tree.root, TrunkNode)

# In a binomial lattice, the trunk alternates up and down around the forward.
def test_binomial_trunk_alternates_up_and_down() -> None:
    tree = _build_tree(nb_steps=4)
    tree.build_trunk()
    first_trunk = tree.root.next_up
    assert isinstance(first_trunk, TrunkNode)
    second_trunk = first_trunk.next_down
    assert isinstance(second_trunk, TrunkNode)
    third_trunk = second_trunk.next_up
    assert isinstance(third_trunk, TrunkNode)
    fourth_trunk = third_trunk.next_down
    assert isinstance(fourth_trunk, TrunkNode)

# Each trunk node must retain a direct link to the previous trunk node.
def test_trunk_backward_links() -> None:
    tree = _build_tree(nb_steps=3)
    last_trunk = tree.build_trunk()
    previous = last_trunk.previous_trunk
    assert previous is not None
    previous = previous.previous_trunk
    assert previous is not None
    previous = previous.previous_trunk
    assert previous is tree.root

# Even trunk dates return exactly to the corresponding forward level.
def test_even_trunk_node_matches_forward() -> None:
    tree = _build_tree(nb_steps=4)
    tree.build_trunk()
    first_trunk = tree.root.next_up
    assert first_trunk is not None
    assert first_trunk.next_down is not None
    second_trunk = first_trunk.next_down
    # After two steps, the alternating trunk lies on S exp(2 r dt).
    expected_price = tree.spot * math.exp(
        tree.rate * 2.0 * tree.dt
    )
    assert math.isclose(
        second_trunk.price,
        expected_price,
        rel_tol=1e-12,
    )

# Backward construction must recreate the complete terminal column.
def test_backward_terminal_column_node_count() -> None:
    tree = _build_tree(nb_steps=4)
    last_trunk = tree.build_trunk()
    terminal_top = tree.build_column_backward(
        trunk_node=last_trunk,
        step=tree.nb_steps,
    )
    assert _count_column_nodes(terminal_top) == 5

# Backward column construction must reuse the existing trunk object.
def test_backward_column_reuses_trunk_node() -> None:
    tree = _build_tree(nb_steps=5)
    last_trunk = tree.build_trunk()
    terminal_top = tree.build_column_backward(
        trunk_node=last_trunk,
        step=tree.nb_steps,
    )
    current_node = terminal_top
    # Move from the top of the odd terminal column to its trunk position.
    for _ in range(tree.nb_steps // 2):
        assert current_node.lower_neighbor is not None
        current_node = current_node.lower_neighbor
    assert current_node is last_trunk

# Consecutive backward-built columns must have consistent branch connections.
def test_backward_columns_are_connected() -> None:
    tree = _build_tree(nb_steps=3)
    last_trunk = tree.build_trunk()
    next_top = tree.build_column_backward(
        trunk_node=last_trunk,
        step=3,
    )
    previous_trunk = last_trunk.previous_trunk
    assert previous_trunk is not None
    current_top = tree.build_column_backward(
        trunk_node=previous_trunk,
        step=2,
        next_top=next_top,
    )
    current_node: Node | None = current_top
    next_node: Node | None = next_top
    # Every node must connect to two adjacent nodes in the next column.
    while current_node is not None:
        assert next_node is not None
        assert current_node.next_up is next_node
        assert current_node.next_down is next_node.lower_neighbor
        current_node = current_node.lower_neighbor
        next_node = next_node.lower_neighbor

# Forward and backward construction must produce identical terminal prices.
def test_backward_terminal_prices_match_forward_tree() -> None:
    forward_tree = _build_tree(nb_steps=4)
    forward_top = _terminal_top(forward_tree)
    backward_tree = _build_tree(nb_steps=4)
    last_trunk = backward_tree.build_trunk()
    backward_top = backward_tree.build_column_backward(
        trunk_node=last_trunk,
        step=backward_tree.nb_steps,
    )
    forward_node: Node | None = forward_top
    backward_node: Node | None = backward_top
    # Compare both independently constructed terminal columns node by node.
    while forward_node is not None:
        assert backward_node is not None
        assert math.isclose(
            forward_node.price,
            backward_node.price,
            rel_tol=1e-12,
        )
        forward_node = forward_node.lower_neighbor
        backward_node = backward_node.lower_neighbor
    assert backward_node is None