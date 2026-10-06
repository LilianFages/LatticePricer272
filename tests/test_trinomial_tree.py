import math

from src.node import Node, TrunkNode
from src.transition_moments import TransitionMoments
from src.trinomial_tree import TrinomialTree


# Centralize the standard market parameters used by the tests.
def _build_tree(
    nb_steps: int,
    spot: float = 100.0,
    rate: float = 0.02,
    volatility: float = 0.20,
    maturity: float = 1.0
) -> TrinomialTree:
    return TrinomialTree(
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        nb_steps=nb_steps,
    )


# Count nodes by following the vertical links of one column.
def _count_column_nodes(top_node: Node) -> int:
    node_count = 0
    current_node: Node | None = top_node

    while current_node is not None:
        node_count += 1
        current_node = (
            current_node.lower_neighbor
        )

    return node_count


# A one-step trinomial tree must create three successors.
def test_one_step_tree_creation() -> None:
    tree = _build_tree(
        nb_steps=1
    )
    tree.build()

    assert tree.root.next_up is not None
    assert tree.root.next_mid is not None
    assert tree.root.next_down is not None

    assert (
        tree.root.next_up.lower_neighbor
        is tree.root.next_mid
    )

    assert (
        tree.root.next_mid.lower_neighbor
        is tree.root.next_down
    )


# Local trinomial probabilities must form an admissible distribution.
def test_probabilities_are_admissible_and_sum_to_one() -> None:
    tree = _build_tree(
        nb_steps=1
    )
    tree.build()

    assert tree.root.up_probability is not None
    assert tree.root.mid_probability is not None
    assert tree.root.down_probability is not None

    probability_sum = (
        tree.root.up_probability
        + tree.root.mid_probability
        + tree.root.down_probability
    )

    assert math.isclose(
        probability_sum,
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    assert 0.0 <= tree.root.up_probability <= 1.0
    assert 0.0 <= tree.root.mid_probability <= 1.0
    assert 0.0 <= tree.root.down_probability <= 1.0


# The three branches must reproduce the target expected value.
def test_probabilities_match_expected_value() -> None:
    tree = _build_tree(
        nb_steps=1
    )
    tree.build()

    up_node = tree.root.next_up
    mid_node = tree.root.next_mid
    down_node = tree.root.next_down

    assert up_node is not None
    assert mid_node is not None
    assert down_node is not None

    assert tree.root.up_probability is not None
    assert tree.root.mid_probability is not None
    assert tree.root.down_probability is not None

    reconstructed_mean = (
        tree.root.up_probability * up_node.price
        + tree.root.mid_probability * mid_node.price
        + tree.root.down_probability * down_node.price
    )

    expected_value, _ = TransitionMoments.compute(
        spot=tree.spot,
        rate=tree.rate,
        volatility=tree.volatility,
        step_start=0.0,
        step_end=tree.dt,
    )

    assert math.isclose(
        reconstructed_mean,
        expected_value,
        rel_tol=1e-12,
    )


# The three branches must reproduce the target variance.
def test_probabilities_match_variance() -> None:
    tree = _build_tree(
        nb_steps=1
    )
    tree.build()

    up_node = tree.root.next_up
    mid_node = tree.root.next_mid
    down_node = tree.root.next_down

    assert up_node is not None
    assert mid_node is not None
    assert down_node is not None

    assert tree.root.up_probability is not None
    assert tree.root.mid_probability is not None
    assert tree.root.down_probability is not None

    expected_value, variance = TransitionMoments.compute(
        spot=tree.spot,
        rate=tree.rate,
        volatility=tree.volatility,
        step_start=0.0,
        step_end=tree.dt,
    )

    second_moment = (
        tree.root.up_probability * up_node.price ** 2
        + tree.root.mid_probability * mid_node.price ** 2
        + tree.root.down_probability * down_node.price ** 2
    )

    reconstructed_variance = (
        second_moment
        - expected_value ** 2
    )

    assert math.isclose(
        reconstructed_variance,
        variance,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


# Trinomial branches must recombine across adjacent starting nodes.
def test_two_step_tree_recombines() -> None:
    tree = _build_tree(
        nb_steps=2
    )
    tree.build()

    first_up = tree.root.next_up
    first_mid = tree.root.next_mid
    first_down = tree.root.next_down

    assert first_up is not None
    assert first_mid is not None
    assert first_down is not None

    # Adjacent paths reach the same nodes in the second column.
    assert first_up.next_mid is first_mid.next_up
    assert first_up.next_down is first_mid.next_mid
    assert first_mid.next_mid is first_down.next_up
    assert first_mid.next_down is first_down.next_mid


# A terminal trinomial column contains exactly 2N + 1 nodes.
def test_terminal_column_has_two_steps_plus_one_nodes() -> None:
    tree = _build_tree(
        nb_steps=3
    )
    tree.build()

    terminal_top = tree.root

    for _ in range(tree.nb_steps):
        assert terminal_top.next_up is not None
        terminal_top = terminal_top.next_up

    assert (
        _count_column_nodes(terminal_top)
        == 2 * tree.nb_steps + 1
    )


# Adjacent nodes in a column follow the geometric alpha spacing.
def test_terminal_prices_are_geometrically_spaced() -> None:
    tree = _build_tree(
        nb_steps=3
    )
    tree.build()

    upper_node: Node = tree.root

    for _ in range(tree.nb_steps):
        assert upper_node.next_up is not None
        upper_node = upper_node.next_up

    while upper_node.lower_neighbor is not None:
        lower_node = (
            upper_node.lower_neighbor
        )

        ratio = (
            upper_node.price
            / lower_node.price
        )

        assert math.isclose(
            ratio,
            tree.alpha,
            rel_tol=1e-12,
        )

        upper_node = lower_node


# Trunk nodes must remain exactly on the forward price curve.
def test_trunk_nodes_match_forward_prices() -> None:
    tree = _build_tree(
        nb_steps=3
    )
    tree.build()

    current_trunk: Node = tree.root

    for step in range(
        1,
        tree.nb_steps + 1,
    ):
        assert current_trunk.next_mid is not None

        current_trunk = (
            current_trunk.next_mid
        )

        assert isinstance(
            current_trunk,
            TrunkNode,
        )

        expected_forward = (
            tree.spot
            * math.exp(
                tree.rate
                * tree.dt
                * step
            )
        )

        assert math.isclose(
            current_trunk.price,
            expected_forward,
            rel_tol=1e-12,
        )


# Zero volatility collapses the transition to the middle branch.
def test_zero_volatility_uses_middle_branch_only() -> None:
    tree = _build_tree(
        nb_steps=1,
        volatility=0.0,
    )
    tree.build()

    assert tree.root.up_probability == 0.0
    assert tree.root.mid_probability == 1.0
    assert tree.root.down_probability == 0.0

# No-dividend probabilities must remain stable for very small time steps.
def test_large_step_count_probabilities_are_admissible() -> None:
    tree = TrinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=1000,
    )

    probability_sum = (
        tree.no_dividend_up_probability
        + tree.no_dividend_mid_probability
        + tree.no_dividend_down_probability
    )

    assert math.isclose(
        probability_sum,
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    assert 0.0 <= tree.no_dividend_up_probability <= 1.0
    assert 0.0 <= tree.no_dividend_mid_probability <= 1.0
    assert 0.0 <= tree.no_dividend_down_probability <= 1.0