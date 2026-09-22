import math

from src.binomial_tree import BinomialTree


def test_one_step_tree_creation() -> None:
    tree = BinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=1,
    )

    tree.build()

    assert tree.root.price == 100.0

    assert tree.root.next_up is not None
    assert tree.root.next_down is not None
    assert tree.root.next_mid is None

    assert (
        tree.root.next_up.lower_neighbor
        is tree.root.next_down
    )


def test_binomial_probabilities_sum_to_one() -> None:
    tree = BinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=1,
    )

    assert math.isclose(
        tree.up_probability + tree.down_probability,
        1.0,
        rel_tol=1e-12,
    )


def test_expected_value_matches_forward() -> None:
    tree = BinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=1,
    )

    tree.build()

    assert tree.root.next_up is not None
    assert tree.root.next_down is not None

    expected_price = (
        tree.up_probability * tree.root.next_up.price
        + tree.down_probability * tree.root.next_down.price
    )

    forward_price = (
        tree.spot
        * math.exp(tree.rate * tree.dt)
    )

    assert math.isclose(
        expected_price,
        forward_price,
        rel_tol=1e-12,
    )


def test_two_step_tree_recombines() -> None:
    tree = BinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=2,
    )

    tree.build()

    assert tree.root.next_up is not None
    assert tree.root.next_down is not None

    assert tree.root.next_up.next_down is not None
    assert tree.root.next_down.next_up is not None

    assert (
        tree.root.next_up.next_down
        is tree.root.next_down.next_up
    )


def test_three_step_tree_structure() -> None:
    tree = BinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=3,
    )

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

    assert second_middle is first_down.next_up

    assert (
        second_top.next_down
        is second_middle.next_up
    )

    assert (
        second_middle.next_down
        is second_bottom.next_up
    )

    third_top = second_top.next_up

    assert third_top is not None

    node_count = 0
    current_node = third_top

    while current_node is not None:
        node_count += 1
        current_node = current_node.lower_neighbor

    assert node_count == 4

def test_up_down_price_matches_two_step_forward() -> None:
    tree = BinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=2,
    )

    tree.build()

    first_up = tree.root.next_up

    assert first_up is not None
    assert first_up.next_down is not None

    middle_price = first_up.next_down.price

    expected_price = (
        tree.spot
        * math.exp(tree.rate * tree.maturity)
    )

    assert math.isclose(
        middle_price,
        expected_price,
        rel_tol=1e-12,
    )


def test_terminal_column_has_nb_steps_plus_one_nodes() -> None:
    tree = BinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=5,
    )

    tree.build()

    terminal_top = tree.root

    for _ in range(tree.nb_steps):
        assert terminal_top.next_up is not None
        terminal_top = terminal_top.next_up

    node_count = 0
    current_node = terminal_top

    while current_node is not None:
        node_count += 1
        current_node = current_node.lower_neighbor

    assert node_count == tree.nb_steps + 1


def test_terminal_prices_are_geometrically_spaced() -> None:
    tree = BinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=4,
    )

    tree.build()

    terminal_top = tree.root

    for _ in range(tree.nb_steps):
        assert terminal_top.next_up is not None
        terminal_top = terminal_top.next_up

    upper_node = terminal_top

    while upper_node.lower_neighbor is not None:
        lower_node = upper_node.lower_neighbor

        price_ratio = (
            upper_node.price
            / lower_node.price
        )

        assert math.isclose(
            price_ratio,
            tree.alpha ** 2,
            rel_tol=1e-12,
        )

        upper_node = lower_node