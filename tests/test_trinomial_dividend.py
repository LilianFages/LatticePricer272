import math

import pytest

from src.dividend import CashDividend
from src.node import Node
from src.option import CallOption, PutOption
from src.recursive_pricer import RecursivePricer
from src.transition_moments import TransitionMoments
from src.trinomial_tree import TrinomialTree


# A dividend must directly reduce the forward-adjusted trunk.
def test_dividend_changes_trunk_forward() -> None:
    dividend = CashDividend(
        time=0.5,
        amount=3.0,
    )

    tree = TrinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=2,
        dividends=[
            dividend
        ],
    )

    tree.build()

    first_trunk = tree.root.next_mid

    assert first_trunk is not None

    expected_first_trunk = (
        100.0
        * math.exp(0.02 * 0.5)
        - 3.0
    )

    assert math.isclose(
        first_trunk.price,
        expected_first_trunk,
        rel_tol=1e-12,
    )

    second_trunk = first_trunk.next_mid

    assert second_trunk is not None

    expected_second_trunk = (
        expected_first_trunk
        * math.exp(0.02 * 0.5)
    )

    assert math.isclose(
        second_trunk.price,
        expected_second_trunk,
        rel_tol=1e-12,
    )


# A dividend transition must still reproduce its target moments.
def test_dividend_transition_matches_target_moments() -> None:
    dividend = CashDividend(
        time=0.5,
        amount=3.0,
    )

    tree = TrinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=1,
        dividends=[
            dividend
        ],
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
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        step_start=0.0,
        step_end=1.0,
        dividend=dividend,
    )

    reconstructed_mean = (
        tree.root.up_probability * up_node.price
        + tree.root.mid_probability * mid_node.price
        + tree.root.down_probability * down_node.price
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
        reconstructed_mean,
        expected_value,
        rel_tol=1e-12,
    )

    assert math.isclose(
        reconstructed_variance,
        variance,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


# Cash dividends make trinomial probabilities node-dependent.
def test_dividend_probabilities_are_local() -> None:
    dividend = CashDividend(
        time=1.0,
        amount=3.0,
    )

    tree = TrinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=2,
        dividends=[
            dividend
        ],
    )

    tree.build()

    upper_node = tree.root.next_up
    middle_node = tree.root.next_mid
    lower_node = tree.root.next_down

    assert upper_node is not None
    assert middle_node is not None
    assert lower_node is not None

    nodes = [
        upper_node,
        middle_node,
        lower_node,
    ]

    for node in nodes:
        assert node.up_probability is not None
        assert node.mid_probability is not None
        assert node.down_probability is not None

        probability_sum = (
            node.up_probability
            + node.mid_probability
            + node.down_probability
        )

        assert math.isclose(
            probability_sum,
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

    assert not math.isclose(
        upper_node.up_probability,
        lower_node.up_probability,
        rel_tol=1e-12,
    )


# At zero rates, a cash dividend lowers the European call value.
def test_dividend_reduces_call_price_at_zero_rate() -> None:
    option = CallOption(
        strike=100.0
    )

    no_dividend_price = RecursivePricer().price(
        TrinomialTree(
            spot=100.0,
            rate=0.0,
            volatility=0.20,
            maturity=1.0,
            nb_steps=20,
        ),
        option,
    )

    dividend_price = RecursivePricer().price(
        TrinomialTree(
            spot=100.0,
            rate=0.0,
            volatility=0.20,
            maturity=1.0,
            nb_steps=20,
            dividends=[
                CashDividend(
                    time=0.5,
                    amount=3.0,
                )
            ],
        ),
        option,
    )

    assert dividend_price < no_dividend_price


# At zero rates, a cash dividend raises the European put value.
def test_dividend_increases_put_price_at_zero_rate() -> None:
    option = PutOption(
        strike=100.0
    )

    no_dividend_price = RecursivePricer().price(
        TrinomialTree(
            spot=100.0,
            rate=0.0,
            volatility=0.20,
            maturity=1.0,
            nb_steps=20,
        ),
        option,
    )

    dividend_price = RecursivePricer().price(
        TrinomialTree(
            spot=100.0,
            rate=0.0,
            volatility=0.20,
            maturity=1.0,
            nb_steps=20,
            dividends=[
                CashDividend(
                    time=0.5,
                    amount=3.0,
                )
            ],
        ),
        option,
    )

    assert dividend_price > no_dividend_price


# A cash dividend may require the lattice to expand beyond 2N + 1 nodes.
def test_dividend_can_expand_trinomial_column() -> None:
    tree = TrinomialTree(
        spot=100.0,
        rate=0.0,
        volatility=0.20,
        maturity=1.0,
        nb_steps=20,
        dividends=[
            CashDividend(
                time=0.5,
                amount=7.0,
            )
        ],
    )

    tree.build()

    terminal_top: Node = tree.root

    for _ in range(
        tree.nb_steps
    ):
        assert terminal_top.next_up is not None
        terminal_top = terminal_top.next_up

    node_count = 0
    current_node: Node | None = terminal_top

    while current_node is not None:
        node_count += 1

        current_node = (
            current_node.lower_neighbor
        )

    # Without dividend the terminal column would contain 2N + 1 nodes.
    assert node_count > 2 * tree.nb_steps + 1


# Several discrete dividends must successively reduce the trunk.
def test_multiple_dividends_change_trunk() -> None:
    tree = TrinomialTree(
        spot=100.0,
        rate=0.0,
        volatility=0.20,
        maturity=1.0,
        nb_steps=4,
        dividends=[
            CashDividend(
                time=0.25,
                amount=2.0,
            ),
            CashDividend(
                time=0.75,
                amount=3.0,
            ),
        ],
    )

    tree.build()

    first_trunk = tree.root.next_mid
    assert first_trunk is not None

    assert math.isclose(
        first_trunk.price,
        98.0,
        rel_tol=1e-12,
    )

    second_trunk = first_trunk.next_mid
    assert second_trunk is not None

    assert math.isclose(
        second_trunk.price,
        98.0,
        rel_tol=1e-12,
    )

    third_trunk = second_trunk.next_mid
    assert third_trunk is not None

    assert math.isclose(
        third_trunk.price,
        95.0,
        rel_tol=1e-12,
    )


# Several cash dividends must lower a European call at zero rate.
def test_multiple_dividends_reduce_call_price() -> None:
    option = CallOption(
        strike=100.0
    )

    no_dividend_price = RecursivePricer().price(
        TrinomialTree(
            spot=100.0,
            rate=0.0,
            volatility=0.20,
            maturity=1.0,
            nb_steps=40,
        ),
        option,
    )

    dividend_price = RecursivePricer().price(
        TrinomialTree(
            spot=100.0,
            rate=0.0,
            volatility=0.20,
            maturity=1.0,
            nb_steps=40,
            dividends=[
                CashDividend(
                    time=0.25,
                    amount=2.0,
                ),
                CashDividend(
                    time=0.75,
                    amount=3.0,
                ),
            ],
        ),
        option,
    )

    assert dividend_price < no_dividend_price


# Two dividends inside the same lattice step are currently rejected.
def test_two_dividends_in_same_step_raise_error() -> None:
    tree = TrinomialTree(
        spot=100.0,
        rate=0.0,
        volatility=0.20,
        maturity=1.0,
        nb_steps=2,
        dividends=[
            CashDividend(
                time=0.20,
                amount=1.0,
            ),
            CashDividend(
                time=0.40,
                amount=1.0,
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Only one dividend per time step is supported",
    ):
        tree.build()