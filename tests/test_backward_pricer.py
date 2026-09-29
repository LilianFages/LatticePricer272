import math

from src.backward_pricer import BackwardPricer
from src.binomial_tree import BinomialTree
from src.option import CallOption, PutOption
from src.recursive_pricer import RecursivePricer


def test_backward_one_step_call() -> None:
    tree = BinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=1,
    )

    option = CallOption(
        strike=100.0
    )

    pricer = BackwardPricer()

    price = pricer.price(
        tree,
        option,
    )

    assert tree.root.next_up is not None
    assert tree.root.next_down is not None

    up_payoff = option.payoff(
        tree.root.next_up.price
    )

    down_payoff = option.payoff(
        tree.root.next_down.price
    )

    expected_price = math.exp(
        -tree.rate * tree.dt
    ) * (
        tree.up_probability * up_payoff
        + tree.down_probability * down_payoff
    )

    assert math.isclose(
        price,
        expected_price,
        rel_tol=1e-12,
    )


def test_backward_matches_recursive_call() -> None:
    backward_tree = BinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=5,
    )

    recursive_tree = BinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=5,
    )

    option = CallOption(
        strike=100.0
    )

    backward_price = BackwardPricer().price(
        backward_tree,
        option,
    )

    recursive_price = RecursivePricer().price(
        recursive_tree,
        option,
    )

    assert math.isclose(
        backward_price,
        recursive_price,
        rel_tol=1e-12,
    )


def test_backward_matches_recursive_put() -> None:
    backward_tree = BinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=6,
    )

    recursive_tree = BinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=6,
    )

    option = PutOption(
        strike=100.0
    )

    backward_price = BackwardPricer().price(
        backward_tree,
        option,
    )

    recursive_price = RecursivePricer().price(
        recursive_tree,
        option,
    )

    assert math.isclose(
        backward_price,
        recursive_price,
        rel_tol=1e-12,
    )


def test_backward_pricer_stores_root_value() -> None:
    tree = BinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=4,
    )

    option = CallOption(
        strike=100.0
    )

    price = BackwardPricer().price(
        tree,
        option,
    )

    assert tree.root.option_value is not None

    assert math.isclose(
        tree.root.option_value,
        price,
        rel_tol=1e-12,
    )