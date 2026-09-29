import math

from src.binomial_tree import BinomialTree
from src.option import CallOption, PutOption
from src.recursive_pricer import RecursivePricer


def test_recursive_one_step_call() -> None:
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

    pricer = RecursivePricer()

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


def test_recursive_one_step_put() -> None:
    tree = BinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=1,
    )

    option = PutOption(
        strike=100.0
    )

    pricer = RecursivePricer()

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


def test_recursive_pricer_stores_root_value() -> None:
    tree = BinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=2,
    )

    option = CallOption(
        strike=100.0
    )

    pricer = RecursivePricer()

    price = pricer.price(
        tree,
        option,
    )

    assert tree.root.option_value is not None

    assert math.isclose(
        tree.root.option_value,
        price,
        rel_tol=1e-12,
    )


def test_recombining_node_has_option_value() -> None:
    tree = BinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=2,
    )

    option = CallOption(
        strike=100.0
    )

    pricer = RecursivePricer()

    pricer.price(
        tree,
        option,
    )

    first_up = tree.root.next_up
    first_down = tree.root.next_down

    assert first_up is not None
    assert first_down is not None

    middle_from_up = first_up.next_down
    middle_from_down = first_down.next_up

    assert middle_from_up is not None

    assert (
        middle_from_up
        is middle_from_down
    )

    assert (
        middle_from_up.option_value
        is not None
    )


def test_recursive_two_step_call_matches_manual_pricing() -> None:
    tree = BinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=2,
    )

    option = CallOption(
        strike=100.0
    )

    pricer = RecursivePricer()

    price = pricer.price(
        tree,
        option,
    )

    first_up = tree.root.next_up
    first_down = tree.root.next_down

    assert first_up is not None
    assert first_down is not None

    up_up = first_up.next_up
    middle = first_up.next_down
    down_down = first_down.next_down

    assert up_up is not None
    assert middle is not None
    assert down_down is not None

    discount_factor = math.exp(
        -tree.rate * tree.dt
    )

    value_up = discount_factor * (
        tree.up_probability
        * option.payoff(up_up.price)
        + tree.down_probability
        * option.payoff(middle.price)
    )

    value_down = discount_factor * (
        tree.up_probability
        * option.payoff(middle.price)
        + tree.down_probability
        * option.payoff(down_down.price)
    )

    expected_price = discount_factor * (
        tree.up_probability * value_up
        + tree.down_probability * value_down
    )

    assert math.isclose(
        price,
        expected_price,
        rel_tol=1e-12,
    )


def test_recursive_pricer_respects_put_call_parity() -> None:
    spot = 100.0
    rate = 0.02
    volatility = 0.20
    maturity = 1.0
    strike = 100.0
    nb_steps = 5

    call_tree = BinomialTree(
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        nb_steps=nb_steps,
    )

    put_tree = BinomialTree(
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        nb_steps=nb_steps,
    )

    call = CallOption(
        strike=strike
    )

    put = PutOption(
        strike=strike
    )

    pricer = RecursivePricer()

    call_price = pricer.price(
        call_tree,
        call,
    )

    put_price = pricer.price(
        put_tree,
        put,
    )

    expected_difference = (
        spot
        - strike
        * math.exp(
            -rate * maturity
        )
    )

    assert math.isclose(
        call_price - put_price,
        expected_difference,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_recursive_american_put_can_exercise_early() -> None:
    european_tree = BinomialTree(
        spot=50.0,
        rate=0.10,
        volatility=0.01,
        maturity=1.0,
        nb_steps=1,
    )

    american_tree = BinomialTree(
        spot=50.0,
        rate=0.10,
        volatility=0.01,
        maturity=1.0,
        nb_steps=1,
    )

    european_put = PutOption(
        strike=100.0,
        is_american=False,
    )

    american_put = PutOption(
        strike=100.0,
        is_american=True,
    )

    pricer = RecursivePricer()

    european_price = pricer.price(
        european_tree,
        european_put,
    )

    american_price = pricer.price(
        american_tree,
        american_put,
    )

    assert american_price > european_price

    assert math.isclose(
        american_price,
        50.0,
        rel_tol=1e-12,
    )


def test_recursive_american_call_equals_european_call() -> None:
    european_tree = BinomialTree(
        spot=100.0,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
        nb_steps=10,
    )

    american_tree = BinomialTree(
        spot=100.0,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
        nb_steps=10,
    )

    european_call = CallOption(
        strike=100.0,
        is_american=False,
    )

    american_call = CallOption(
        strike=100.0,
        is_american=True,
    )

    pricer = RecursivePricer()

    european_price = pricer.price(
        european_tree,
        european_call,
    )

    american_price = pricer.price(
        american_tree,
        american_call,
    )

    assert math.isclose(
        american_price,
        european_price,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )