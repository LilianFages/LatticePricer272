import math

from src.black_scholes import BlackScholesPricer
from src.option import CallOption, PutOption
from src.recursive_pricer import RecursivePricer
from src.trinomial_tree import TrinomialTree


# A one-step trinomial price must equal its direct discounted expectation.
def test_recursive_one_step_trinomial_call() -> None:
    tree = TrinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=1,
    )
    option = CallOption(
        strike=100.0
    )

    price = RecursivePricer().price(
        tree,
        option,
    )

    up_node = tree.root.next_up
    mid_node = tree.root.next_mid
    down_node = tree.root.next_down

    assert up_node is not None
    assert mid_node is not None
    assert down_node is not None

    assert tree.root.up_probability is not None
    assert tree.root.mid_probability is not None
    assert tree.root.down_probability is not None

    expected_price = math.exp(
        -tree.rate * tree.dt
    ) * (
        tree.root.up_probability
        * option.payoff(up_node.price)
        + tree.root.mid_probability
        * option.payoff(mid_node.price)
        + tree.root.down_probability
        * option.payoff(down_node.price)
    )

    assert math.isclose(
        price,
        expected_price,
        rel_tol=1e-12,
    )


# European trinomial prices must satisfy put-call parity.
def test_recursive_trinomial_put_call_parity() -> None:
    spot = 100.0
    strike = 105.0
    rate = 0.03
    maturity = 2.0

    call_tree = TrinomialTree(
        spot=spot,
        rate=rate,
        volatility=0.25,
        maturity=maturity,
        nb_steps=10,
    )
    put_tree = TrinomialTree(
        spot=spot,
        rate=rate,
        volatility=0.25,
        maturity=maturity,
        nb_steps=10,
    )

    call_price = RecursivePricer().price(
        call_tree,
        CallOption(strike=strike),
    )
    put_price = RecursivePricer().price(
        put_tree,
        PutOption(strike=strike),
    )

    expected_difference = (
        spot
        - strike
        * math.exp(-rate * maturity)
    )

    assert math.isclose(
        call_price - put_price,
        expected_difference,
        rel_tol=1e-10,
        abs_tol=1e-10,
    )


# A sufficiently refined trinomial tree must approach Black-Scholes.
def test_recursive_trinomial_converges_to_black_scholes() -> None:
    spot = 100.0
    strike = 100.0
    rate = 0.05
    volatility = 0.20
    maturity = 1.0

    option = CallOption(
        strike=strike
    )

    tree_price = RecursivePricer().price(
        TrinomialTree(
            spot=spot,
            rate=rate,
            volatility=volatility,
            maturity=maturity,
            nb_steps=50,
        ),
        option,
    )

    black_scholes_price = BlackScholesPricer().price(
        option=option,
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    assert math.isclose(
        tree_price,
        black_scholes_price,
        abs_tol=0.05,
    )