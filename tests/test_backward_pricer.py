import math

from src.backward_pricer import BackwardPricer
from src.binomial_tree import BinomialTree
from src.option import CallOption, Option, PutOption
from src.recursive_pricer import RecursivePricer


# Centralize the common market parameters used by the pricing tests.
def _build_tree(
    nb_steps: int,
    spot: float = 100.0,
    rate: float = 0.02,
    volatility: float = 0.20
) -> BinomialTree:
    return BinomialTree(
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=1.0,
        nb_steps=nb_steps,
    )


# Compare both algorithms on independent copies of the same lattice.
def _price_with_both_pricers(
    option: Option,
    nb_steps: int,
    spot: float = 100.0,
    rate: float = 0.02,
    volatility: float = 0.20
) -> tuple[float, float]:
    backward_tree = _build_tree(
        nb_steps,
        spot,
        rate,
        volatility,
    )
    recursive_tree = _build_tree(
        nb_steps,
        spot,
        rate,
        volatility,
    )

    backward_price = BackwardPricer().price(
        backward_tree,
        option,
    )

    # Use the recursive implementation as an independent pricing benchmark.
    recursive_price = RecursivePricer().price(
        recursive_tree,
        option,
    )

    return backward_price, recursive_price


# A one-step price must equal the discounted risk-neutral expectation.
def test_backward_one_step_call() -> None:
    tree = _build_tree(nb_steps=1)
    option = CallOption(strike=100.0)

    price = BackwardPricer().price(
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

    # Compute directly the theoretical one-step continuation value.
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


# Backward and recursive induction must agree for a European call.
def test_backward_matches_recursive_call() -> None:
    option = CallOption(strike=100.0)

    backward_price, recursive_price = _price_with_both_pricers(
        option=option,
        nb_steps=5,
    )

    assert math.isclose(
        backward_price,
        recursive_price,
        rel_tol=1e-12,
    )


# The same equivalence must hold for a European put.
def test_backward_matches_recursive_put() -> None:
    option = PutOption(strike=100.0)

    backward_price, recursive_price = _price_with_both_pricers(
        option=option,
        nb_steps=6,
    )

    assert math.isclose(
        backward_price,
        recursive_price,
        rel_tol=1e-12,
    )


# Backward induction must store the final price at the root node.
def test_backward_pricer_stores_root_value() -> None:
    tree = _build_tree(nb_steps=4)
    option = CallOption(strike=100.0)

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


# Both algorithms must also agree when early exercise is allowed.
def test_backward_matches_recursive_american_put() -> None:
    option = PutOption(
        strike=100.0,
        is_american=True,
    )

    backward_price, recursive_price = _price_with_both_pricers(
        option=option,
        nb_steps=8,
        spot=90.0,
        rate=0.05,
        volatility=0.25,
    )

    assert math.isclose(
        backward_price,
        recursive_price,
        rel_tol=1e-12,
    )


# A deep in-the-money American put can optimally exercise immediately.
def test_backward_american_put_can_exercise_early() -> None:
    european_tree = _build_tree(
        nb_steps=1,
        spot=50.0,
        rate=0.10,
        volatility=0.01,
    )
    american_tree = _build_tree(
        nb_steps=1,
        spot=50.0,
        rate=0.10,
        volatility=0.01,
    )

    european_put = PutOption(
        strike=100.0,
        is_american=False,
    )
    american_put = PutOption(
        strike=100.0,
        is_american=True,
    )

    european_price = BackwardPricer().price(
        european_tree,
        european_put,
    )

    # Compare early exercise against the corresponding European contract.
    american_price = BackwardPricer().price(
        american_tree,
        american_put,
    )

    # Immediate exercise gives the intrinsic value K - S = 50.
    assert american_price > european_price
    assert math.isclose(
        american_price,
        50.0,
        rel_tol=1e-12,
    )


# Negative rates can make early exercise relevant for American calls.
def test_backward_matches_recursive_american_call_negative_rate() -> None:
    option = CallOption(
        strike=100.0,
        is_american=True,
    )

    backward_price, recursive_price = _price_with_both_pricers(
        option=option,
        nb_steps=8,
        spot=150.0,
        rate=-0.10,
        volatility=0.15,
    )

    assert math.isclose(
        backward_price,
        recursive_price,
        rel_tol=1e-12,
    )