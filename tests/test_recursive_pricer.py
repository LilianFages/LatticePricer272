import math

from src.binomial_tree import BinomialTree
from src.option import CallOption, Option, PutOption
from src.recursive_pricer import RecursivePricer


# Centralize the market parameters shared by recursive pricing tests.
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


# Compute directly the discounted one-step risk-neutral value.
def _one_step_expected_price(
    tree: BinomialTree,
    option: Option
) -> float:
    assert tree.root.next_up is not None
    assert tree.root.next_down is not None

    up_payoff = option.payoff(
        tree.root.next_up.price
    )
    down_payoff = option.payoff(
        tree.root.next_down.price
    )

    return math.exp(
        -tree.rate * tree.dt
    ) * (
        tree.up_probability * up_payoff
        + tree.down_probability * down_payoff
    )


# A recursive one-step call must match its direct risk-neutral value.
def test_recursive_one_step_call() -> None:
    tree = _build_tree(nb_steps=1)
    option = CallOption(strike=100.0)

    price = RecursivePricer().price(
        tree,
        option,
    )

    expected_price = _one_step_expected_price(
        tree,
        option,
    )

    assert math.isclose(
        price,
        expected_price,
        rel_tol=1e-12,
    )


# The same one-step valuation identity must hold for a put.
def test_recursive_one_step_put() -> None:
    tree = _build_tree(nb_steps=1)
    option = PutOption(strike=100.0)

    price = RecursivePricer().price(
        tree,
        option,
    )

    expected_price = _one_step_expected_price(
        tree,
        option,
    )

    assert math.isclose(
        price,
        expected_price,
        rel_tol=1e-12,
    )


# Recursive pricing must store the final option value at the root.
def test_recursive_pricer_stores_root_value() -> None:
    tree = _build_tree(nb_steps=2)
    option = CallOption(strike=100.0)

    price = RecursivePricer().price(
        tree,
        option,
    )

    assert tree.root.option_value is not None
    assert math.isclose(
        tree.root.option_value,
        price,
        rel_tol=1e-12,
    )


# A recombining node must be shared and priced only once.
def test_recombining_node_has_option_value() -> None:
    tree = _build_tree(nb_steps=2)
    option = CallOption(strike=100.0)

    RecursivePricer().price(
        tree,
        option,
    )

    first_up = tree.root.next_up
    first_down = tree.root.next_down

    assert first_up is not None
    assert first_down is not None

    # Both paths must reach the same middle node after two steps.
    middle_from_up = first_up.next_down
    middle_from_down = first_down.next_up

    assert middle_from_up is not None
    assert middle_from_up is middle_from_down
    assert middle_from_up.option_value is not None


# Two-step recursive induction must match a manual backward calculation.
def test_recursive_two_step_call_matches_manual_pricing() -> None:
    tree = _build_tree(nb_steps=2)
    option = CallOption(strike=100.0)

    price = RecursivePricer().price(
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

    # Price each first-step node from the terminal option payoffs.
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

    # Discount once more from the first column to the root.
    expected_price = discount_factor * (
        tree.up_probability * value_up
        + tree.down_probability * value_down
    )

    assert math.isclose(
        price,
        expected_price,
        rel_tol=1e-12,
    )


# European lattice prices must satisfy put-call parity.
def test_recursive_pricer_respects_put_call_parity() -> None:
    spot = 100.0
    rate = 0.02
    maturity = 1.0
    strike = 100.0

    call_tree = _build_tree(
        nb_steps=5,
        spot=spot,
        rate=rate,
        maturity=maturity,
    )
    put_tree = _build_tree(
        nb_steps=5,
        spot=spot,
        rate=rate,
        maturity=maturity,
    )

    # Price both contracts under identical market assumptions.
    pricer = RecursivePricer()

    call_price = pricer.price(
        call_tree,
        CallOption(strike=strike),
    )
    put_price = pricer.price(
        put_tree,
        PutOption(strike=strike),
    )

    # Put-call parity implies C - P = S - K exp(-rT).
    expected_difference = (
        spot
        - strike
        * math.exp(-rate * maturity)
    )

    assert math.isclose(
        call_price - put_price,
        expected_difference,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


# A deep in-the-money American put can optimally exercise immediately.
def test_recursive_american_put_can_exercise_early() -> None:
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

    # Compare European continuation with American early-exercise flexibility.
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

    # Immediate exercise gives the intrinsic value K - S = 50.
    assert american_price > european_price
    assert math.isclose(
        american_price,
        50.0,
        rel_tol=1e-12,
    )


# Without dividends and with positive rates, early call exercise has no value.
def test_recursive_american_call_equals_european_call() -> None:
    european_tree = _build_tree(
        nb_steps=10,
        rate=0.05,
    )
    american_tree = _build_tree(
        nb_steps=10,
        rate=0.05,
    )

    # Use identical contracts except for the exercise style.
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


# Negative rates can make immediate exercise optimal for an American call.
def test_recursive_american_call_can_exercise_with_negative_rate() -> None:
    european_tree = _build_tree(
        nb_steps=1,
        spot=150.0,
        rate=-0.10,
        volatility=0.01,
    )
    american_tree = _build_tree(
        nb_steps=1,
        spot=150.0,
        rate=-0.10,
        volatility=0.01,
    )

    # Isolate the effect of American exercise under a negative rate.
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

    # Immediate exercise gives the intrinsic value S - K = 50.
    assert american_price > european_price
    assert math.isclose(
        american_price,
        50.0,
        rel_tol=1e-12,
    )