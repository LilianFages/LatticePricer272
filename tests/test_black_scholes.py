import math

import pytest

from src.black_scholes import BlackScholesPricer
from src.black_scholes_terms import standardized_terms
from src.dividend_exercise import _exercise_at_ex_date
from src.option import CallOption, PutOption


# Validate the implementation against the standard Black-Scholes call price.
def test_black_scholes_call_known_price() -> None:
    option = CallOption(strike=100.0)

    price = BlackScholesPricer().price(
        option=option,
        spot=100.0,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
    )

    assert math.isclose(
        price,
        10.4505835722,
        rel_tol=1e-10,
    )


# Validate the implementation against the standard Black-Scholes put price.
def test_black_scholes_put_known_price() -> None:
    option = PutOption(strike=100.0)

    price = BlackScholesPricer().price(
        option=option,
        spot=100.0,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
    )

    assert math.isclose(
        price,
        5.5735260223,
        rel_tol=1e-10,
    )


# European call and put prices must satisfy put-call parity.
def test_black_scholes_put_call_parity() -> None:
    spot = 100.0
    strike = 105.0
    rate = 0.03
    volatility = 0.25
    maturity = 2.0
    pricer = BlackScholesPricer()

    call_price = pricer.price(
        option=CallOption(strike=strike),
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )
    put_price = pricer.price(
        option=PutOption(strike=strike),
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
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


# A zero-strike European call is worth the underlying spot.
def test_black_scholes_zero_strike_call() -> None:
    price = BlackScholesPricer().price(
        option=CallOption(strike=0.0),
        spot=100.0,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
    )

    assert price == 100.0


# A zero-strike European put has no positive payoff.
def test_black_scholes_zero_strike_put() -> None:
    price = BlackScholesPricer().price(
        option=PutOption(strike=0.0),
        spot=100.0,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
    )

    assert price == 0.0


# With zero volatility, the option value is deterministic.
def test_black_scholes_zero_volatility() -> None:
    option = CallOption(strike=90.0)

    price = BlackScholesPricer().price(
        option=option,
        spot=100.0,
        rate=0.02,
        volatility=0.0,
        maturity=1.0,
    )

    expected_price = max(
        100.0 - 90.0 * math.exp(-0.02),
        0.0,
    )

    assert math.isclose(
        price,
        expected_price,
        rel_tol=1e-12,
    )


# Black-Scholes in this project is restricted to European exercise.
def test_black_scholes_rejects_american_option() -> None:
    option = PutOption(
        strike=100.0,
        is_american=True,
    )

    with pytest.raises(
        ValueError,
        match="Black-Scholes only prices European options",
    ):
        BlackScholesPricer().price(
            option=option,
            spot=100.0,
            rate=0.02,
            volatility=0.20,
            maturity=1.0,
        )


@pytest.mark.parametrize(
    ("spot", "strike", "rate", "volatility", "maturity"),
    [
        (100.0, 100.0, 0.02, 0.20, 1.0),
        (84.0, 115.0, -0.02, 0.30, 0.50),
        (145.0, 90.0, 0.07, 0.12, 2.25),
    ],
)
def test_shared_black_scholes_terms_match_independent_formula(
    spot, strike, rate, volatility, maturity,
):
    """One common convention, including negative rates and non-ATM cases."""
    d1, d2 = standardized_terms(spot, strike, rate, volatility, maturity)
    sigma_sqrt_t = volatility * math.sqrt(maturity)
    direct_d1 = (math.log(spot / strike)
                 + (rate + volatility ** 2 / 2) * maturity) / sigma_sqrt_t
    assert (d1, d2) == pytest.approx((direct_d1, direct_d1 - sigma_sqrt_t))
    assert BlackScholesPricer._compute_d1_d2(
        spot, strike, rate, volatility, maturity,
    ) == (d1, d2)


@pytest.mark.parametrize("option_cls", [CallOption, PutOption])
@pytest.mark.parametrize("cash_adjustment", [0.0, 3.0])
def test_offgrid_exercise_uses_same_regular_bs_terms(option_cls, cash_adjustment):
    """Forced exercise expectation is BS with an explicitly shifted strike."""
    actual = _exercise_at_ex_date(
        100.0, option_cls(strike=100.0), 0.02, 0.20, 0.5,
        cash_adjustment=cash_adjustment,
    )
    reference = BlackScholesPricer().price(
        option_cls(strike=100.0 + cash_adjustment),
        spot=100.0, rate=0.02, volatility=0.20, maturity=0.5,
    )
    assert actual == pytest.approx(reference, rel=1e-13, abs=1e-12)
