import math

import pytest

from src.black_scholes import BlackScholesPricer
from src.option import CallOption, PutOption


def test_black_scholes_call_known_price() -> None:
    option = CallOption(
        strike=100.0
    )

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


def test_black_scholes_put_known_price() -> None:
    option = PutOption(
        strike=100.0
    )

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


def test_black_scholes_put_call_parity() -> None:
    spot = 100.0
    strike = 105.0
    rate = 0.03
    volatility = 0.25
    maturity = 2.0

    pricer = BlackScholesPricer()

    call_price = pricer.price(
        option=CallOption(
            strike=strike
        ),
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )

    put_price = pricer.price(
        option=PutOption(
            strike=strike
        ),
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
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


def test_black_scholes_zero_strike_call() -> None:
    price = BlackScholesPricer().price(
        option=CallOption(
            strike=0.0
        ),
        spot=100.0,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
    )

    assert price == 100.0


def test_black_scholes_zero_strike_put() -> None:
    price = BlackScholesPricer().price(
        option=PutOption(
            strike=0.0
        ),
        spot=100.0,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
    )

    assert price == 0.0


def test_black_scholes_zero_volatility() -> None:
    option = CallOption(
        strike=90.0
    )

    price = BlackScholesPricer().price(
        option=option,
        spot=100.0,
        rate=0.02,
        volatility=0.0,
        maturity=1.0,
    )

    expected_price = max(
        100.0
        - 90.0
        * math.exp(-0.02),
        0.0,
    )

    assert math.isclose(
        price,
        expected_price,
        rel_tol=1e-12,
    )


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