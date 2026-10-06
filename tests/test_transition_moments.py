import math

import pytest

from src.dividend import CashDividend
from src.transition_moments import TransitionMoments


# Without dividend, the expected value must equal the forward price.
def test_expected_value_without_dividend() -> None:
    expected_value, _ = TransitionMoments.compute(
        spot=100.0,
        rate=0.05,
        volatility=0.20,
        step_start=0.0,
        step_end=1.0,
    )

    assert math.isclose(
        expected_value,
        105.12710963760242,
        rel_tol=1e-12,
    )


# Without dividend, the variance must match the lognormal formula.
def test_variance_without_dividend() -> None:
    _, variance = TransitionMoments.compute(
        spot=100.0,
        rate=0.05,
        volatility=0.20,
        step_start=0.0,
        step_end=1.0,
    )

    assert math.isclose(
        variance,
        451.0288078157963,
        rel_tol=1e-12,
    )


# A dividend paid at the end of the step reduces only the mean.
def test_dividend_at_step_end() -> None:
    dividend = CashDividend(
        time=1.0,
        amount=4.0,
    )

    expected_value, variance = TransitionMoments.compute(
        spot=100.0,
        rate=0.05,
        volatility=0.20,
        step_start=0.0,
        step_end=1.0,
        dividend=dividend,
    )

    assert math.isclose(
        expected_value,
        101.12710963760242,
        rel_tol=1e-12,
    )
    assert math.isclose(
        variance,
        451.0288078157963,
        rel_tol=1e-12,
    )


# A mid-step dividend affects both the expected value and the variance.
def test_mid_step_dividend() -> None:
    dividend = CashDividend(
        time=0.5,
        amount=4.0,
    )

    expected_value, variance = TransitionMoments.compute(
        spot=100.0,
        rate=0.05,
        volatility=0.20,
        step_start=0.0,
        step_end=1.0,
        dividend=dividend,
    )

    assert math.isclose(
        expected_value,
        101.0258491555047,
        rel_tol=1e-12,
    )
    assert math.isclose(
        variance,
        433.16806702629077,
        rel_tol=1e-12,
    )


# Zero volatility must produce zero conditional variance.
def test_zero_volatility_gives_zero_variance() -> None:
    dividend = CashDividend(
        time=0.5,
        amount=4.0,
    )

    _, variance = TransitionMoments.compute(
        spot=100.0,
        rate=0.05,
        volatility=0.0,
        step_start=0.0,
        step_end=1.0,
        dividend=dividend,
    )

    assert variance == 0.0


# Cash dividends cannot have negative amounts.
def test_negative_dividend_amount_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="Dividend amount cannot be negative",
    ):
        CashDividend(
            time=0.5,
            amount=-1.0,
        )


# Dividend times cannot precede the valuation date.
def test_negative_dividend_time_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="Dividend time cannot be negative",
    ):
        CashDividend(
            time=-0.5,
            amount=4.0,
        )


# A supplied dividend must belong to the transition being evaluated.
def test_dividend_outside_step_raises_error() -> None:
    dividend = CashDividend(
        time=1.5,
        amount=4.0,
    )

    with pytest.raises(
        ValueError,
        match="Dividend must occur inside the time step",
    ):
        TransitionMoments.compute(
            spot=100.0,
            rate=0.05,
            volatility=0.20,
            step_start=0.0,
            step_end=1.0,
            dividend=dividend,
        )