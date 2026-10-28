import math

import pytest

from src.dividend import CashDividend
from src.transition_moments import TransitionMoments
from src.trinomial_probability import (
    TransitionProbabilities,
    TrinomialProbabilitySolver,
)


# A symmetric distribution provides an easy analytical benchmark.
def test_symmetric_probabilities() -> None:
    probabilities = TrinomialProbabilitySolver.solve(
        expected_value=100.0,
        variance=200.0,
        up_price=120.0,
        mid_price=100.0,
        down_price=80.0,
    )

    assert math.isclose(
        probabilities.up,
        0.25,
        rel_tol=1e-12,
    )
    assert math.isclose(
        probabilities.mid,
        0.50,
        rel_tol=1e-12,
    )
    assert math.isclose(
        probabilities.down,
        0.25,
        rel_tol=1e-12,
    )


# Trinomial probabilities must always sum to one by construction.
def test_probabilities_sum_to_one() -> None:
    probabilities = TrinomialProbabilitySolver.solve(
        expected_value=105.0,
        variance=100.0,
        up_price=120.0,
        mid_price=100.0,
        down_price=80.0,
    )

    assert math.isclose(
        probabilities.total(),
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


# The solved probabilities must reproduce the target first moment.
def test_probabilities_match_expected_value() -> None:
    expected_value = 105.0

    probabilities = TrinomialProbabilitySolver.solve(
        expected_value=expected_value,
        variance=100.0,
        up_price=120.0,
        mid_price=100.0,
        down_price=80.0,
    )

    reconstructed_expected_value = (
        probabilities.up * 120.0
        + probabilities.mid * 100.0
        + probabilities.down * 80.0
    )

    assert math.isclose(
        reconstructed_expected_value,
        expected_value,
        rel_tol=1e-12,
    )


# The solved distribution must also reproduce the target variance.
def test_probabilities_match_variance() -> None:
    expected_value = 105.0
    variance = 100.0

    probabilities = TrinomialProbabilitySolver.solve(
        expected_value=expected_value,
        variance=variance,
        up_price=120.0,
        mid_price=100.0,
        down_price=80.0,
    )

    second_moment = (
        probabilities.up * 120.0 ** 2
        + probabilities.mid * 100.0 ** 2
        + probabilities.down * 80.0 ** 2
    )

    reconstructed_variance = (
        second_moment
        - expected_value ** 2
    )

    assert math.isclose(
        reconstructed_variance,
        variance,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


# Impossible moments may solve algebraically but give invalid probabilities.
def test_impossible_distribution_is_not_admissible() -> None:
    probabilities = TrinomialProbabilitySolver.solve(
        expected_value=130.0,
        variance=0.0,
        up_price=120.0,
        mid_price=100.0,
        down_price=80.0,
    )

    assert not probabilities.is_admissible()


# A standard trinomial grid should admit a valid no-dividend distribution.
def test_standard_trinomial_grid_is_admissible() -> None:
    spot = 100.0
    volatility = 0.20
    dt = 1.0

    expected_value, variance = TransitionMoments.compute(
        spot=spot,
        rate=0.0,
        volatility=volatility,
        step_start=0.0,
        step_end=dt,
    )

    alpha = math.exp(
        volatility
        * math.sqrt(3.0 * dt)
    )

    probabilities = TrinomialProbabilitySolver.solve(
        expected_value=expected_value,
        variance=variance,
        up_price=expected_value * alpha,
        mid_price=expected_value,
        down_price=expected_value / alpha,
    )

    assert probabilities.is_admissible()


# The same solver must also work with moments including a cash dividend.
def test_dividend_transition_is_admissible() -> None:
    dividend = CashDividend(
        time=0.25,
        amount=1.0,
    )

    expected_value, variance = TransitionMoments.compute(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        step_start=0.0,
        step_end=0.25,
        dividend=dividend,
    )

    alpha = math.exp(
        0.20
        * math.sqrt(3.0 * 0.25)
    )

    probabilities = TrinomialProbabilitySolver.solve(
        expected_value=expected_value,
        variance=variance,
        up_price=expected_value * alpha,
        mid_price=expected_value,
        down_price=expected_value / alpha,
    )

    assert probabilities.is_admissible()


# Variance is a non-negative quantity.
def test_negative_variance_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="Variance cannot be negative",
    ):
        TrinomialProbabilitySolver.solve(
            expected_value=100.0,
            variance=-1.0,
            up_price=120.0,
            mid_price=100.0,
            down_price=80.0,
        )


# Node ordering is required to define up, middle and down branches.
def test_invalid_node_order_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="Node prices must satisfy up > mid > down",
    ):
        TrinomialProbabilitySolver.solve(
            expected_value=100.0,
            variance=100.0,
            up_price=100.0,
            mid_price=120.0,
            down_price=80.0,
        )