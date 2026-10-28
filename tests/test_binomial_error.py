import math

import pytest

from src.binomial_error import BinomialErrorEstimator


# The theoretical relative gap must decrease exactly as 1 / N.
def test_relative_gap_is_inverse_to_number_of_steps() -> None:
    estimator = BinomialErrorEstimator()

    gap_100 = estimator.relative_gap(
        nb_steps=100,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
    )
    gap_200 = estimator.relative_gap(
        nb_steps=200,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
    )

    assert math.isclose(
        gap_100,
        2.0 * gap_200,
        rel_tol=1e-12,
    )


# The reference market case requires 41 steps for a 0.1% target.
def test_steps_for_one_tenth_percent_precision() -> None:
    estimator = BinomialErrorEstimator()

    nb_steps = estimator.steps_for_relative_precision(
        target_precision=0.001,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
    )

    assert nb_steps == 41


# The recommended step count must be the smallest one meeting the target.
def test_recommended_steps_meet_target_precision() -> None:
    estimator = BinomialErrorEstimator()
    target_precision = 0.001

    nb_steps = estimator.steps_for_relative_precision(
        target_precision=target_precision,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
    )
    achieved_precision = estimator.relative_gap(
        nb_steps=nb_steps,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
    )

    assert achieved_precision <= target_precision

    previous_precision = estimator.relative_gap(
        nb_steps=nb_steps - 1,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
    )

    assert previous_precision > target_precision


# With zero volatility, the theoretical discretization gap vanishes.
def test_zero_volatility_requires_one_step() -> None:
    estimator = BinomialErrorEstimator()

    nb_steps = estimator.steps_for_relative_precision(
        target_precision=0.001,
        rate=0.02,
        volatility=0.0,
        maturity=1.0,
    )

    assert nb_steps == 1


# A non-positive target precision is not mathematically admissible.
def test_invalid_target_precision_raises_error() -> None:
    estimator = BinomialErrorEstimator()

    with pytest.raises(
        ValueError,
        match="Target precision must be strictly positive",
    ):
        estimator.steps_for_relative_precision(
            target_precision=0.0,
            rate=0.02,
            volatility=0.20,
            maturity=1.0,
        )