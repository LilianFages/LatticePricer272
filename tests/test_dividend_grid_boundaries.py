"""Dividend ex-date boundaries and true within-step timing regression tests."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from frontend.pricing_service import (
    DividendInput,
    PricingInputError,
    PricingRequest,
    _dividend_step,
    price_contract,
    validate_request,
)
from src.dividend import CashDividend
from src.dividend_timing import dividend_grid_step, dividend_time_in_step
from src.transition_moments import TransitionMoments


def test_calendar_fraction_matches_exact_grid_endpoint() -> None:
    # 195/365 = 39/73 mathematically, but not always as binary floats.
    assert 195 / 365 > 39 * (1.0 / 73)
    assert dividend_grid_step(195 / 365, 1.0, 73) == 39
    assert _dividend_step(195 / 365, 1.0, 73) == 39


@pytest.mark.parametrize("k", (1, 7, 39, 73))
def test_grid_boundaries_are_right_closed(k: int) -> None:
    date_on_boundary = k / 73
    assert dividend_grid_step(date_on_boundary, 1.0, 73) == k
    assert dividend_grid_step(195 / 365, 1.0, 73) == 39


def test_genuine_off_grid_date_moves_to_next_step() -> None:
    assert dividend_grid_step(0.5, 1.0, 4) == 2
    assert dividend_grid_step(0.6, 1.0, 4) == 3
    assert dividend_grid_step(0.5 + 1e-8, 1.0, 4) == 3
    assert dividend_grid_step(0.5 - 1e-8, 1.0, 4) == 2


def test_only_binary_roundoff_is_snapped() -> None:
    end = 39 * (1.0 / 73)
    date = 195 / 365
    assert date > end
    assert dividend_time_in_step(date, 38 / 73, end) == end
    assert dividend_time_in_step(0.6, 0.5, 0.75) == 0.6
    with pytest.raises(ValueError, match="time step"):
        dividend_time_in_step(end + 1e-8, 38 / 73, end)
    with pytest.raises(ValueError, match="time step"):
        dividend_time_in_step(0.5, 0.5, 0.75)


@pytest.mark.parametrize("time", (0.0, -0.1, 1.01, float("nan")))
def test_invalid_event_times_rejected(time: float) -> None:
    with pytest.raises(ValueError):
        dividend_grid_step(time, 1.0, 100)


def test_exact_boundary_moments_match_mathematically_equal_calendar() -> None:
    start = 38 * (1.0 / 73)
    end = 39 * (1.0 / 73)
    kwargs = dict(
        spot=100.0, rate=0.03, volatility=0.2,
        step_start=start, step_end=end,
    )
    endpoint = TransitionMoments.compute(
        **kwargs, dividend=CashDividend(time=end, amount=4.0),
    )
    calendar = TransitionMoments.compute(
        **kwargs, dividend=CashDividend(time=195 / 365, amount=4.0),
    )
    assert calendar == endpoint
    assert endpoint[0] == pytest.approx(100 * math.exp(0.03 / 73) - 4.0)


def test_within_step_moments_use_real_date_not_end_of_step() -> None:
    kwargs = dict(
        spot=100.0, rate=0.03, volatility=0.2,
        step_start=0.5, step_end=0.75,
    )
    middle = TransitionMoments.compute(
        **kwargs, dividend=CashDividend(time=0.6, amount=4.0),
    )
    endpoint = TransitionMoments.compute(
        **kwargs, dividend=CashDividend(time=0.75, amount=4.0),
    )
    expected_mean = 100 * math.exp(0.03 * 0.25) - 4 * math.exp(0.03 * 0.15)
    weight = 0.15 / 0.25
    expected_variance = (
        (100 - weight * 4) ** 2
        * math.exp(2 * 0.03 * 0.25)
        * math.expm1(0.2 ** 2 * 0.25)
    )
    assert middle[0] == pytest.approx(expected_mean)
    assert middle[1] == pytest.approx(expected_variance)
    assert middle[0] != pytest.approx(endpoint[0], abs=1e-7)


def test_validation_uses_same_event_step_as_tree() -> None:
    req = PricingRequest(
        model="Hybrid", steps=73,
        dividends=(DividendInput(time=195 / 365, amount=3.0),),
    )
    validate_request(req)
    with pytest.raises(PricingInputError, match="same lattice time step"):
        validate_request(replace(req, dividends=(
            DividendInput(time=195 / 365, amount=3.0),
            DividendInput(time=0.53, amount=2.0),
        )))


@pytest.mark.parametrize("model", (
    "Hybrid", "Trinomial · Full", "Trinomial · Branch pruning",
    "Trinomial · Allocation pruning",
))
@pytest.mark.parametrize("algorithm", ("Recursive", "Backward"))
@pytest.mark.parametrize("exercise", ("European", "American"))
def test_equivalent_on_grid_dates_reprice_identically(
    model: str, algorithm: str, exercise: str,
) -> None:
    req = PricingRequest(
        model=model, option_type="Call", exercise=exercise,
        steps=73, spot=100.0, strike=102.0,
        dividends=(DividendInput(time=39 * (1.0 / 73), amount=2.0),),
    )
    from_calendar = replace(
        req, dividends=(DividendInput(time=195 / 365, amount=2.0),),
    )
    exact = price_contract(req, algorithm=algorithm).price
    calendar = price_contract(from_calendar, algorithm=algorithm).price
    assert math.isclose(exact, calendar, rel_tol=1e-12, abs_tol=1e-12)
