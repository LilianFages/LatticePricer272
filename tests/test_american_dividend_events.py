"""Exercise at dividend jumps: independent financial and implementation tests."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from frontend.pricing_service import DividendInput, PricingRequest, price_contract
from src.dividend_exercise import aligned_event_step, dividend_exercise_steps
from src.risk.dividend_events import (
    assess_dividend_exercise,
    deterministic_american_reference,
)

FAMILIES = (
    "Hybrid", "Trinomial · Full", "Trinomial · Branch pruning",
    "Trinomial · Allocation pruning",
)
ALGORITHMS = ("Recursive", "Backward")


def contract(**changes: object) -> PricingRequest:
    base = PricingRequest(
        spot=120.0, strike=100.0, rate=0.05, volatility=0.0,
        maturity=1.0, steps=4, model="Hybrid", option_type="Call",
        exercise="American",
        dividends=(DividendInput(time=0.5, amount=20.0),),
    )
    return replace(base, **changes)


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("steps", (2, 4, 10))
def test_aligned_zero_vol_call_exercises_pre_dividend(
    model: str, algorithm: str, steps: int,
) -> None:
    request = contract(model=model, steps=steps)
    exact, event = deterministic_american_reference(request)
    assert event == "pre-dividend"
    assert exact == pytest.approx(120.0 - 100.0 * math.exp(-0.025))
    measured = price_contract(request, algorithm=algorithm)
    assert measured.price == pytest.approx(exact, abs=1e-8)
    assert assess_dividend_exercise(request, measured.price).warning is None


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_aligned_stochastic_american_call_has_event_value(
    model: str, algorithm: str,
) -> None:
    request = contract(model=model, steps=16, volatility=0.20)
    value = price_contract(request, algorithm=algorithm).price
    european = price_contract(
        replace(request, exercise="European"), algorithm=algorithm,
    ).price
    assert value >= european - 1e-10
    assert assess_dividend_exercise(request, value).warning is None


@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_off_grid_warns_and_preserves_recursive_backward_parity(
    model: str, algorithm: str,
) -> None:
    request = contract(model=model, steps=4, dividends=(
        DividendInput(time=0.6, amount=20.0),
    ))
    result = price_contract(request, algorithm=algorithm)
    assessment = assess_dividend_exercise(request, result.price)
    assert assessment.warning is not None
    assert "off-grid" in assessment.warning
    assert assessment.deterministic_reference is not None
    assert assessment.optimal_exercise_event == "pre-dividend"
    assert result.price >= 20.0


@pytest.mark.parametrize("model", FAMILIES)
def test_parity_on_grid_dividends_with_american_put(model: str) -> None:
    request = contract(
        model=model, option_type="Put", strike=135.0, steps=12,
        dividends=(DividendInput(time=0.25, amount=7.0),
                   DividendInput(time=0.75, amount=9.0)),
    )
    recursive = price_contract(request, algorithm="Recursive")
    backward = price_contract(request, algorithm="Backward")
    assert recursive.price == pytest.approx(backward.price, abs=1e-10)
    assert recursive.price >= 0.0


def test_exact_time_boundary_rejects_off_grid_date() -> None:
    assert aligned_event_step(0.50, 1.0, 4) == 2
    assert aligned_event_step(0.60, 1.0, 4) is None
    assert aligned_event_step(195 / 365, 1.0, 73) == 39


def test_no_dividend_and_non_american_have_no_event_warning() -> None:
    assert assess_dividend_exercise(contract(dividends=()), 20.0).warning is None
    assert assess_dividend_exercise(contract(exercise="European"), 20.0).warning is None


def test_deterministic_reference_identifies_post_dividend_put() -> None:
    request = contract(spot=100.0, strike=105.0, option_type="Put",
                       dividends=(DividendInput(time=0.5, amount=10.0),))
    value, event = deterministic_american_reference(request)
    expected_spot = 100.0 * math.exp(0.025) - 10.0
    assert event == "post-dividend"
    assert value == pytest.approx(math.exp(-0.025) * (105.0 - expected_spot))


def test_reference_does_not_modify_frozen_request() -> None:
    request = contract()
    original = replace(request)
    deterministic_american_reference(request)
    assert original == request


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("option_type", ("Call", "Put"))
def test_binomial_american_without_dividend_unchanged(
    algorithm: str, option_type: str,
) -> None:
    request = contract(
        model="Binomial", option_type=option_type,
        steps=8, dividends=(),
    )
    result = price_contract(request, algorithm=algorithm)
    assert math.isfinite(result.price)
    assert result.price >= 0.0
    assert assess_dividend_exercise(request, result.price).warning is None


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_aligned_american_exceeds_european_at_sigma_zero(
    algorithm: str,
) -> None:
    request = contract()
    american = price_contract(request, algorithm=algorithm).price
    european = price_contract(
        replace(request, exercise="European"), algorithm=algorithm,
    ).price
    assert american > european + 1.0
