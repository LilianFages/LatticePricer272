"""Deep binomial pricing regressions for the guarded Pricing Terminal mode."""

from __future__ import annotations

import math

import pytest

from frontend.pricing_service import (
    FULL_GRAPH_MAX_STEPS,
    PricingInputError,
    PricingRequest,
    price_contract,
    validate_request,
)
from src.large_binomial_pricer import price_large_binomial


@pytest.mark.parametrize("option_type,exercise", [
    ("Call", "European"), ("Put", "European"),
    ("Call", "American"), ("Put", "American"),
])
def test_rolling_matches_object_binomial(option_type: str, exercise: str) -> None:
    request = PricingRequest(
        model="Binomial", steps=200, option_type=option_type,
        exercise=exercise, rate=0.05, volatility=0.20,
    )
    reference = price_contract(request, algorithm="Backward").price
    rolling = price_large_binomial(
        spot=request.spot, strike=request.strike,
        rate=request.rate, volatility=request.volatility,
        maturity=request.maturity, steps=request.steps,
        is_call=option_type == "Call", is_american=exercise == "American",
    )
    assert rolling == pytest.approx(reference, abs=1e-9)


def test_deep_binomial_10k_is_finite_without_full_tree() -> None:
    request = PricingRequest(model="Binomial", steps=10_000)
    result = price_contract(request, algorithm="Backward")
    assert math.isfinite(result.price)
    assert result.price == pytest.approx(8.916037278572539, abs=0.002)
    assert result.reachable_nodes == 50_015_001
    assert result.algorithm == "Backward"


def test_other_models_remain_guarded() -> None:
    with pytest.raises(PricingInputError, match="Explicit-graph"):
        validate_request(PricingRequest(
            model="Trinomial · Allocation pruning",
            steps=FULL_GRAPH_MAX_STEPS + 1,
        ))


def test_deep_binomial_cash_dividends_still_rejected() -> None:
    from frontend.pricing_service import DividendInput

    with pytest.raises(PricingInputError, match="does not support"):
        validate_request(PricingRequest(
            model="Binomial", steps=10_000,
            dividends=(DividendInput(0.5, 3.0),),
        ))


def test_deep_binomial_requires_backward() -> None:
    with pytest.raises(PricingInputError, match="requires the Backward"):
        price_contract(PricingRequest(model="Binomial", steps=10_000))
