"""Backward/Recursive routing through pricing and the seven-Greek risk API."""

from __future__ import annotations

import csv
import io
import math
from dataclasses import replace
from functools import partial

import pytest

from frontend.greeks_service import calculate_greeks
from frontend.pricing_service import (
    PRICING_ALGORITHMS,
    DividendInput,
    PricingInputError,
    PricingRequest,
    PricingResult,
    price_contract,
    ticket_csv,
)
from frontend.trading_risk import compute_desk_risk, compute_greek_curve

MODELS = (
    "Binomial",
    "Hybrid",
    "Trinomial · Full",
    "Trinomial · Branch pruning",
    "Trinomial · Allocation pruning",
)


def make_request(
    model: str,
    option_type: str,
    exercise: str,
    *,
    dividends: tuple[DividendInput, ...] = (),
    steps: int = 14,
) -> PricingRequest:
    return PricingRequest(
        model=model,
        option_type=option_type,
        exercise=exercise,
        dividends=dividends,
        steps=steps,
        spot=103.0,
        strike=100.0,
        rate=0.02,
        volatility=0.22,
        maturity=1.0,
    )


# Recursive and backward algorithms should agree on prices and node accounting.
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("option_type", ("Call", "Put"))
@pytest.mark.parametrize("exercise", ("European", "American"))
def test_public_api_parity_no_dividends(
    model: str, option_type: str, exercise: str,
) -> None:
    request = make_request(model, option_type, exercise)
    recursive = price_contract(request, algorithm="Recursive")
    backward = price_contract(request, algorithm="Backward")
    assert math.isclose(recursive.price, backward.price, rel_tol=1e-12, abs_tol=1e-12)
    assert recursive.request == backward.request == request
    assert recursive.algorithm == "Recursive"
    assert backward.algorithm == "Backward"
    assert recursive.reachable_nodes == backward.reachable_nodes
    assert recursive.allocated_nodes == backward.allocated_nodes
    assert recursive.pruned_transitions == backward.pruned_transitions


# Cash-dividend parity also guards ex-date event ordering across engines.
@pytest.mark.parametrize("model", MODELS[1:])
@pytest.mark.parametrize("option_type", ("Call", "Put"))
@pytest.mark.parametrize("exercise", ("European", "American"))
def test_public_api_parity_cash_dividends(
    model: str, option_type: str, exercise: str,
) -> None:
    request = make_request(
        model, option_type, exercise,
        dividends=(DividendInput(time=0.5, amount=3.0),),
        steps=20,
    )
    recursive = price_contract(request, algorithm="Recursive")
    backward = price_contract(request, algorithm="Backward")
    assert math.isclose(recursive.price, backward.price, rel_tol=1e-12, abs_tol=1e-12)
    assert recursive.reachable_nodes == backward.reachable_nodes
    assert recursive.allocated_nodes == backward.allocated_nodes
    assert recursive.pruned_transitions == backward.pruned_transitions


# Backward pricing is opt-in and must not change the public default.
def test_default_stays_recursive() -> None:
    request = PricingRequest(steps=8)
    default = price_contract(request)
    explicit = price_contract(request, algorithm="Recursive")
    assert default.algorithm == explicit.algorithm == "Recursive"
    assert default.price == explicit.price


@pytest.mark.parametrize("bad", ("Unknown", "forward", "", None))
def test_invalid_algorithm_is_rejected(bad: object) -> None:
    with pytest.raises(PricingInputError, match="pricing algorithm"):
        price_contract(PricingRequest(steps=6), algorithm=bad)


def test_csv_exports_executed_algorithm() -> None:
    result = price_contract(PricingRequest(steps=7), algorithm="Backward")
    fields = dict(csv.reader(io.StringIO(ticket_csv(result))))
    assert fields["pricing_algorithm"] == "Backward"
    assert fields["price"] == str(result.price)


# Algorithm selection is execution metadata, never mutable contract state.
def test_immutable_request_is_shared_between_methods() -> None:
    request = PricingRequest(
        model="Hybrid", steps=14, option_type="Put", exercise="American",
        dividends=(DividendInput(time=0.5, amount=2.0),),
    )
    baseline = replace(request)
    price_contract(request, algorithm="Backward")
    price_contract(request, algorithm="Recursive")
    assert request == baseline
    assert not hasattr(request, "algorithm")


# Every Greek shock must use the chosen pricing algorithm consistently.
@pytest.mark.parametrize("algorithm", PRICING_ALGORITHMS)
def test_seven_greeks_use_injected_pricing_algorithm(algorithm: str) -> None:
    request = PricingRequest(
        model="Trinomial · Allocation pruning", steps=14, volatility=0.22,
    )
    observed: list[str] = []

    def observed_pricer(shocked: PricingRequest) -> PricingResult:
        value = price_contract(shocked, algorithm=algorithm)
        observed.append(value.algorithm)
        return value

    snapshot = compute_desk_risk(request, pricer=observed_pricer)
    assert len(observed) == snapshot.base.evaluations + snapshot.extra_evaluations
    assert len(observed) > 8
    assert set(observed) == {algorithm}
    for name in ("Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Vomma"):
        assert snapshot.value(name) is not None


def test_greek_values_agree_across_algorithms() -> None:
    request = PricingRequest(
        model="Hybrid", option_type="Put", exercise="American",
        spot=104.0, strike=100.0, steps=20,
        dividends=(DividendInput(time=0.5, amount=2.0),),
    )
    recursive = compute_desk_risk(
        request, pricer=partial(price_contract, algorithm="Recursive"),
    )
    backward = compute_desk_risk(
        request, pricer=partial(price_contract, algorithm="Backward"),
    )
    for name in ("Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Vomma"):
        r_value, b_value = recursive.value(name), backward.value(name)
        if r_value is None or b_value is None:
            assert r_value is b_value
        else:
            assert math.isclose(r_value, b_value, rel_tol=1e-7, abs_tol=1e-7)


# The callback must propagate through selected-curve repricing.
def test_selected_greek_curve_uses_backward_callback() -> None:
    request = PricingRequest(model="Binomial", steps=9)
    seen = []

    def backward(shocked: PricingRequest) -> PricingResult:
        result = price_contract(shocked, algorithm="Backward")
        seen.append(result.algorithm)
        return result

    curve = compute_greek_curve(request, "Delta", count=3, pricer=backward)
    assert len(curve) == 3
    assert seen and set(seen) == {"Backward"}
    assert all(point.value is not None for point in curve)


def test_binomial_cash_dividend_restriction_remains() -> None:
    request = PricingRequest(
        model="Binomial", steps=8,
        dividends=(DividendInput(time=0.5, amount=3.0),),
    )
    with pytest.raises(PricingInputError, match="BinomialTree"):
        price_contract(request, algorithm="Backward")

# LOT_5C3_GUARDED_REFACTOR
