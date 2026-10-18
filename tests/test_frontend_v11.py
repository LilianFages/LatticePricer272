"""New v1.1 functionality: real model revaluation and honest node baselines."""

import csv
import io
import math

import pytest

from frontend.pricing_service import (
    DividendInput,
    PricingInputError,
    PricingRequest,
    compute_spot_sensitivity,
    price_contract,
    ticket_csv,
)


def test_relative_price_gap_against_black_scholes() -> None:
    result = price_contract(PricingRequest(steps=40, model="Binomial"))
    assert result.black_scholes is not None
    assert result.black_scholes_relative_error_pct == pytest.approx(
        100.0 * (result.price - result.black_scholes) / result.black_scholes
    )


def test_no_dividend_trinomial_allocation_reference() -> None:
    result = price_contract(PricingRequest(steps=50))
    assert result.full_trinomial_reference_nodes == (50 + 1) ** 2
    assert result.allocated_nodes is not None
    assert result.allocated_nodes <= result.full_trinomial_reference_nodes
    assert result.allocation_savings_pct == pytest.approx(
        100 * (1 - result.allocated_nodes / result.full_trinomial_reference_nodes)
    )


def test_cash_dividend_has_no_unverified_full_tree_node_baseline() -> None:
    result = price_contract(PricingRequest(
        steps=25,
        dividends=(DividendInput(time=0.5, amount=3.0),),
    ))
    assert result.full_trinomial_reference_nodes is None
    assert result.allocation_savings_pct is None


def test_binomial_does_not_claim_trinomial_allocation_reference() -> None:
    result = price_contract(PricingRequest(model="Binomial", steps=25))
    assert result.full_trinomial_reference_nodes is None
    assert result.allocation_savings_pct is None


def test_spot_revaluation_matches_direct_engine_run() -> None:
    req = PricingRequest(steps=20, model="Binomial", option_type="Call")
    points = compute_spot_sensitivity(req, points=5, span=0.20)
    assert [round(p.spot, 10) for p in points] == [80, 90, 100, 110, 120]
    assert all(point.error is None for point in points)
    assert points[2].price == pytest.approx(price_contract(req).price)
    assert all(points[i].price <= points[i+1].price + 1e-10 for i in range(4))


def test_spot_sensitivity_uses_existing_american_logic() -> None:
    req = PricingRequest(
        steps=20, model="Trinomial · Allocation pruning",
        option_type="Put", exercise="American",
    )
    points = compute_spot_sensitivity(req, points=3, span=0.10)
    assert all(point.error is None and point.price >= 0.0 for point in points)
    assert points[0].price >= points[1].price >= points[2].price


@pytest.mark.parametrize("points, span", [(2, .2), (22, .2), (9, 0.), (9, 1.), (9, float('nan'))])
def test_spot_sensitivity_rejects_invalid_grid(points: int, span: float) -> None:
    with pytest.raises(PricingInputError):
        compute_spot_sensitivity(PricingRequest(steps=10), points=points, span=span)


def test_sensitivity_does_not_mutate_original_request() -> None:
    request = PricingRequest(spot=100, steps=15)
    compute_spot_sensitivity(request, points=3)
    assert request.spot == 100.0


def test_new_metrics_are_in_exported_ticket() -> None:
    result = price_contract(PricingRequest(steps=10))
    values = dict(csv.reader(io.StringIO(ticket_csv(result))))
    assert math.isclose(float(values["allocation_savings_pct"]), result.allocation_savings_pct)
    assert values["full_trinomial_reference_nodes"] == "121"
