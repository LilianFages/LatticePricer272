"""Integration checks for the Streamlit adapter, using the REAL src/ engines.

Run from repository root:
    python -m pytest tests/test_frontend_pricing.py -q
"""

import csv
import io
import math

import pytest

from frontend.pricing_service import (
    MODELS,
    DividendInput,
    PricingInputError,
    PricingRequest,
    count_reachable_nodes,
    price_contract,
    ticket_csv,
    validate_request,
)
from src.node import Node


# Check nonnegative prices, populated references and coherent lattice sizes.
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("kind", ("Call", "Put"))
def test_european_models_return_finite_price_and_valid_node_count(
    model: str, kind: str
) -> None:
    result = price_contract(PricingRequest(model=model, option_type=kind, steps=20))
    assert math.isfinite(result.price)
    assert result.price >= 0.0
    assert result.reachable_nodes >= result.request.steps + 1
    assert result.elapsed_ms >= 0.0
    assert result.black_scholes is not None
    if model == "Trinomial · Allocation pruning":
        assert result.allocated_nodes is not None
        assert result.allocated_nodes >= result.reachable_nodes
    else:
        assert result.allocated_nodes is None


# Immediate exercise gives a rigorous lower bound for an American put.
@pytest.mark.parametrize("model", MODELS)
def test_american_put_not_below_intrinsic_value(model: str) -> None:
    result = price_contract(PricingRequest(
        model=model, steps=25, exercise="American",
        option_type="Put", spot=90.0, strike=100.0,
    ))
    assert result.price >= 10.0 - 1e-10
    assert result.black_scholes is None


# Cash dividends invalidate the no-dividend analytic benchmark.
@pytest.mark.parametrize("model", (
    "Trinomial · Full",
    "Trinomial · Branch pruning",
    "Trinomial · Allocation pruning", "Hybrid",
))
def test_supported_models_price_with_cash_dividend(model: str) -> None:
    result = price_contract(PricingRequest(
        model=model, steps=20,
        dividends=(DividendInput(time=0.5, amount=3.0),),
    ))
    assert math.isfinite(result.price)
    assert result.black_scholes is None


# At zero pruning, the allocation and full-grid prices must coincide.
def test_variants_match_at_zero_pruning_threshold() -> None:
    common = dict(steps=25, option_type="Put", pruning_threshold=0.0)
    full = price_contract(PricingRequest(**common, model="Trinomial · Full"))
    allocation = price_contract(PricingRequest(
        **common, model="Trinomial · Allocation pruning",
    ))
    assert allocation.price == pytest.approx(full.price, abs=1e-10)


# The allocation engine must handle the two-ex-date contract.
def test_two_dividends_can_be_priced_by_allocation_tree() -> None:
    result = price_contract(PricingRequest(
        model="Trinomial · Allocation pruning", steps=80,
        dividends=(DividendInput(0.25, 2.0), DividendInput(0.75, 3.0)),
    ))
    assert result.price > 0.0


# Disallow cash-dividend pricing when the engine does not support it.
def test_binomial_dividends_are_rejected_with_clear_message() -> None:
    with pytest.raises(PricingInputError, match="does not support cash dividends"):
        validate_request(PricingRequest(
            model="Binomial", dividends=(DividendInput(0.5, 3.0),),
        ))


# Enforce the engine convention of one cash dividend per time step.
def test_dividends_cannot_share_lattice_step() -> None:
    with pytest.raises(PricingInputError, match="same lattice time step"):
        validate_request(PricingRequest(
            model="Hybrid", steps=10,
            dividends=(DividendInput(0.51, 1.0), DividendInput(0.55, 1.0)),
        ))


# Inputs must be rejected before the numerical engine is invoked.
@pytest.mark.parametrize("override", [
    {"spot": 0.0}, {"volatility": -0.1}, {"maturity": 0.0},
    {"steps": 0}, {"steps": 351}, {"steps": 12.5},
    {"strike": -1.0}, {"rate": math.nan},
    {"pruning_threshold": 1.0}, {"exercise": "Bermudan"},
])
def test_invalid_contract_parameters_are_rejected(override: dict) -> None:
    with pytest.raises(PricingInputError):
        validate_request(PricingRequest(**override))


# A cash payment cannot destroy positivity of the adjusted stock process.
def test_dividend_greater_than_spot_is_rejected() -> None:
    with pytest.raises(PricingInputError, match="Discounted cash dividends"):
        validate_request(PricingRequest(
            model="Hybrid", dividends=(DividendInput(0.5, 200.0),),
        ))


# Reachable-node counts must not double-count shared successors.
def test_reachable_count_deduplicates_recombining_links() -> None:
    root = Node(100.0)
    a = Node(110.0)
    b = Node(90.0)
    shared = Node(100.0)
    root.next_up = a
    root.next_down = b
    a.next_down = shared
    b.next_up = shared
    assert count_reachable_nodes(root) == 4


# A downloadable ticket must retain the full contract specification.
def test_ticket_export_contains_market_parameters_and_price() -> None:
    result = price_contract(PricingRequest(steps=15))
    fields = dict(csv.reader(io.StringIO(ticket_csv(result))))
    assert fields["model"] == "Trinomial · Allocation pruning"
    assert fields["steps"] == "15"
    assert float(fields["price"]) == pytest.approx(result.price)
    assert float(fields["reachable_nodes"]) >= 16


# Displayed price gaps must preserve the intended benchmark sign.
def test_black_scholes_difference_sign_is_correct() -> None:
    result = price_contract(PricingRequest(steps=30, model="Binomial"))
    assert result.black_scholes_error == pytest.approx(
        result.price - result.black_scholes
    )
