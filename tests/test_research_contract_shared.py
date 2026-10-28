"""Lot 5B.2 — shared research contract fields must preserve financial semantics."""
from __future__ import annotations

import pandas as pd
import pytest

from frontend import analytics_views, research_efficiency_views
from frontend.pricing_service import PricingInputError
from frontend.research_contract import (
    build_research_request,
    render_research_contract_fields,
)


def _fields(dividends=None):
    return {
        "spot": 120.0, "strike": 110.0, "volatility": 0.32,
        "rate": -0.01, "maturity": 1.5,
        "option_type": "Put", "exercise": "American",
        "dividends_frame": pd.DataFrame(dividends or [], columns=["time", "amount"]),
    }


def test_both_workspaces_reuse_identical_contract_functions():
    assert analytics_views._contract_fields is render_research_contract_fields
    assert research_efficiency_views._contract_fields is render_research_contract_fields
    assert analytics_views._get_request is build_research_request
    assert research_efficiency_views._get_request is build_research_request


def test_pricing_request_preserves_units_market_and_lattice_settings():
    request = build_research_request(
        _fields(), "Trinomial · Allocation pruning", 125, 1e-12,
    )
    assert (request.spot, request.strike, request.volatility) == (120.0, 110.0, 0.32)
    assert (request.rate, request.maturity) == (-0.01, 1.5)
    assert (request.option_type, request.exercise) == ("Put", "American")
    assert (request.model, request.steps, request.pruning_threshold) == (
        "Trinomial · Allocation pruning", 125, 1e-12,
    )
    assert request.dividends == ()


def test_discrete_cash_dividends_are_sorted_and_zero_ignored():
    request = build_research_request(
        _fields([{"time": 1.0, "amount": 2.0},
                 {"time": 0.5, "amount": 1.5},
                 {"time": 0.8, "amount": 0.0}]),
        "Trinomial · Allocation pruning", 100, 1e-18,
    )
    assert [(p.time, p.amount) for p in request.dividends] == [(0.5, 1.5), (1.0, 2.0)]


def test_incomplete_dividend_is_not_silently_dropped():
    fields = _fields([{"time": 0.5, "amount": None}])
    with pytest.raises(PricingInputError, match="Dividend row 1"):
        build_research_request(fields, "Trinomial · Full", 80, 0.0)


def test_research_contract_does_not_depend_on_analytics_view():
    from inspect import getsource
    import frontend.research_contract as shared
    code = getsource(shared)
    assert "from frontend.analytics_views" not in code
    assert "from frontend.model_comparison" not in code
    assert "from frontend.dividend_input import parse_dividends" in code
