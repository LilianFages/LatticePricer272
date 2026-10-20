"""Phase 1B.3 — deliberate, exact reference import into Model Comparison."""

from dataclasses import replace

import pandas as pd
import pytest

from frontend.contract_workflow import get_reference, save_reference
from frontend.model_comparison import (
    apply_reference_to_comparison,
    comparison_dividend_rows,
    parse_dividends,
)
from frontend.pricing_service import DividendInput, PricingRequest


def test_no_reference_does_not_touch_existing_comparison_state() -> None:
    state = {"cmp_models": ("Binomial",), "last_comparison_report": "old"}
    before = state.copy()
    imported, issues = apply_reference_to_comparison(state)
    assert not imported
    assert "no reference" in " ".join(issues).lower()
    assert state == before


def test_default_reference_imports_exactly_without_pricing() -> None:
    req = PricingRequest()
    state = {"last_comparison_report": "old"}
    save_reference(state, req)
    imported, issues = apply_reference_to_comparison(state)
    assert imported and not issues
    assert state["cmp_reference_seed"] is req
    assert state["cmp_import_nonce"] == 1
    assert state["last_comparison_report"] is None
    assert get_reference(state) is req


def test_comparison_benchmark_settings_are_not_overwritten() -> None:
    req = PricingRequest(option_type="Put", exercise="American", steps=75)
    state = {
        "cmp_models": ("Binomial", "Hybrid"),
        "cmp_repetitions": 5,
        "cmp_reference": "Black–Scholes",
        "last_comparison_report": object(),
    }
    save_reference(state, req)
    imported, issues = apply_reference_to_comparison(state)
    assert imported and issues == ()
    assert state["cmp_models"] == ("Binomial", "Hybrid")
    assert state["cmp_repetitions"] == 5
    assert state["cmp_reference"] == "Black–Scholes"
    assert state["cmp_reference_seed"] is req


def test_repeated_explicit_import_increments_widget_generation() -> None:
    first = PricingRequest(spot=93.25)
    state = {}
    save_reference(state, first)
    assert apply_reference_to_comparison(state)[0]
    assert state["cmp_import_nonce"] == 1
    second = replace(first, spot=107.5, steps=60)
    save_reference(state, second)
    assert apply_reference_to_comparison(state)[0]
    assert state["cmp_import_nonce"] == 2
    assert state["cmp_reference_seed"] is second


def test_incompatible_threshold_rejected_atomically() -> None:
    state = {
        "cmp_import_nonce": 3,
        "cmp_reference_seed": PricingRequest(spot=81.0),
        "last_comparison_report": "do-not-discard",
    }
    bad = PricingRequest(pruning_threshold=1e-5)
    save_reference(state, bad)
    before = state.copy()
    imported, issues = apply_reference_to_comparison(state)
    assert not imported
    assert any("threshold" in msg.lower() for msg in issues)
    assert state == before


def test_out_of_bounds_contract_rejected_without_truncation() -> None:
    state = {}
    bad = PricingRequest(maturity=110.0)
    save_reference(state, bad)
    ok, issues = apply_reference_to_comparison(state)
    assert not ok and any("maturity" in x for x in issues)
    assert "cmp_reference_seed" not in state
    assert bad.maturity == 110.0


def test_two_cash_dividends_are_transferred_without_loss() -> None:
    dividends = (DividendInput(0.25, 2.0), DividendInput(0.75, 3.0))
    req = PricingRequest(steps=100, dividends=dividends)
    state = {}
    save_reference(state, req)
    imported, issues = apply_reference_to_comparison(state)
    assert imported and not issues
    frame = pd.DataFrame(comparison_dividend_rows(state["cmp_reference_seed"]))
    assert parse_dividends(frame) == dividends


def test_no_dividends_keeps_existing_empty_row_semantics() -> None:
    rows = comparison_dividend_rows(PricingRequest())
    assert rows == [{"time": 0.5, "amount": 0.0}]
    assert parse_dividends(pd.DataFrame(rows)) == ()


def test_import_is_never_a_pricing_or_comparison_trigger(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Import must not price contracts")

    monkeypatch.setattr("frontend.comparison_service.compare_models", forbidden)
    monkeypatch.setattr("frontend.pricing_service.price_contract", forbidden)
    state = {}
    save_reference(state, PricingRequest(steps=40))
    assert apply_reference_to_comparison(state)[0]


def test_reference_engine_does_not_overwrite_comparison_model_selection() -> None:
    req = PricingRequest(model="Trinomial · Branch pruning", steps=40)
    state = {"cmp_models": ("Binomial",)}
    save_reference(state, req)
    assert apply_reference_to_comparison(state)[0]
    assert state["cmp_models"] == ("Binomial",)
    assert state["cmp_reference_seed"].model == req.model
