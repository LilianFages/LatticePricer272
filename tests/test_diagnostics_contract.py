"""Lot 2B1: transferring the market contract across diagnostic investigations."""
from dataclasses import replace

import pytest

from frontend.diagnostics_contract import (
    THRESHOLDS, apply_handoff, plan_handoff,
)
from frontend.lattice_explorer_service import MAX_VISUAL_STEPS
from frontend.pricing_service import DividendInput, PricingRequest


def test_exact_comparison_to_explorer_handoff() -> None:
    req = PricingRequest(steps=5, option_type="Put", exercise="American")
    planned, issues = plan_handoff(req, "Explore lattice")
    assert planned is not req
    assert planned == req
    assert issues == ()


def test_large_depth_never_silently_clamped() -> None:
    request = PricingRequest(steps=101)
    planned, issues = plan_handoff(request, "Explore lattice")
    assert planned.steps == 101
    assert issues and any("N=101" in message for message in issues)
    adjusted, issues = plan_handoff(request, "Explore lattice", steps=20)
    assert not issues
    assert adjusted.steps == 20
    assert request.steps == 101



def test_threshold_is_not_silently_changed() -> None:
    request = PricingRequest(steps=5, pruning_threshold=1e-3)
    planned, issues = plan_handoff(request, "Compare models")
    assert planned.pruning_threshold == 1e-3
    assert issues
    adjusted, issues = plan_handoff(request, "Compare models", threshold=1e-18)
    assert issues == ()
    assert adjusted.pruning_threshold in THRESHOLDS["Compare models"]
    assert request.pruning_threshold == 1e-3


def test_never_silently_discard_dividends() -> None:
    divs = (DividendInput(.25, 1.), DividendInput(.75, 2.))
    request = PricingRequest(steps=5, dividends=divs)
    planned, issues = plan_handoff(request, "Explore lattice")
    assert issues and any("at most one" in s for s in issues)
    assert planned.dividends == divs


def test_atomic_rejection_preserves_state() -> None:
    original = PricingRequest(steps=5)
    state = {"explorer_reference_seed": original, "explorer_import_nonce": 2}
    previous = state.copy()
    with pytest.raises(ValueError, match="Incompatible"):
        apply_handoff(state, "Explore lattice", replace(original, steps=MAX_VISUAL_STEPS + 1))
    assert state == previous


def test_apply_explorer_rotates_widget_generation_and_forgets_stale_graph() -> None:
    req = PricingRequest(steps=5)
    state = {"lattice_explorer_result": "old", "explorer_import_nonce": 1}
    apply_handoff(state, "Explore lattice", req)
    assert state["explorer_import_nonce"] == 2
    assert state["explorer_reference_seed"] == req
    assert "lattice_explorer_result" not in state


def test_apply_comparison_keeps_benchmark_choices_and_clears_stale_result() -> None:
    req = PricingRequest(steps=5)
    state = {"cmp_models": ("Hybrid",), "cmp_repetitions": 3,
             "last_comparison_report": "old"}
    apply_handoff(state, "Compare models", req)
    assert state["cmp_import_nonce"] == 1
    assert state["cmp_reference_seed"] == req
    assert state["last_comparison_report"] is None
    assert state["cmp_models"] == ("Hybrid",)
    assert state["cmp_repetitions"] == 3


def test_invalid_nonce_rejected_without_mutation() -> None:
    state = {"cmp_import_nonce": "wrong"}
    before = state.copy()
    with pytest.raises(ValueError, match="generation"):
        apply_handoff(state, "Compare models", PricingRequest())
    assert state == before


def test_transfer_never_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected pricing")
    monkeypatch.setattr("frontend.pricing_service.price_contract", forbidden)
    req = PricingRequest(steps=5)
    adjusted, issues = plan_handoff(req, "Explore lattice")
    assert not issues
    apply_handoff({}, "Explore lattice", adjusted)


def test_unknown_destination_rejected() -> None:
    with pytest.raises(ValueError):
        plan_handoff(PricingRequest(), "Unknown")
