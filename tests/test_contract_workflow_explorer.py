"""Phase 1B.4: Lattice Explorer accepts only exact, deliberate contract transfers."""

from dataclasses import replace

import pytest

from frontend.contract_workflow import get_reference, save_reference
from frontend.lattice_explorer import (
    apply_reference_to_explorer,
    explorer_dividend_defaults,
)
from frontend.pricing_service import DividendInput, PricingRequest


def _valid(**overrides) -> PricingRequest:
    return replace(PricingRequest(steps=5), **overrides)


# No saved reference means no implicit mutation of lattice explorer state.
def test_absent_reference_never_changes_explorer() -> None:
    state = {"lattice_explorer_result": "old geometry", "lx_left_model": "Binomial"}
    before = state.copy()
    accepted, problems = apply_reference_to_explorer(state)
    assert not accepted and "no reference" in " ".join(problems).lower()
    assert state == before


def test_above_visual_cap_requires_explicit_change() -> None:
    state = {"lattice_explorer_result": "old geometry"}
    save_reference(state, PricingRequest(steps=101))
    before = state.copy()
    accepted, problems = apply_reference_to_explorer(state)
    assert not accepted
    assert any("N=101" in message for message in problems)
    assert state == before



def test_small_grid_import_is_exact_and_clears_obsolete_geometry() -> None:
    request = _valid(spot=105.0, strike=100.0, volatility=.25,
                     rate=.03, option_type="Put", exercise="American")
    state = {"lattice_explorer_result": "old geometry"}
    save_reference(state, request)
    accepted, problems = apply_reference_to_explorer(state)
    assert accepted and problems == ()
    assert state["explorer_reference_seed"] is request
    assert state["explorer_import_nonce"] == 1
    assert "lattice_explorer_result" not in state
    assert get_reference(state) is request


# Reimport rotates widget IDs so cached controls cannot override the new contract.
def test_reimport_switches_seed_and_generates_fresh_widget_ids() -> None:
    state = {}
    first = _valid(spot=95.0)
    second = _valid(spot=108.0, steps=8)
    save_reference(state, first)
    assert apply_reference_to_explorer(state)[0]
    save_reference(state, second)
    assert apply_reference_to_explorer(state)[0]
    assert state["explorer_import_nonce"] == 2
    assert state["explorer_reference_seed"] is second


# The explorer accepts only supported cash-dividend schedules without truncation.
def test_rejected_multiple_dividends_are_never_dropped() -> None:
    request = _valid(dividends=(DividendInput(.25, 2.0), DividendInput(.75, 3.0)))
    state = {"lattice_explorer_result": "old", "explorer_import_nonce": 7}
    save_reference(state, request)
    before = state.copy()
    ok, issues = apply_reference_to_explorer(state)
    assert not ok and any("at most one" in msg for msg in issues)
    assert state == before
    assert request.dividends == (DividendInput(.25, 2.0), DividendInput(.75, 3.0))


@pytest.mark.parametrize("threshold", [0.0, 1e-12, 1e-15])
def test_nonselectable_threshold_is_not_silently_replaced(threshold: float) -> None:
    state = {"lattice_explorer_result": "old"}
    save_reference(state, _valid(pruning_threshold=threshold))
    previous = state.copy()
    ok, issues = apply_reference_to_explorer(state)
    assert not ok and any("threshold" in msg.lower() for msg in issues)
    assert state == previous


# Successful imports retain the actual dividend time and amount.
def test_single_cash_dividend_is_transferred_exactly() -> None:
    request = _valid(dividends=(DividendInput(.4, 2.3),))
    state = {}
    save_reference(state, request)
    ok, issues = apply_reference_to_explorer(state)
    assert ok and not issues
    assert explorer_dividend_defaults(state["explorer_reference_seed"]) == (True, .4, 2.3)


def test_without_dividend_keeps_existing_optional_defaults() -> None:
    assert explorer_dividend_defaults(None) == (False, .5, 3.0)
    assert explorer_dividend_defaults(_valid()) == (False, .5, 3.0)


def test_model_comparison_choices_and_explorer_views_remain_independent() -> None:
    state = {
        "lx_left_model": "Binomial", "lx_right_model": "Hybrid",
        "lattice_v23_layout": "Focus right", "lattice_v23_geometry": "Connected only",
        "cmp_models": ("Trinomial · Full",),
    }
    original_choices = state.copy()
    save_reference(state, _valid(model="Trinomial · Branch pruning", steps=6))
    assert apply_reference_to_explorer(state)[0]
    assert all(state[key] == value for key, value in original_choices.items())


# Unsupported depth must fail visibly instead of silently changing N.
def test_low_depth_is_rejected_without_clamping() -> None:
    state = {}
    save_reference(state, _valid(steps=1))
    ok, issues = apply_reference_to_explorer(state)
    assert not ok and any("N=1" in issue for issue in issues)
    assert "explorer_reference_seed" not in state


def test_dividend_below_editor_minimum_rejected_without_rounding() -> None:
    state = {}
    save_reference(state, _valid(dividends=(DividendInput(.005, 2.0),)))
    ok, issues = apply_reference_to_explorer(state)
    assert not ok and any("0.01" in issue for issue in issues)
    assert "explorer_reference_seed" not in state


# Workspace navigation does not execute the financial engine.
def test_import_does_not_call_tree_or_pricer(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("An import cannot build, price or compare lattices.")
    monkeypatch.setattr("frontend.lattice_explorer_service.snapshot_priced_tree", forbidden)
    monkeypatch.setattr("frontend.pricing_service.price_contract", forbidden)
    state = {}
    save_reference(state, _valid(steps=5))
    assert apply_reference_to_explorer(state)[0]


def test_explorer_import_preserves_current_reference_after_clear_of_results() -> None:
    state = {"lattice_explorer_result": [("Binomial", None, "previous failure")]}
    request = _valid(steps=4)
    save_reference(state, request)
    assert apply_reference_to_explorer(state)[0]
    assert get_reference(state) is request
    assert "lattice_explorer_result" not in state

# LOT_5C3_GUARDED_REFACTOR
