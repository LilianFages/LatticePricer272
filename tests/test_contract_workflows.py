"""Reference-contract transfers, comparison and lattice-explorer workflows.

Consolidated without dropping individual regression cases.
"""

from __future__ import annotations

from dataclasses import replace
import pytest
from frontend.contract_workflow import (
    REFERENCE_KEY,
    assess_transfer,
    clear_reference,
    get_reference,
    save_reference,
    save_reference_from_last_result,
)
from frontend.pricing_service import PricingInputError, PricingRequest, PricingResult
import pandas as pd
from frontend.contract_workflow import get_reference, save_reference
from frontend.model_comparison import (
    apply_reference_to_comparison,
    comparison_dividend_rows,
    parse_dividends,
)
from frontend.pricing_service import DividendInput, PricingRequest
from frontend.lattice_explorer import (
    apply_reference_to_explorer,
    explorer_dividend_defaults,
)


# test_contract_workflow.py (tests preserved)
# Reference lifecycle: explicit save/read/clear and immutable ownership.
def test_save_read_clear_reference() -> None:
    state: dict = {}
    request = PricingRequest(option_type="Put", exercise="American")
    assert get_reference(state) is None
    save_reference(state, request)
    assert get_reference(state) is request
    assert clear_reference(state) is True
    assert get_reference(state) is None
    assert clear_reference(state) is False


def test_saved_reference_does_not_follow_local_replacement() -> None:
    state: dict = {}
    original = PricingRequest(spot=100.0)
    save_reference(state, original)
    changed = replace(original, spot=120.0)
    assert changed.spot == 120.0
    assert get_reference(state).spot == 100.0


# Keep pricing comparisons on identical contracts and lattice settings.
def test_invalid_save_does_not_replace_existing_reference() -> None:
    state: dict = {}
    original = PricingRequest()
    save_reference(state, original)
    with pytest.raises(PricingInputError):
        save_reference(state, replace(original, spot=-10.0))
    assert get_reference(state) is original


def test_type_error_for_wrong_reference_object() -> None:
    with pytest.raises(TypeError):
        save_reference({}, object())
    with pytest.raises(TypeError):
        get_reference({REFERENCE_KEY: "invalid"})


def test_reference_save_does_not_run_pricer(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("Saving a reference must not price the contract")

    # Keep pricing comparisons on identical contracts and lattice settings.
    monkeypatch.setattr("frontend.pricing_service.price_contract", forbidden)
    request = PricingRequest()
    state: dict = {}
    save_reference(state, request)
    assert get_reference(state) is request


# Transfers must preserve contract values or report an explicit incompatibility.
@pytest.mark.parametrize("target", [
    "model_comparison", "quant_analytics", "exercise_boundary",
])
def test_default_contract_transfers_exactly(target: str) -> None:
    check = assess_transfer(PricingRequest(), target)
    assert check.exact
    assert not check.issues


def test_target_market_limits_are_explicit() -> None:
    request = PricingRequest(volatility=6.0, rate=1.1)
    check = assess_transfer(request, "model_comparison")
    assert not check.exact
    assert len(check.issues) == 2


def test_research_lab_is_offline_and_never_claims_import() -> None:
    check = assess_transfer(PricingRequest(), "research_lab")
    assert not check.exact
    # Verify the expected numerical or workflow invariant on this case.
    assert "archived experiments" in check.issues[0]


def test_unknown_target_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        assess_transfer(PricingRequest(), "unknown")


def test_invalid_contract_cannot_be_transferred() -> None:
    with pytest.raises(PricingInputError):
        assess_transfer(PricingRequest(spot=0.0), "model_comparison")


def test_large_maturity_not_exact_in_comparison_or_greeks() -> None:
    request = PricingRequest(maturity=120.0)
    for target in ("model_comparison", "quant_analytics"):
        check = assess_transfer(request, target)
        assert not check.exact
        assert any("maturity" in reason for reason in check.issues)


# Saved successful-ticket workflow (formerly test_contract_workflow_ui.py).
def _result(request: PricingRequest) -> PricingResult:
    return PricingResult(
        request=request, price=8.0, elapsed_ms=1.0,
        reachable_nodes=10, allocated_nodes=10,
        pruned_transitions=0, black_scholes=None,
    )


# Only an already completed pricing result may become the shared reference.
def test_successful_pricing_ticket_can_be_saved() -> None:
    request = PricingRequest(
        option_type="Put", exercise="American", steps=30,
    )
    state = {"last_pricing_result": _result(request)}
    assert save_reference_from_last_result(state) is request
    assert get_reference(state) is request
    assert state["last_pricing_result"].request is request


def test_no_successful_price_does_not_replace_reference() -> None:
    original = PricingRequest(steps=20)
    state = {"last_pricing_result": _result(original)}
    save_reference_from_last_result(state)
    state["last_pricing_result"] = None
    with pytest.raises(ValueError, match="successful pricing"):
        save_reference_from_last_result(state)
    assert get_reference(state) is original


# Keep pricing comparisons on identical contracts and lattice settings.
def test_invalid_result_type_cannot_be_saved() -> None:
    state = {"last_pricing_result": "fake-pricing-result"}
    with pytest.raises(ValueError, match="successful pricing"):
        save_reference_from_last_result(state)
    assert get_reference(state) is None


def test_new_pricing_result_does_not_implicitly_replace_reference() -> None:
    original = PricingRequest(spot=100.0)
    state = {"last_pricing_result": _result(original)}
    save_reference_from_last_result(state)
    later = replace(original, spot=125.0)
    state["last_pricing_result"] = _result(later)
    assert get_reference(state) is original
    assert get_reference(state).spot == 100.0
    save_reference_from_last_result(state)
    assert get_reference(state) is later


# Keep pricing comparisons on identical contracts and lattice settings.
def test_clear_reference_preserves_last_pricing_result() -> None:
    request = PricingRequest()
    result = _result(request)
    state = {"last_pricing_result": result}
    save_reference_from_last_result(state)
    assert clear_reference(state) is True
    assert get_reference(state) is None
    assert state["last_pricing_result"] is result


def test_saving_already_priced_result_never_calls_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    request = PricingRequest()
    state = {"last_pricing_result": _result(request)}

    def forbidden(*_args: object, **_kwargs: object):
        raise AssertionError("Reference save must not build or price a lattice")

    monkeypatch.setattr("frontend.pricing_service.price_contract", forbidden)
    assert save_reference_from_last_result(state) is request

# LOT_5C4_GUARDED_REFACTOR


# test_contract_workflow_comparison.py (tests preserved)
# The comparison workspace stays unchanged when no reference exists.
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


# Benchmark choices belong to the destination workspace, not the transferred ticket.
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
    # Keep pricing comparisons on identical contracts and lattice settings.
    state = {}
    save_reference(state, first)
    assert apply_reference_to_comparison(state)[0]
    assert state["cmp_import_nonce"] == 1
    second = replace(first, spot=107.5, steps=60)
    save_reference(state, second)
    assert apply_reference_to_comparison(state)[0]
    assert state["cmp_import_nonce"] == 2
    assert state["cmp_reference_seed"] is second


# Rejected handoffs preserve all previous state and numerical settings.
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
    # Keep pricing comparisons on identical contracts and lattice settings.
    ok, issues = apply_reference_to_comparison(state)
    assert not ok and any("maturity" in x for x in issues)
    assert "cmp_reference_seed" not in state
    assert bad.maturity == 110.0


# The comparison engine supports the complete multi-dividend schedule.
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


# Importing reference metadata must not initiate valuation.
def test_import_is_never_a_pricing_or_comparison_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object):
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

# LOT_5C3_GUARDED_REFACTOR


# test_contract_workflow_explorer.py (tests preserved)
def _valid(**overrides: object) -> PricingRequest:
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



# Keep pricing comparisons on identical contracts and lattice settings.
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
    # Keep pricing comparisons on identical contracts and lattice settings.
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
def test_import_does_not_call_tree_or_pricer(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object):
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
