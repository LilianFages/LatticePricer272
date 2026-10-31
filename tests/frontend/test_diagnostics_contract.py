"""Lot 2B1: transferring the market contract across diagnostic investigations."""
from dataclasses import replace

import pytest

from frontend.contract_workflow import get_reference, save_reference

from frontend.diagnostics_contract import (
    THRESHOLDS, apply_handoff, import_saved_reference, plan_handoff,
)
from frontend.lattice_explorer_service import MAX_VISUAL_STEPS
from frontend.pricing_service import DividendInput, PricingRequest


def _sample_request(*, steps: int = 5, **changes: object) -> PricingRequest:
    """Use the same small, admissible lattice contract across handoff tests."""
    return PricingRequest(steps=steps, **changes)


def _saved_reference_state(request: PricingRequest) -> dict[str, object]:
    """Prepare a saved contract without changing destination workspace state."""
    state: dict[str, object] = {}
    save_reference(state, request)
    return state


# Handoff must preserve the full financial contract, not merely visible widgets.
def test_exact_comparison_to_explorer_handoff() -> None:
    req = _sample_request(option_type="Put", exercise="American")
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



# Incompatible display thresholds require explicit user choices.
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


# Rejected imports must be atomic and keep all previous cached state.
def test_atomic_rejection_preserves_state() -> None:
    original = _sample_request()
    state = {"explorer_reference_seed": original, "explorer_import_nonce": 2}
    previous = state.copy()
    with pytest.raises(ValueError, match="Incompatible"):
        apply_handoff(state, "Explore lattice", replace(original, steps=MAX_VISUAL_STEPS + 1))
    assert state == previous


def test_apply_explorer_rotates_widget_generation_and_forgets_stale_graph() -> None:
    req = _sample_request()
    state = {"lattice_explorer_result": "old", "explorer_import_nonce": 1}
    apply_handoff(state, "Explore lattice", req)
    assert state["explorer_import_nonce"] == 2
    assert state["explorer_reference_seed"] == req
    assert "lattice_explorer_result" not in state


def test_apply_comparison_keeps_benchmark_choices_and_clears_stale_result() -> None:
    # Verify that the pricing workflow keeps its input contract intact.
    req = _sample_request()
    state = {"cmp_models": ("Hybrid",), "cmp_repetitions": 3,
             "last_comparison_report": "old"}
    apply_handoff(state, "Compare models", req)
    assert state["cmp_import_nonce"] == 1
    assert state["cmp_reference_seed"] == req
    assert state["last_comparison_report"] is None
    assert state["cmp_models"] == ("Hybrid",)
    assert state["cmp_repetitions"] == 3


# Widget generations prevent stale Streamlit keys after a transfer.
def test_invalid_nonce_rejected_without_mutation() -> None:
    state = {"cmp_import_nonce": "wrong"}
    before = state.copy()
    with pytest.raises(ValueError, match="generation"):
        apply_handoff(state, "Compare models", PricingRequest())
    assert state == before


def test_transfer_never_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unexpected pricing")
    monkeypatch.setattr("frontend.pricing_service.price_contract", forbidden)
    req = _sample_request()
    adjusted, issues = plan_handoff(req, "Explore lattice")
    assert not issues
    apply_handoff({}, "Explore lattice", adjusted)


def test_unknown_destination_rejected() -> None:
    # Reject invalid cases explicitly rather than silently substituting values.
    with pytest.raises(ValueError):
        plan_handoff(PricingRequest(), "Unknown")


# Imported references preserve identity while invalidating destination-only output.
@pytest.mark.parametrize("destination,seed,nonce,stale", [
    ("Compare models", "cmp_reference_seed", "cmp_import_nonce", "last_comparison_report"),
    ("Explore lattice", "explorer_reference_seed", "explorer_import_nonce", "lattice_explorer_result"),
])
def test_import_exact_identity_and_clear_stale_output(destination: str, seed: str, nonce: str, stale: str) -> None:
    request = _sample_request(option_type="Put", exercise="American")
    state = {stale: "old output", "cmp_models": ("Binomial",),
             "lx_left_model": "Hybrid"}
    save_reference(state, request)
    assert import_saved_reference(state, destination) == (True, ())
    assert state[seed] is request
    assert state[nonce] == 1
    assert stale not in state or state[stale] is None
    assert get_reference(state) is request
    assert state["cmp_models"] == ("Binomial",)
    assert state["lx_left_model"] == "Hybrid"


@pytest.mark.parametrize("destination", ["Compare models", "Explore lattice"])
# Verify that the pricing workflow keeps its input contract intact.
def test_absent_reference_does_not_mutate_state(destination: str) -> None:
    state = {"last_comparison_report": object(), "lattice_explorer_result": object()}
    before = state.copy()
    assert import_saved_reference(state, destination)[0] is False
    assert state == before


# Repeated imports must not disturb unrelated workspace generations.
@pytest.mark.parametrize("destination", ["Compare models", "Explore lattice"])
def test_repeated_import_rotates_only_destination_generation(destination: str) -> None:
    request = _sample_request()
    state = _saved_reference_state(request)
    assert import_saved_reference(state, destination)[0]
    save_reference(state, replace(request, spot=110.0))
    assert import_saved_reference(state, destination)[0]
    key = "cmp_import_nonce" if destination == "Compare models" else "explorer_import_nonce"
    assert state[key] == 2


def test_explorer_cannot_silently_drop_second_dividend() -> None:
    original = _sample_request()
    state = {"lattice_explorer_result": "old"}
    save_reference(state, original)
    bad = replace(original, dividends=(DividendInput(.2, 1.), DividendInput(.7, 2.)))
    save_reference(state, bad)
    before = state.copy()
    # Verify that the pricing workflow keeps its input contract intact.
    accepted, issues = import_saved_reference(state, "Explore lattice")
    assert not accepted and any("at most one" in message for message in issues)
    assert state == before


# Bad generations cannot leave partially imported contracts behind.
@pytest.mark.parametrize("destination,nonce", [
    ("Compare models", "cmp_import_nonce"),
    ("Explore lattice", "explorer_import_nonce"),
])
def test_invalid_nonce_does_not_partially_import(destination: str, nonce: str) -> None:
    state = {nonce: "invalid"}
    save_reference(state, _sample_request())
    before = state.copy()
    with pytest.raises(ValueError, match="generation"):
        import_saved_reference(state, destination)
    assert state == before


def test_import_refuses_unknown_destination_without_mutation() -> None:
    state = {}
    save_reference(state, _sample_request())
    before = state.copy()
    with pytest.raises(ValueError, match="Unknown"):
        import_saved_reference(state, "Unknown")
    # Verify the expected numerical or workflow invariant on this case.
    assert state == before

# LOT_5C3_GUARDED_REFACTOR
