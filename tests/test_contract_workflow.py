"""Regression tests for the Contract Workflow foundation (phase 1B.1)."""

from dataclasses import replace

import pytest

from frontend.contract_workflow import (
    REFERENCE_KEY,
    assess_transfer,
    clear_reference,
    get_reference,
    save_reference,
)
from frontend.pricing_service import DividendInput, PricingInputError, PricingRequest


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
    def forbidden(*args, **kwargs):
        raise AssertionError("Saving a reference must not price the contract")

    monkeypatch.setattr("frontend.pricing_service.price_contract", forbidden)
    request = PricingRequest()
    state: dict = {}
    save_reference(state, request)
    assert get_reference(state) is request


@pytest.mark.parametrize("target", [
    "model_comparison", "quant_analytics", "exercise_boundary",
])
def test_default_contract_transfers_exactly(target: str) -> None:
    check = assess_transfer(PricingRequest(), target)
    assert check.exact
    assert not check.issues


def test_small_grid_transfers_exactly_to_explorer() -> None:
    request = PricingRequest(steps=5)
    assert assess_transfer(request, "lattice_explorer").exact


def test_explorer_never_silently_reduces_steps() -> None:
    request = PricingRequest(steps=101)
    check = assess_transfer(request, "lattice_explorer")
    assert not check.exact
    assert "N=101" in check.issues[0]
    assert request.steps == 101



def test_explorer_rejects_multiple_dividends_without_dropping_them() -> None:
    dividends = (DividendInput(0.25, 2.0), DividendInput(0.75, 3.0))
    request = PricingRequest(steps=8, dividends=dividends)
    check = assess_transfer(request, "lattice_explorer")
    assert not check.exact
    assert any("at most one" in reason for reason in check.issues)
    assert request.dividends == dividends
    assert assess_transfer(request, "model_comparison").exact


def test_boundary_rejects_n_above_200_without_clamping() -> None:
    request = PricingRequest(steps=250)
    check = assess_transfer(request, "exercise_boundary")
    assert not check.exact
    assert any("N=200" in reason for reason in check.issues)
    assert request.steps == 250


def test_explorer_flags_unavailable_threshold() -> None:
    request = PricingRequest(steps=5, pruning_threshold=0.0)
    check = assess_transfer(request, "lattice_explorer")
    assert not check.exact
    assert any("threshold" in reason for reason in check.issues)


def test_target_market_limits_are_explicit() -> None:
    request = PricingRequest(volatility=6.0, rate=1.1)
    check = assess_transfer(request, "model_comparison")
    assert not check.exact
    assert len(check.issues) == 2


def test_research_lab_is_offline_and_never_claims_import() -> None:
    check = assess_transfer(PricingRequest(), "research_lab")
    assert not check.exact
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
