"""Regression tests for the Contract Workflow foundation (phase 1B.1)."""

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


# Saved successful-ticket workflow (formerly test_contract_workflow_ui.py).
def _result(request: PricingRequest) -> PricingResult:
    return PricingResult(
        request=request, price=8.0, elapsed_ms=1.0,
        reachable_nodes=10, allocated_nodes=10,
        pruned_transitions=0, black_scholes=None,
    )


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


def test_clear_reference_preserves_last_pricing_result() -> None:
    request = PricingRequest()
    result = _result(request)
    state = {"last_pricing_result": result}
    save_reference_from_last_result(state)
    assert clear_reference(state) is True
    assert get_reference(state) is None
    assert state["last_pricing_result"] is result


def test_saving_already_priced_result_never_calls_engine(monkeypatch) -> None:
    request = PricingRequest()
    state = {"last_pricing_result": _result(request)}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Reference save must not build or price a lattice")

    monkeypatch.setattr("frontend.pricing_service.price_contract", forbidden)
    assert save_reference_from_last_result(state) is request
