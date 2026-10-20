"""Phase 1B.2: saving a successful pricing ticket is opt-in and side-effect free."""

from dataclasses import replace

import pytest

from frontend.contract_workflow import (
    get_reference,
    save_reference_from_last_result,
    clear_reference,
)
from frontend.pricing_service import PricingRequest, PricingResult


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
