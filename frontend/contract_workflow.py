"""Session-scoped contract reference, independent of Streamlit widgets.

The reference is a validated, immutable PricingRequest. Reading or checking it
never builds a lattice, runs a pricer, or changes the original contract.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from frontend.calendar_contract import ContractCalendar
from frontend.calendar_ticket import calendar_matches_request
from frontend.workspace_policy import (
    ANALYTICS_THRESHOLDS as _ANALYTICS_THRESHOLDS,
    COMPARISON_THRESHOLDS as _COMPARISON_THRESHOLDS,
    EXPLORER_THRESHOLDS as _EXPLORER_THRESHOLDS,
    MAX_BOUNDARY_STEPS, MAX_VISUAL_STEPS,
)
from frontend.pricing_service import PricingRequest, PricingResult, validate_request


REFERENCE_KEY = "lattice272_reference_contract"
REFERENCE_CALENDAR_KEY = "lattice272_reference_calendar"
TARGETS = (
    "model_comparison",
    "lattice_explorer",
    "quant_analytics",
    "exercise_boundary",
    "research_lab",
)

@dataclass(frozen=True, slots=True)
class TransferCheck:
    """Whether the complete request fits the target's current input controls."""

    target: str
    issues: tuple[str, ...]

    @property
    def exact(self) -> bool:
        """False means the UI must ask for an explicit adjustment."""
        return not self.issues


def save_reference(
    state: MutableMapping[str, Any], request: PricingRequest,
) -> None:
    """Save a validated contract without executing any financial model."""
    if not isinstance(request, PricingRequest):
        raise TypeError("Reference contract must be a PricingRequest.")
    validate_request(request)
    state[REFERENCE_KEY] = request
    state.pop(REFERENCE_CALENDAR_KEY, None)


def save_reference_from_last_result(
    state: MutableMapping[str, Any],
) -> PricingRequest:
    """Explicitly save the latest *successful* pricing ticket, without repricing.

    A subsequent valuation, failed attempt or ticket reset will never replace
    the reference until the user explicitly requests another save or clear.
    """
    result = state.get("last_pricing_result")
    if not isinstance(result, PricingResult):
        raise ValueError("No successful pricing result is available to save.")
    calendar = state.get("last_pricing_calendar")
    if calendar is not None:
        if not isinstance(calendar, ContractCalendar) or not calendar_matches_request(
            calendar, result.request
        ):
            raise ValueError("Last pricing calendar does not match the priced contract.")
    save_reference(state, result.request)
    if calendar is not None:
        state[REFERENCE_CALENDAR_KEY] = calendar
    return result.request


def get_reference(state: MutableMapping[str, Any]) -> PricingRequest | None:
    """Return the session reference, or None when none has been saved."""
    stored = state.get(REFERENCE_KEY)
    if stored is None:
        return None
    if not isinstance(stored, PricingRequest):
        raise TypeError("Stored reference contract is not a PricingRequest.")
    return stored


def get_reference_calendar(state: MutableMapping[str, Any]) -> ContractCalendar | None:
    """Retrieve original dates, when this reference originated from a dated ticket."""
    calendar = state.get(REFERENCE_CALENDAR_KEY)
    if calendar is not None and not isinstance(calendar, ContractCalendar):
        raise TypeError("Invalid reference calendar metadata.")
    return calendar


def clear_reference(state: MutableMapping[str, Any]) -> bool:
    """Discard both the reference ticket and its optional calendar metadata."""
    was_saved = state.pop(REFERENCE_KEY, None) is not None
    state.pop(REFERENCE_CALENDAR_KEY, None)
    return was_saved


def _collect_destination_limits(request: PricingRequest, target: str, issues: list[str]) -> None:
    """Report destination-specific grid, dividend and pruning constraints."""
    if target == "lattice_explorer":
        if not 2 <= request.steps <= MAX_VISUAL_STEPS:
            issues.append(
                f"Lattice Explorer requires N from 2 to {MAX_VISUAL_STEPS}; "
                f"the reference has N={request.steps}."
            )
        if len(request.dividends) > 1:
            issues.append(
                "Lattice Explorer supports at most one cash dividend; "
                f"the reference has {len(request.dividends)}."
            )
        if request.dividends and (
            request.dividends[0].time < 0.01
            or request.dividends[0].amount < 0.01
        ):
            issues.append(
                "Lattice Explorer dividend fields require time and amount >= 0.01."
            )
        if request.pruning_threshold not in _EXPLORER_THRESHOLDS:
            issues.append(
                "The reference pruning threshold is not offered in Lattice Explorer."
            )
    elif target == "exercise_boundary":
        if request.steps > MAX_BOUNDARY_STEPS:
            issues.append(
                f"Exercise Boundary allows at most N={MAX_BOUNDARY_STEPS}; "
                f"the reference has N={request.steps}."
            )
        if request.pruning_threshold not in _ANALYTICS_THRESHOLDS:
            issues.append(
                "The reference pruning threshold is not offered in Exercise Boundary."
            )
    elif target == "model_comparison":
        if request.pruning_threshold not in _COMPARISON_THRESHOLDS:
            issues.append(
                "The reference pruning threshold is not offered in Model Comparison."
            )
    elif target == "quant_analytics":
        if request.pruning_threshold not in _ANALYTICS_THRESHOLDS:
            issues.append(
                "The reference pruning threshold is not offered in Quant Analytics."
            )


def _collect_market_limits(request: PricingRequest, target: str, issues: list[str]) -> None:
    """Report shared form limits without modifying the pricing request."""
    if request.volatility > 5.0:
        issues.append("The target volatility input is limited to 500%.")
    if not -0.5 <= request.rate <= 1.0:
        issues.append("The target interest-rate input is limited to -50% through 100%.")
    if request.maturity < 0.01:
        issues.append("The target maturity input requires at least 0.01 year.")
    if request.spot < 0.01:
        issues.append("The target spot input requires at least EUR 0.01.")
    if target in ("model_comparison", "quant_analytics"):
        if request.spot > 1e7 or request.strike > 1e7:
            issues.append("The target spot and strike inputs are capped at EUR 10,000,000.")
        if request.maturity > 100.0:
            issues.append("The target maturity input is capped at 100 years.")


def assess_transfer(request: PricingRequest, target: str) -> TransferCheck:
    """Check *exact* transfer; never clamp N or silently drop dividends.

    This is a request-level preflight, not a promise that any numerical engine
    will successfully price the contract. Individual study-grid controls and
    model-specific failures are handled by the destination workspace.
    """
    if target not in TARGETS:
        raise ValueError(f"Unknown contract destination: {target}.")
    if not isinstance(request, PricingRequest):
        raise TypeError("Reference contract must be a PricingRequest.")
    validate_request(request)

    issues: list[str] = []
    if target == "research_lab":
        return TransferCheck(target, (
            "Research Lab reads archived experiments, not live pricing contracts.",
        ))

    _collect_destination_limits(request, target, issues)

    # These bounds are shared by the contract fields in the target UI forms.
    _collect_market_limits(request, target, issues)
    return TransferCheck(target, tuple(issues))

# LOT_5C5_STRUCTURAL_CLEANUP
