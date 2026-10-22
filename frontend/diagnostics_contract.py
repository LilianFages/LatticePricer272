"""Safe contract handoff between Model Diagnostics investigations.

No implicit pricing, N clamping, threshold substitution or cashflow deletion.
All transformations are explicit and all state changes are atomic.
"""
from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import replace
from typing import Any

from frontend.contract_workflow import assess_transfer
from frontend.pricing_service import PricingRequest
from frontend.workspace_policy import (COMPARISON_THRESHOLDS, EXPLORER_THRESHOLDS)


VIEW_TO_TARGET = {
    "Compare models": "model_comparison",
    "Explore lattice": "lattice_explorer",
}
THRESHOLDS = {
    "Compare models": COMPARISON_THRESHOLDS,
    "Explore lattice": EXPLORER_THRESHOLDS,
}


def plan_handoff(
    request: PricingRequest,
    destination: str,
    *,
    steps: int | None = None,
    threshold: float | None = None,
) -> tuple[PricingRequest, tuple[str, ...]]:
    """Compute the *explicitly proposed* target ticket without modifying state."""
    if destination not in VIEW_TO_TARGET:
        raise ValueError(f"Unknown Model Diagnostics view: {destination}.")
    if not isinstance(request, PricingRequest):
        raise TypeError("Diagnostics handoff requires a PricingRequest.")
    adjusted = replace(
        request,
        steps=request.steps if steps is None else steps,
        pruning_threshold=(request.pruning_threshold if threshold is None else threshold),
    )
    return adjusted, assess_transfer(adjusted, VIEW_TO_TARGET[destination]).issues


def apply_handoff(
    state: MutableMapping[str, Any], destination: str, request: PricingRequest,
) -> None:
    """Prefill only compatible widgets; no numerical calculation or shared-reference edits."""
    adjusted, issues = plan_handoff(request, destination)
    if issues:
        raise ValueError("Incompatible target ticket: " + " ".join(issues))
    if destination == "Compare models":
        generation_key = "cmp_import_nonce"
        seed_key = "cmp_reference_seed"
    else:
        generation_key = "explorer_import_nonce"
        seed_key = "explorer_reference_seed"
    generation = state.get(generation_key, 0)
    if type(generation) is not int or generation < 0:
        raise ValueError("Invalid import widget generation.")
    # Every validation precedes mutation. Old numerical results become stale.
    state[seed_key] = adjusted
    state[generation_key] = generation + 1
    if destination == "Compare models":
        state["last_comparison_report"] = None
    else:
        state.pop("lattice_explorer_result", None)
