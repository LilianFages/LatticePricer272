"""Shared interactive workspace for cross-model and lattice investigations.

The two existing investigation forms intentionally remain separate in Lot 2A;
this navigation layer does not silently change model parameters or rerun pricing.
"""
from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import streamlit as st

from frontend.workspace_policy import MAX_VISUAL_STEPS


WORKSPACES = (
    "01 / Pricing terminal",
    "02 / Model diagnostics",
    "03 / Quant research",
)
DIAGNOSTIC_MODES = ("Compare models", "Explore lattice")


def migrate_navigation_state(state: MutableMapping[str, Any]) -> None:
    """Preserve the selected analysis during the legacy-to-three page migration.

    Called before the Streamlit navigation widget is instantiated. The
    operation is idempotent and never launches a pricing calculation.
    """
    migrations = {
        "02 / Model comparison": ("02 / Model diagnostics", "Compare models"),
        "03 / Lattice explorer": ("02 / Model diagnostics", "Explore lattice"),
        "04 / Quant analytics": ("03 / Quant research", None),
        "05 / Research lab": ("03 / Quant research", None),
        "03 / Quant analytics": ("03 / Quant research", None),
        "04 / Research lab": ("03 / Quant research", None),
    }
    old = state.get("lattice_workspace")
    if old not in migrations:
        return
    destination, mode = migrations[old]
    state["lattice_workspace"] = destination
    if mode is not None:
        state["model_diagnostics_view"] = mode




def _handoff_visual_steps(request: Any, mode: str) -> int:
    """Ask for an explicit N adjustment only outside the visual depth range."""
    proposed_steps = request.steps
    if mode == "Explore lattice" and not 2 <= request.steps <= MAX_VISUAL_STEPS:
        st.info(f"Lattice inspection requires 2 ≤ N ≤ {MAX_VISUAL_STEPS}. "
                "Choose a new visual depth explicitly.")
        proposed_steps = st.selectbox(
            "New visualization depth (N)", range(2, MAX_VISUAL_STEPS + 1), index=3,
            key="diagnostics_transfer_visual_steps",
        )
    return proposed_steps


def _handoff_threshold(request: Any, mode: str, available: tuple[float, ...]) -> float:
    """Offer only allowed pruning thresholds; preserve the original widget key."""
    proposed_threshold = request.pruning_threshold
    if proposed_threshold not in available:
        st.info("The original pruning threshold is unavailable in this investigation.")
        proposed_threshold = st.selectbox(
            "New pruning threshold", available,
            index=available.index(1e-18),
            format_func=lambda value: "0 (off)" if value == 0.0 else f"{value:.0e}",
            key=f"diagnostics_transfer_threshold_{mode.replace(' ', '_')}",
        )
    return proposed_threshold


def _render_shared_ticket_handoff(mode: str) -> None:
    """Opt-in transfer of the last *submitted* contract across investigations."""
    from frontend.diagnostics_contract import THRESHOLDS, apply_handoff, plan_handoff
    from frontend.pricing_service import PricingRequest

    request = st.session_state.get("diagnostics_last_contract")
    origin = st.session_state.get("diagnostics_last_source")
    if not isinstance(request, PricingRequest) or origin == mode:
        return

    with st.expander("Reuse contract from previous investigation", expanded=False):
        st.caption(
            f"Previous: {origin} · {request.exercise} {request.option_type} · "
            f"S={request.spot:g} € · K={request.strike:g} € · N={request.steps} · "
            f"{len(request.dividends)} dividend(s). No pricing runs until submission."
        )
        proposed_steps = _handoff_visual_steps(request, mode)

        available = THRESHOLDS[mode]
        proposed_threshold = _handoff_threshold(
            request, mode, available,
        )

        planned, issues = plan_handoff(
            request, mode, steps=proposed_steps, threshold=proposed_threshold,
        )
        if issues:
            for message in issues:
                st.warning(message)
        if st.button(
            "Prefill current investigation", key="diagnostics_transfer_apply",
            disabled=bool(issues), width="stretch",
        ):
            apply_handoff(st.session_state, mode, planned)
            st.success("Contract copied to the editor. Run the investigation when ready.")


def render_model_diagnostics_page() -> None:
    """Render one investigation at a time and retain its stored results."""
    st.markdown('<div class="desk-kicker">Model diagnostics / 02</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="desk-title">Model Diagnostics</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="desk-subtitle">Compare prices and computational cost, '
        'then inspect the underlying lattice geometry and exercise decisions.</div>',
        unsafe_allow_html=True,
    )
    mode = st.segmented_control(
        "Investigation", DIAGNOSTIC_MODES,
        selection_mode="single", default="Compare models",
        key="model_diagnostics_view", width="stretch",
    ) or "Compare models"
    _render_shared_ticket_handoff(mode)
    if mode == "Compare models":
        st.caption(
            "Cross-model pricing, numerical reference and measured runtime. "
            "Choose Explore lattice to inspect the actual small-step trees."
        )
        from frontend.model_comparison import render_comparison_page
        render_comparison_page(embedded=True)
    else:
        st.caption(
            "Interactive lattice inspection up to N=100, with adaptive link rendering. "
            "Large-N benchmark configurations are not silently reduced."
        )
        from frontend.lattice_explorer import render_lattice_explorer_page
        render_lattice_explorer_page(embedded=True)

# LOT_5C5_STRUCTURAL_CLEANUP
