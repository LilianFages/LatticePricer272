"""Common, side-effect-free Streamlit editors for Model Diagnostics.

The two investigations retain their own widget keys, imported defaults and
specific model/timing/geometry controls. No pricing is performed here.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from frontend.contract_workflow import get_reference
from frontend.diagnostics_contract import import_saved_reference


def render_option_pair(
    key_for: Callable[[str], str],
    default_for: Callable[[str, Any], Any],
    *,
    widget_names: tuple[str, str],
    default_option: str = "Call",
    default_exercise: str = "European",
) -> tuple[str, str]:
    """Render option kind and exercise; preserve existing Streamlit widget IDs."""
    option_choices = ("Call", "Put")
    exercise_choices = ("European", "American")
    a, b = st.columns(2)
    kind = a.selectbox(
        "Option", option_choices,
        index=option_choices.index(default_for("option_type", default_option)),
        key=key_for(widget_names[0]),
    )
    exercise = b.selectbox(
        "Exercise", exercise_choices,
        index=exercise_choices.index(default_for("exercise", default_exercise)),
        key=key_for(widget_names[1]),
    )
    return kind, exercise


def render_market_fields(
    key_for: Callable[[str], str],
    default_for: Callable[[str, Any], Any],
    *,
    widget_names: tuple[str, str, str, str, str],
    heading: str | None = None,
    spot_max: float | None = None,
    strike_max: float | None = None,
    maturity_max: float | None = None,
    rate_label: str = "Risk-free rate · %",
) -> tuple[float, float, float, float, float]:
    """One common financial ticket, with investigation-specific input limits."""
    if heading:
        st.markdown(f'<div class="desk-section">{heading}</div>',
                    unsafe_allow_html=True)
    spot_key, strike_key, vol_key, rate_key, maturity_key = widget_names
    a, b = st.columns(2)
    spot = a.number_input(
        "Spot · €", min_value=0.01, max_value=spot_max,
        value=float(default_for("spot", 100.0)), step=1.0,
        format="%.2f", key=key_for(spot_key),
    )
    strike = b.number_input(
        "Strike · €", min_value=0.0, max_value=strike_max,
        value=float(default_for("strike", 100.0)), step=1.0,
        format="%.2f", key=key_for(strike_key),
    )
    # Rates and volatilities are displayed in percent, not decimal units.
    a, b = st.columns(2)
    vol_pct = a.number_input(
        "Volatility · %", min_value=0.0, max_value=500.0,
        value=float(default_for("volatility", 0.20)) * 100.0,
        step=1.0, format="%.2f", key=key_for(vol_key),
    )
    rate_pct = b.number_input(
        rate_label, min_value=-50.0, max_value=100.0,
        value=float(default_for("rate", 0.02)) * 100.0,
        step=0.25, format="%.2f", key=key_for(rate_key),
    )
    # The target view retains its original maturity cap and widget key.
    maturity = st.number_input(
        "Maturity · years", min_value=0.01, max_value=maturity_max,
        value=float(default_for("maturity", 1.0)), step=0.25,
        format="%.2f", key=key_for(maturity_key),
    )
    return spot, strike, vol_pct, rate_pct, maturity


# Mode-specific widget IDs and copy; all import mechanics are shared.
_REFERENCE_IMPORT_UI = {
    "Compare models": (
        "cmp_import_contract",
        "cmp_import_feedback",
        "Reference contract imported into the form. Run comparison when ready.",
        "Copy contract/market, N, threshold and all dividends. "
        "Selected architectures, repetitions and price reference stay unchanged.",
        "Import fills the ticket only. Model selection and benchmark "
        "settings stay independent; no valuation runs automatically.",
    ),
    "Explore lattice": (
        "lx_import_reference",
        "explorer_import_feedback",
        "Reference contract imported. Build the lattices when ready.",
        "Requires 2 <= N <= 100, at most one cash dividend and a compatible "
        "pruning threshold. Incompatible references are never adjusted.",
        "Import fills the ticket only; left/right models remain independent. "
        "No lattice is built until BUILD & COMPARE LATTICES is clicked.",
    ),
}


def _reference_import_settings(mode: str) -> tuple[str, str, str, str, str]:
    try:
        return _REFERENCE_IMPORT_UI[mode]
    except KeyError:
        raise ValueError(f"Unknown diagnostic import mode: {mode}") from None


def on_reference_import(mode: str) -> None:
    """Perform only an explicit contract transfer, not a pricing run."""
    _, feedback_key, success, _, _ = _reference_import_settings(mode)
    accepted, issues = import_saved_reference(st.session_state, mode)
    st.session_state[feedback_key] = (
        ("success", success) if accepted
        else ("warning", "Import unavailable: " + " ".join(issues))
    )


def render_reference_import(mode: str) -> None:
    """Common reference-import widget for both Model Diagnostics investigations."""
    button_key, feedback_key, _, help_text, caption = _reference_import_settings(mode)
    st.markdown('<div class="desk-section">Reference contract · optional</div>',
                unsafe_allow_html=True)
    reference = get_reference(st.session_state)
    if reference is None:
        st.caption("Save a successfully priced contract in Pricing Terminal first.")
    st.button(
        "Import reference contract", on_click=on_reference_import, args=(mode,),
        disabled=reference is None, key=button_key, width="stretch", help=help_text,
    )
    feedback = st.session_state.get(feedback_key)
    if feedback is not None:
        level, message = feedback
        if level == "success":
            st.success(message)
        else:
            st.warning(message)
    st.caption(caption)
