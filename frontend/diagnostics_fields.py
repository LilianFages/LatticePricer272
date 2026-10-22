"""Common, side-effect-free Streamlit editors for Model Diagnostics.

The two investigations retain their own widget keys, imported defaults and
specific model/timing/geometry controls. No pricing is performed here.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st


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
    maturity = st.number_input(
        "Maturity · years", min_value=0.01, max_value=maturity_max,
        value=float(default_for("maturity", 1.0)), step=0.25,
        format="%.2f", key=key_for(maturity_key),
    )
    return spot, strike, vol_pct, rate_pct, maturity
