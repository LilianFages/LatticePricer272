"""Shared Quant Research contract form and request adapter.

Used by accuracy, efficiency, market sweeps and Greek numerical studies.
Input widget keys and pricing conventions are intentionally unchanged.
All valuation and dividend schedule validation remain in the existing services.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.dividend_input import parse_dividends
from frontend.pricing_service import PricingRequest


def render_research_contract_fields(prefix: str) -> dict:
    """Reusable but independently keyed input fields inside the calling form."""
    st.markdown('<div class="desk-section">01 · Contract & market</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    option = a.selectbox("Option", ("Call", "Put"), key=f"{prefix}_option")
    exercise = b.selectbox("Exercise", ("European", "American"), key=f"{prefix}_exercise")
    a, b = st.columns(2)
    spot = a.number_input("Spot · €", min_value=0.01, value=100.0,
                          key=f"{prefix}_spot", format="%.2f")
    strike = b.number_input("Strike · €", min_value=0.0, value=100.0,
                            key=f"{prefix}_strike", format="%.2f")
    a, b = st.columns(2)
    vol = a.number_input("Volatility · %", min_value=0.0, max_value=500.0,
                         value=20.0, key=f"{prefix}_vol")
    rate = b.number_input("Risk-free rate · %", min_value=-50.0, max_value=100.0,
                          value=2.0, key=f"{prefix}_rate")
    maturity = st.number_input("Maturity · years", min_value=0.01, value=1.0,
                               key=f"{prefix}_maturity")
    with st.expander("Cash dividends · optional"):
        # Collect model-sensitivity inputs through the current Streamlit form.
        st.caption("Discrete cash payments from today, expressed in years and € per share.")
        data = st.data_editor(
            pd.DataFrame([{"time": 0.5, "amount": 0.0}]),
            key=f"{prefix}_dividends", num_rows="dynamic", hide_index=True,
            width="stretch",
            column_config={
                "time": st.column_config.NumberColumn("Payment time · years", format="%.4f"),
                "amount": st.column_config.NumberColumn("Cash amount · €", format="%.2f"),
            },
        )
    return dict(spot=spot, strike=strike, rate=rate/100.0,
                volatility=vol/100.0, maturity=maturity, option_type=option,
                exercise=exercise, dividends_frame=data)


def build_research_request(values: dict, model: str, n: int, threshold: float) -> PricingRequest:
    return PricingRequest(
        # Preserve on-demand execution to avoid unnecessary valuations.
        spot=float(values["spot"]), strike=float(values["strike"]),
        rate=float(values["rate"]), volatility=float(values["volatility"]),
        maturity=float(values["maturity"]), option_type=values["option_type"],
        exercise=values["exercise"],
        dividends=parse_dividends(values["dividends_frame"]),
        model=model, steps=n, pruning_threshold=threshold,
    )
