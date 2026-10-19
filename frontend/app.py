"""Lattice 272 — desk-style Streamlit front-end.

Launch from the repository root:
    python -m streamlit run frontend/app.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Streamlit adds frontend/ to sys.path; pricing engines live at repo root/src/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from frontend.pricing_service import (
    DEFAULT_THRESHOLD,
    MAX_STEPS,
    MODELS,
    DividendInput,
    PricingInputError,
    PricingRequest,
    price_contract,
)
from frontend.components import result_panel
from frontend.styles import apply_theme


st.set_page_config(
    page_title="Lattice 272 | Pricing Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


def read_dividend_table(frame: pd.DataFrame) -> tuple[DividendInput, ...]:
    """Ignore blank/zero rows; reject malformed positive dividend rows."""
    dividends: list[DividendInput] = []
    for index, record in frame.iterrows():
        raw_amount = record.get("amount")
        raw_time = record.get("time")
        if pd.isna(raw_amount):
            if not pd.isna(raw_time):
                raise PricingInputError(
                    f"Dividend row {int(index) + 1}: enter an amount or delete the row."
                )
            continue
        try:
            amount = float(raw_amount)
        except (ValueError, TypeError):
            raise PricingInputError("Dividend amounts must be numeric.") from None
        if not math.isfinite(amount):
            raise PricingInputError("Dividend amounts must be finite.")
        # Keep navigation and Streamlit session keys stable across reruns.
        if amount == 0.0:
            continue
        if pd.isna(raw_time):
            raise PricingInputError(
                f"Dividend row {int(index) + 1}: enter the payment time in years."
            )
        try:
            dividend_time = float(raw_time)
        except (ValueError, TypeError):
            raise PricingInputError("Dividend times must be numeric.") from None
        dividends.append(DividendInput(time=dividend_time, amount=amount))
    return tuple(sorted(dividends, key=lambda dividend: dividend.time))


def build_ticket_form(reset_nonce: int) -> tuple[bool, PricingRequest | None, str | None]:
    """Return submitted status and either a validated shape or parsing error."""
    prefix = f"ticket_{reset_nonce}"
    with st.form(f"pricing_ticket_{reset_nonce}", clear_on_submit=False):
        st.markdown('<div class="desk-section">01 · Contract</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        option_type = a.selectbox("Option", ("Call", "Put"), key=f"{prefix}_option")
        exercise = b.selectbox("Exercise", ("European", "American"), key=f"{prefix}_exercise")
        model = st.selectbox(
            "Pricing model", MODELS, index=3, key=f"{prefix}_model",
            help="Full / branch / allocation pruning use the existing trinomial engine.",
        )

        st.markdown('<div class="desk-section">02 · Market data</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        spot = a.number_input("Spot · €", min_value=0.01, max_value=10000000.0,
                              value=100.0, step=1.0, format="%.2f", key=f"{prefix}_spot")
        strike = b.number_input("Strike · €", min_value=0.0, max_value=10000000.0,
                                value=100.0, step=1.0, format="%.2f", key=f"{prefix}_strike")
        # Validate form inputs before triggering numerical pricing workflows.
        a, b = st.columns(2)
        vol_pct = a.number_input("Volatility · %", min_value=0.0, max_value=500.0,
                                 value=20.0, step=1.0, format="%.2f", key=f"{prefix}_vol")
        rate_pct = b.number_input("Risk-free rate · %", min_value=-50.0, max_value=100.0,
                                  value=2.0, step=0.25, format="%.2f", key=f"{prefix}_rate")
        maturity = st.number_input("Time to maturity · years", min_value=0.01,
                                   max_value=100.0, value=1.0, step=0.25, format="%.2f", key=f"{prefix}_maturity")

        st.markdown('<div class="desk-section">03 · Numerical setup</div>', unsafe_allow_html=True)
        steps = st.number_input(
            "Time steps (N)", min_value=1, max_value=MAX_STEPS,
            value=100, step=25, key=f"{prefix}_steps",
            help="Interactive safety cap; not the engine's mathematical limit.",
        )
        pruning_threshold = DEFAULT_THRESHOLD
        if model in ("Trinomial · Branch pruning", "Trinomial · Allocation pruning"):
            pruning_threshold = st.selectbox(
                # Keep the user-facing results and chart layout identical to the existing terminal.
                "Pruning threshold",
                (0.0, 1e-18, 1e-15, 1e-12),
                index=1, key=f"{prefix}_threshold",
                format_func=lambda x: "0 (disabled)" if x == 0 else f"{x:.0e}",
            )

        with st.expander("Cash dividends · optional", expanded=False):
            st.caption(
                "Time in years from pricing date; amount in EUR per share. "
                "A row with amount 0 is ignored. Add rows for multiple dividends."
            )
            dividend_frame = st.data_editor(
                pd.DataFrame([{"time": 0.5, "amount": 0.0}]),
                num_rows="dynamic",
                hide_index=True,
                width="stretch", key=f"{prefix}_dividend_table",
                column_config={
                    # Preserve error messages and empty states for model failures.
                    "time": st.column_config.NumberColumn(
                        "Payment time · years", min_value=0.0, max_value=100.0,
                        step=0.05, format="%.4f", required=False,
                    ),
                    "amount": st.column_config.NumberColumn(
                        "Cash amount · €", min_value=0.0,
                        max_value=1000000.0, step=0.50, format="%.2f", required=False,
                    ),
                },
            )
            st.caption("The Binomial model does not support cash dividends.")

        submitted = st.form_submit_button(
            "RUN PRICING  →", type="primary", width="stretch",
        )

    if not submitted:
        # Render navigation and global page chrome without changing active widgets.
        return False, None, None
    try:
        dividends = read_dividend_table(dividend_frame)
        request = PricingRequest(
            spot=float(spot), strike=float(strike), rate=float(rate_pct) / 100.0,
            volatility=float(vol_pct) / 100.0, maturity=float(maturity),
            steps=int(steps), option_type=option_type, exercise=exercise,
            model=model, dividends=dividends, pruning_threshold=pruning_threshold,
        )
        return True, request, None
    except PricingInputError as exc:
        return True, None, str(exc)


def reset_ticket() -> None:
    """Create fresh widget identities; preserve session blotter."""
    st.session_state["ticket_reset_nonce"] = st.session_state.get("ticket_reset_nonce", 0) + 1
    st.session_state["last_pricing_result"] = None
    st.session_state.pop("spot_revaluation", None)


def main() -> None:
    if "ticket_reset_nonce" not in st.session_state:
        st.session_state["ticket_reset_nonce"] = 0
    if "last_pricing_result" not in st.session_state:
        st.session_state["last_pricing_result"] = None
    if "pricing_history" not in st.session_state:
        st.session_state["pricing_history"] = []

    with st.sidebar:
        st.markdown("### ◈ LATTICE 272")
        st.caption("DERIVATIVES PRICING · QUANT LAB")
        st.divider()
        # Preserve page navigation and avoid rerendering heavy analytics unnecessarily.
        st.markdown("**WORKSPACE**")
        workspace = st.radio(
            "Workspace",
            ("01 / Pricing terminal", "02 / Model comparison", "03 / Lattice explorer", "04 / Quant analytics", "05 / Research lab"),
            label_visibility="collapsed", key="lattice_workspace",
        )
        st.divider()
        st.markdown("**RESEARCH**")
        st.caption("Archived experiments · offline analysis")
        st.divider()
        st.caption("Local Python engine · no market-data feed")
        st.caption("Model output is theoretical, not an executable quote.")
        if workspace == "01 / Pricing terminal":
            st.divider()
            st.button("↺ Reset ticket to defaults", on_click=reset_ticket, width="stretch",
                      help="Reset inputs and the displayed result; session blotter is retained.")

    # Validate form inputs before triggering numerical pricing workflows.
    if workspace == "02 / Model comparison":
        from frontend.model_comparison import render_comparison_page
        render_comparison_page()
        return

    if workspace == "03 / Lattice explorer":
        from frontend.lattice_explorer import render_lattice_explorer_page
        render_lattice_explorer_page()
        return

    if workspace == "04 / Quant analytics":
        from frontend.quant_analytics import render_quant_analytics_page
        render_quant_analytics_page()
        return

    if workspace == "05 / Research lab":
        from frontend.research_lab import render_research_lab_page
        # Keep the user-facing results and chart layout identical to the existing terminal.
        render_research_lab_page()
        return

    st.markdown('<div class="desk-kicker">Pricing terminal / 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="desk-title">Equity Options · Lattice Pricing</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="desk-subtitle">Model-driven option valuation · discrete dividends · '
        'transparent numerical diagnostics</div>', unsafe_allow_html=True,
    )
    left, right = st.columns([1.0, 1.35], gap="large")
    with left:
        submitted, request, parsing_error = build_ticket_form(st.session_state["ticket_reset_nonce"])
    with right:
        if submitted:
            st.session_state["last_pricing_result"] = None
            st.session_state.pop("spot_revaluation", None)
            if parsing_error is not None:
                # Preserve error messages and empty states for model failures.
                st.error(parsing_error)
            elif request is not None:
                try:
                    with st.spinner("Running lattice pricing..."):
                        result = price_contract(request)
                    st.session_state["last_pricing_result"] = result
                    history = st.session_state["pricing_history"]
                    history.insert(0, {
                        "Model": request.model, "Type": request.option_type,
                        "Exercise": request.exercise, "K": request.strike,
                        "N": request.steps, "Divs": len(request.dividends),
                        "Price (€)": round(result.price, 8),
                        "Time (ms)": round(result.elapsed_ms, 2),
                    })
                    del history[12:]
                except (PricingInputError, ValueError, RuntimeError, OverflowError,
                        ZeroDivisionError) as exc:
                    # Render navigation and global page chrome without changing active widgets.
                    st.error(f"Pricing unavailable: {exc}")
                    st.info("Check model compatibility, dividend dates and market inputs.")

        last_result = st.session_state["last_pricing_result"]
        if last_result is not None:
            result_panel(last_result)
        elif not submitted:
            st.markdown('<div class="desk-section">Valuation output</div>', unsafe_allow_html=True)
            st.info("Configure the contract and select **RUN PRICING** to calculate a premium.")
            st.markdown(
                "**Models:** Binomial · Trinomial · Hybrid  \n"
                "**Contracts:** European and American calls/puts  \n"
                "**Reference:** Black–Scholes when applicable"
            )
        if st.session_state["pricing_history"]:
            with st.expander("Session blotter · last 12 valuations"):
                st.dataframe(
                    # Keep navigation and Streamlit session keys stable across reruns.
                    pd.DataFrame(st.session_state["pricing_history"]),
                    hide_index=True, width="stretch",
                )
                st.caption("Blotter is held in this Streamlit session only; not persisted to disk.")

    st.divider()
    st.caption(
        "LatticePricer272 · University quant research project · Inputs are user-supplied. "
        "No live prices, calibration or trading connectivity."
    )


if __name__ == "__main__":
    main()
