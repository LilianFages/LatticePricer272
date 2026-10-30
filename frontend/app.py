"""Lattice 272 — desk-style Streamlit front-end.

Launch from the repository root:
    python -m streamlit run frontend/app.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import date, timedelta
from functools import partial
from pathlib import Path

# Streamlit adds frontend/ to sys.path; pricing engines live at repo root/src/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from frontend.calendar_contract import CalendarInputError
from frontend.dividend_input import parse_cash_dividends
from frontend.calendar_ticket import dated_request, parse_dated_dividends
from frontend.model_selection import resolve_model_selection
from frontend.pricing_service import (
    DEFAULT_THRESHOLD,
    MAX_STEPS,
    FULL_GRAPH_MAX_STEPS,
    DividendInput,
    PricingInputError,
    PricingRequest,
    PricingResult,
    PRICING_ALGORITHMS,
    price_contract,
)
from frontend.components import result_panel
from frontend.trading_risk import compute_desk_risk
from frontend.contract_workflow import (
    clear_reference,
    get_reference,
    get_reference_calendar,
    save_reference_from_last_result,
)
from frontend.styles import apply_theme


st.set_page_config(
    page_title="Lattice 272 | Pricing Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


def read_dividend_table(frame: pd.DataFrame) -> tuple[DividendInput, ...]:
    """Parse the terminal's dividend editor using the shared UI validator."""
    return parse_cash_dividends(frame, context="terminal")


def _render_ticket_configuration(prefix: str) -> tuple[str, str, bool]:
    """Render date convention and model-family controls outside the pricing form."""
    date_col, model_col = st.columns([1.2, 1.0], gap="medium")
    with date_col:
        date_mode = st.radio(
            "Maturity convention", ("Calendar dates", "Time in years · legacy"),
            horizontal=True, key=f"{prefix}_date_mode",
            help="Changes the input editor only; it never reprices.",
        )
    with model_col:
        model_family = st.selectbox(
            "Model family", ("Binomial", "Hybrid", "Trinomial"),
            index=2, key=f"{prefix}_model_family",
            help="A financial model family, independent of the pricing algorithm.",
        )
    pruning_enabled = False
    if model_family == "Trinomial":
        pruning_mode = st.radio(
            "Trinomial pruning", ("On", "Off"), horizontal=True,
            key=f"{prefix}_pruning_mode",
            help="Off uses the full trinomial; On enables probability pruning.",
        )
        pruning_enabled = pruning_mode == "On"
    return date_mode, model_family, pruning_enabled


def _render_contract_dates(
    prefix: str, date_mode: str,
) -> tuple[date | None, date | None, float | None]:
    """Render the calendar or legacy maturity inputs inside the form."""
    st.markdown('<div class="desk-section">02 · Contract dates</div>', unsafe_allow_html=True)
    valuation_date = None
    expiry_date = None
    maturity = None
    if date_mode == "Calendar dates":
        today = date.today()
        a, b = st.columns(2)
        valuation_date = a.date_input(
            "Valuation date", value=today, key=f"{prefix}_valuation_date",
        )
        expiry_date = b.date_input(
            "Expiry date", value=today + timedelta(days=365),
            key=f"{prefix}_expiry_date",
        )
        if expiry_date > valuation_date:
            days = (expiry_date - valuation_date).days
            st.caption(f"ACT/365F · {days} calendar days · T = {days / 365:.6f} years")
        else:
            st.caption("Expiry must be strictly later than valuation date.")
    else:
        maturity = st.number_input(
            "Time to maturity · years", min_value=0.01, max_value=100.0,
            value=1.0, step=0.25, format="%.2f", key=f"{prefix}_maturity",
        )
    return valuation_date, expiry_date, maturity


def _render_market_inputs(prefix: str) -> tuple[float, float, float, float]:
    """Render spot, strike, volatility and risk-free rate inputs."""
    st.markdown('<div class="desk-section">03 · Market data</div>', unsafe_allow_html=True)
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
    return spot, strike, vol_pct, rate_pct


def _render_pricing_setup(
    prefix: str, model_family: str, pruning_enabled: bool,
) -> tuple[str, int, str, float]:
    """Render the pricing algorithm, lattice depth and pruning configuration."""
    st.markdown(
        '<div class="desk-section">04 · Model & numerical setup</div>',
        unsafe_allow_html=True,
    )
    engine_col, steps_col = st.columns(2, gap="small")
    algorithm = engine_col.selectbox(
        "Pricing algorithm", PRICING_ALGORITHMS,
        index=0, key=f"{prefix}_pricing_algorithm",
        help="Recursive memoization or iterative backward induction. "
             "Both use the same financial lattice and transitions.",
    )
    st.caption("One pricing algorithm for the premium and every Greek.")
    step_limit = MAX_STEPS if model_family == "Binomial" else FULL_GRAPH_MAX_STEPS
    steps = steps_col.number_input(
        "Time steps (N)", min_value=1, max_value=step_limit,
        value=100, step=25, key=f"{prefix}_steps",
        help=(
            "Binomial: up to N=10,000 via memory-bounded rolling arrays. "
            "Other models: up to N=350 to protect application memory."
        ),
    )
    if model_family == "Binomial" and steps > FULL_GRAPH_MAX_STEPS:
        st.caption(
            "Deep binomial mode: rolling arrays, price only. "
            "Greeks and tree visualization are not calculated."
        )
    pruning_method = "Allocation"
    pruning_threshold = DEFAULT_THRESHOLD
    if model_family == "Trinomial" and pruning_enabled:
        with st.expander("Trinomial pruning · Advanced", expanded=False):
            pruning_method = st.selectbox(
                "Implementation", ("Allocation", "Branch"),
                key=f"{prefix}_pruning_method",
                help="Select allocation or branch pruning for research comparability.",
            )
            pruning_threshold = st.selectbox(
                "Pruning threshold", (1e-18, 1e-15, 1e-12),
                key=f"{prefix}_threshold",
                format_func=lambda value: f"{value:.0e}",
            )
    return algorithm, steps, pruning_method, pruning_threshold


def _render_dividend_editor(
    prefix: str, date_mode: str, valuation_date: date | None,
) -> pd.DataFrame:
    """Render the model-compatible cash-dividend table editor."""
    with st.expander("Cash dividends · optional", expanded=False):
        if date_mode == "Calendar dates":
            st.caption("Ex-dates are calendar dates. Amounts are EUR per share. "
                       "A zero-amount row is ignored.")
            dividend_frame = st.data_editor(
                pd.DataFrame([{
                    "ex_date": valuation_date + timedelta(days=182),
                    "amount": 0.0,
                }]),
                num_rows="dynamic", hide_index=True, width="stretch",
                key=f"{prefix}_dated_dividend_table",
                column_config={
                    "ex_date": st.column_config.DateColumn("Ex-dividend date", format="DD/MM/YYYY"),
                    "amount": st.column_config.NumberColumn(
                        "Cash amount · €", min_value=0.0,
                        max_value=1000000.0, step=0.50, format="%.2f",
                    ),
                },
            )
        else:
            st.caption("Dates expressed in years from pricing. "
                       "A zero-amount row is ignored.")
            dividend_frame = st.data_editor(
                pd.DataFrame([{"time": 0.5, "amount": 0.0}]),
                num_rows="dynamic", hide_index=True, width="stretch",
                key=f"{prefix}_dividend_table",
                column_config={
                    "time": st.column_config.NumberColumn(
                        "Payment time · years", min_value=0.0,
                        max_value=100.0, step=0.05, format="%.4f",
                    ),
                    "amount": st.column_config.NumberColumn(
                        "Cash amount · €", min_value=0.0,
                        max_value=1000000.0, step=0.50, format="%.2f",
                    ),
                },
            )
        st.caption("Binomial does not support discrete cash dividends.")
    return dividend_frame


def build_ticket_form(reset_nonce: int) -> tuple[bool, PricingRequest | None, str | None]:
    """Return submitted status and either a validated shape or parsing error."""
    prefix = f"ticket_{reset_nonce}"
    date_mode, model_family, pruning_enabled = _render_ticket_configuration(prefix)
    with st.form(f"pricing_ticket_{reset_nonce}", clear_on_submit=False):
        st.markdown('<div class="desk-section">01 · Contract</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        option_type = a.selectbox("Option", ("Call", "Put"), key=f"{prefix}_option")
        exercise = b.selectbox("Exercise", ("European", "American"), key=f"{prefix}_exercise")
        valuation_date, expiry_date, maturity = _render_contract_dates(prefix, date_mode)
        spot, strike, vol_pct, rate_pct = _render_market_inputs(prefix)
        algorithm, steps, pruning_method, pruning_threshold = _render_pricing_setup(
            prefix, model_family, pruning_enabled,
        )
        dividend_frame = _render_dividend_editor(prefix, date_mode, valuation_date)
        submitted = st.form_submit_button(
            "RUN PRICING  →", type="primary", width="stretch",
        )

    if not submitted:
        # Render navigation and global page chrome without changing active widgets.
        return False, None, None
    st.session_state.pop("ticket_submitted_calendar", None)
    st.session_state.pop("ticket_submitted_algorithm", None)
    try:
        model, pruning_threshold = resolve_model_selection(
            model_family, pruning_enabled, pruning_method, pruning_threshold,
        )
        request = PricingRequest(
            spot=float(spot), strike=float(strike), rate=float(rate_pct) / 100.0,
            volatility=float(vol_pct) / 100.0,
            maturity=float(maturity) if maturity is not None else 1.0,
            steps=int(steps), option_type=option_type, exercise=exercise,
            model=model, pruning_threshold=pruning_threshold,
        )
        if date_mode == "Calendar dates":
            dates = parse_dated_dividends(dividend_frame)
            request, calendar = dated_request(
                request, valuation_date, expiry_date, dates,
            )
            st.session_state["ticket_submitted_calendar"] = calendar
        else:
            request = replace(request, dividends=read_dividend_table(dividend_frame))
        st.session_state["ticket_submitted_algorithm"] = algorithm
        return True, request, None
    except (PricingInputError, CalendarInputError) as exc:
        return True, None, str(exc)


def reset_ticket() -> None:
    """Create fresh widget identities; preserve session blotter."""
    st.session_state["ticket_reset_nonce"] = st.session_state.get("ticket_reset_nonce", 0) + 1
    st.session_state["last_pricing_result"] = None
    st.session_state.pop("desk_risk_snapshot", None)
    st.session_state.pop("desk_risk_error", None)
    st.session_state.pop("desk_greek_curve", None)
    st.session_state.pop("last_pricing_calendar", None)
    st.session_state.pop("ticket_submitted_calendar", None)
    st.session_state.pop("spot_revaluation", None)

def _initialize_session_state() -> None:
    if "ticket_reset_nonce" not in st.session_state:
        st.session_state["ticket_reset_nonce"] = 0
    if "last_pricing_result" not in st.session_state:
        st.session_state["last_pricing_result"] = None
    if "pricing_history" not in st.session_state:
        st.session_state["pricing_history"] = []


def _render_reference_sidebar() -> None:
    """Display saved reference contract without touching session state."""
    reference = get_reference(st.session_state)
    if reference is None:
        st.caption("No reference saved yet.")
    else:
        st.caption(
            f"{reference.exercise} {reference.option_type} · "
            f"S={reference.spot:g} € · K={reference.strike:g} €"
        )
        st.caption(
            f"{reference.model} · N={reference.steps} · "
            f"{len(reference.dividends)} cash dividend(s)"
        )
        reference_calendar = get_reference_calendar(st.session_state)
        if reference_calendar is not None:
            st.caption(
                f"Dates: {reference_calendar.valuation_date:%d/%m/%Y} → "
                f"{reference_calendar.expiry_date:%d/%m/%Y} · ACT/365F"
            )
        st.button(
            "Clear reference",
            on_click=clear_reference,
            args=(st.session_state,),
            key="reference_contract_clear",
            width="stretch",
            help="Remove only the shared reference, not your pricing results.",
        )


def _render_sidebar() -> str:
    from frontend.model_diagnostics import WORKSPACES
    with st.sidebar:
        st.markdown("### ◈ LATTICE 272")
        st.caption("DERIVATIVES PRICING · QUANT LAB")
        st.divider()
        # Preserve page navigation and avoid rerendering heavy analytics unnecessarily.
        st.markdown("**WORKSPACE**")
        workspace = st.radio(
            "Workspace",
            WORKSPACES,
            label_visibility="collapsed", key="lattice_workspace",
        )
        st.divider()
        st.markdown("**REFERENCE CONTRACT**")
        _render_reference_sidebar()
        st.divider()
        st.markdown("**RESEARCH**")
        st.caption("Five live investigations · optional historical evidence")
        st.divider()
        st.caption("Local Python engine · no market-data feed")
        st.caption("Model output is theoretical, not an executable quote.")
        if workspace == "01 / Pricing terminal":
            st.divider()
            st.button("↺ Reset ticket to defaults", on_click=reset_ticket, width="stretch",
                      help="Reset inputs and the displayed result; session blotter is retained.")
    return workspace


def _record_pricing_history(request: PricingRequest, result: PricingResult, selected_algorithm: str) -> None:
    """Store at most twelve successful valuations in the session blotter."""
    history = st.session_state["pricing_history"]
    history.insert(0, {
        "Model": request.model, "Engine": selected_algorithm,
        "Type": request.option_type,
        "Exercise": request.exercise, "K": request.strike,
        "N": request.steps, "Divs": len(request.dividends),
        "Price (€)": round(result.price, 8),
        "Time (ms)": round(result.elapsed_ms, 2),
    })
    del history[12:]


def _process_pricing_submission(
    request: PricingRequest | None, parsing_error: str | None,
) -> None:
    st.session_state["last_pricing_result"] = None
    st.session_state.pop("desk_risk_snapshot", None)
    st.session_state.pop("desk_risk_error", None)
    st.session_state.pop("desk_greek_curve", None)
    st.session_state.pop("last_pricing_calendar", None)
    st.session_state.pop("spot_revaluation", None)
    if parsing_error is not None:
        # Preserve error messages and empty states for model failures.
        st.error(parsing_error)
    elif request is not None:
        try:
            selected_algorithm = st.session_state.get(
                "ticket_submitted_algorithm", "Recursive"
            )
            with st.spinner("Running lattice pricing..."):
                result = price_contract(request, algorithm=selected_algorithm)
            st.session_state["last_pricing_result"] = result
            if request.model == "Binomial" and request.steps > FULL_GRAPH_MAX_STEPS:
                st.session_state["desk_risk_error"] = (
                    "Deep binomial mode calculates the premium only. "
                    "For Greeks and Delta hedge, select N <= 350."
                )
            else:
                _calculate_standard_risk(request, selected_algorithm)
            st.session_state["last_pricing_calendar"] = st.session_state.get(
                "ticket_submitted_calendar"
            )
            _record_pricing_history(request, result, selected_algorithm)
        except (PricingInputError, ValueError, RuntimeError, OverflowError,
                ZeroDivisionError) as exc:
            st.error(f"Pricing unavailable: {exc}")
            st.info("Check model compatibility, dividend dates and market inputs.")


def _calculate_standard_risk(request: PricingRequest, selected_algorithm: str) -> None:
    """Compute Greeks only for supported ordinary-depth lattices."""
    try:
        with st.spinner("Calculating desk Greeks..."):
            risk_snapshot = compute_desk_risk(
                request,
                pricer=partial(price_contract, algorithm=selected_algorithm),
            )
        st.session_state["desk_risk_snapshot"] = (request, risk_snapshot)
    except (PricingInputError, ValueError, RuntimeError,
            OverflowError, ZeroDivisionError) as risk_exc:
        st.session_state["desk_risk_error"] = str(risk_exc)


def _render_pricing_results(submitted: bool) -> None:
    last_result = st.session_state["last_pricing_result"]
    if last_result is not None:
        result_panel(last_result)
        st.button(
            "Save this contract as reference",
            on_click=save_reference_from_last_result,
            args=(st.session_state,),
            key="reference_contract_save",
            width="stretch",
            help="Keep these successfully priced inputs in this session; "
                 "other workspaces will import them only on request.",
        )
        if get_reference(st.session_state) == last_result.request:
            st.caption("Active reference · saved from this pricing ticket.")
        else:
            st.caption("Save this ticket to reuse its inputs across workspaces.")
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


def _render_pricing_terminal() -> None:
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
            _process_pricing_submission(request, parsing_error)

        _render_pricing_results(submitted)

    st.divider()
    st.caption(
        "LatticePricer272 · University quant research project · Inputs are user-supplied. "
        "No live prices, calibration or trading connectivity."
    )


def main() -> None:
    _initialize_session_state()
    from frontend.model_diagnostics import migrate_navigation_state
    migrate_navigation_state(st.session_state)

    workspace = _render_sidebar()

    # Validate form inputs before triggering numerical pricing workflows.
    if workspace == "02 / Model diagnostics":
        from frontend.model_diagnostics import render_model_diagnostics_page
        render_model_diagnostics_page()
        return

    if workspace == "03 / Quant research":
        from frontend.quant_research import render_quant_research_page
        render_quant_research_page()
        return

    _render_pricing_terminal()


if __name__ == "__main__":
    main()
