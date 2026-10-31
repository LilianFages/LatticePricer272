"""Trading-desk risk panel: seven Greeks, on-demand chart and optional hedge.

The only position inputs appear inside the collapsed Delta Hedge subsection.
"""

from __future__ import annotations

from functools import partial

import altair as alt
import pandas as pd
import streamlit as st

from frontend.pricing_service import PricingInputError, PricingResult, price_contract
from frontend.trading_risk import (
    GREEK_NAMES, GREEK_UNITS, DeskRiskSnapshot, StockHedge, calculate_stock_hedge,
    compute_greek_curve, greek_table,
)


def _compute_greek_curve_on_demand(result: PricingResult, snapshot: DeskRiskSnapshot, chosen: str) -> None:
    """Compute the requested five-spot curve only on explicit user action."""
    if st.button("Calculate selected Greek curve · 5 spots", key="desk_plot_greek",
                 width="stretch"):
        st.session_state.pop("desk_greek_curve", None)
        with st.spinner(f"Calculating {chosen} over five spot levels..."):
            try:
                points = compute_greek_curve(
                    result.request, chosen, count=5, span_pct=10.0,
                    bumps=snapshot.base.bumps,
                    pricer=partial(price_contract, algorithm=result.algorithm),
                )
            except (PricingInputError, ValueError, RuntimeError, OverflowError,
                    ZeroDivisionError) as exc:
                st.error(f"Greek curve unavailable: {exc}")
            else:
                st.session_state["desk_greek_curve"] = (
                    result.request, result.algorithm, chosen, points,
                )


def _render_greek_curve_chart(points: tuple, chosen: str) -> None:
    """Render measured points without replacing failed scenarios by zero."""
    successful = [{"Spot (€)": point.spot, "Greek value": point.value}
                  for point in points if point.value is not None]
    if len(successful) >= 2:
        frame = pd.DataFrame(successful)
        chart = (
            alt.Chart(frame)
            .mark_line(point=alt.OverlayMarkDef(size=65), strokeWidth=2.2)
            .encode(
                x=alt.X("Spot (€):Q", title="Underlying spot (€)",
                        scale=alt.Scale(zero=False)),
                y=alt.Y("Greek value:Q", title=f"{chosen} · {GREEK_UNITS[chosen]}",
                        scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("Spot (€):Q", format=".2f"),
                         alt.Tooltip("Greek value:Q", format=".7f")],
            )
            .properties(height=275)
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.warning("Not enough successful scenarios to plot a curve.")


def _render_greek_curve_warnings(points: tuple) -> None:
    """Describe failed pricing scenarios and bump discretization."""
    failed = [point for point in points if point.error is not None]
    if failed:
        st.warning(
            f"{len(failed)} / {len(points)} scenarios unavailable; "
            "failed points were not replaced by zero."
        )
        with st.expander("Curve failures · details", expanded=False):
            for point in failed:
                st.caption(f"S={point.spot:.4f} €: {point.error}")
    st.caption(
        "Five separate risk snapshots at spot ±10% (fixed expiry, model and N). "
        "Bump-based tree Greeks can oscillate with spot and discretization."
    )


def _show_curve(result: PricingResult, snapshot: DeskRiskSnapshot) -> None:
    st.markdown('<div class="desk-section">Greek curve · on demand</div>', unsafe_allow_html=True)
    chosen = st.selectbox(
        "Greek to visualize", GREEK_NAMES, key="desk_selected_greek",
        help="A curve shows the selected Greek at several spots, not the payoff.",
    )
    # Every curve request is explicit: no repricing just to switch the selector.
    _compute_greek_curve_on_demand(result, snapshot, chosen)
    cached = st.session_state.get("desk_greek_curve")
    if not (isinstance(cached, tuple) and len(cached) == 4
            and cached[0] == result.request
            and cached[1] == result.algorithm and cached[2] == chosen):
        st.caption("Select a Greek and click Calculate to display its spot-sensitivity curve.")
        return
    _, _, _, points = cached
    _render_greek_curve_chart(points, chosen)
    _render_greek_curve_warnings(points)


def _render_delta_hedge_trade(hedge: StockHedge) -> None:
    """Show the signed whole-share trade and residual exposure."""
    label = (
        f"{hedge.instruction} {abs(hedge.rounded_trade):,} shares"
        if hedge.instruction != "HOLD" else "HOLD · no shares to trade"
    )
    st.metric("Delta-neutral trade · whole shares", label)
    a, b = st.columns(2)
    a.metric("Exact trade · signed shares", f"{hedge.exact_trade:+,.4f}")
    b.metric(
        "Residual delta after rounded trade",
        f"{hedge.residual_delta_shares:+,.4f} shares",
    )
    st.caption(
        f"Option position delta: {hedge.option_delta_shares:+,.4f} share-equivalents. "
        f"Target stock position: {hedge.target_shares:+,.4f} shares. "
        "This is a model-based theoretical hedge, not a trading "
        "instruction or an executed order."
    )


def _show_delta_hedge(snapshot: DeskRiskSnapshot) -> None:
    with st.expander("Delta hedge · shares (optional)", expanded=False):
        st.caption(
            "Position sizing is optional and affects only the stock-hedge instruction, "
            "not the option premium or the Greeks. Positive trade = BUY; negative = SELL."
        )
        a, b = st.columns(2)
        side = a.selectbox("Option position", ("Long", "Short"),
                           key="desk_hedge_side")
        contracts = b.number_input("Number of contracts", min_value=1,
                                   max_value=1_000_000, value=1, step=1,
                                   key="desk_hedge_contracts")
        a, b = st.columns(2)
        multiplier = a.number_input("Shares per contract", min_value=1,
                                    max_value=1_000_000, value=100, step=1,
                                    key="desk_hedge_multiplier",
                                    help="100 is illustrative. Verify the actual "
                                         "option contract specification.")
        existing = b.number_input("Existing stock shares · signed", value=0.0,
                                  step=1.0, key="desk_hedge_shares",
                                  help="Long stock positive; short stock negative.")
        # Hedge quantities depend on the chosen position, not on model repricing.
        delta = snapshot.value("Delta")
        if delta is None:
            st.warning("Delta unavailable: stock hedge cannot be calculated.")
            return
        try:
            hedge = calculate_stock_hedge(
                delta, side, int(contracts), int(multiplier), float(existing),
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        _render_delta_hedge_trade(hedge)


def render_desk_risk(result: PricingResult) -> None:
    """Render only the successfully priced contract's cached Greeks."""
    st.markdown('<div class="desk-section">Risk · Greeks & Delta hedge</div>',
                unsafe_allow_html=True)
    cached = st.session_state.get("desk_risk_snapshot")
    if not (isinstance(cached, tuple) and len(cached) == 2
            and cached[0] == result.request
            and isinstance(cached[1], DeskRiskSnapshot)):
        risk_error = st.session_state.get("desk_risk_error")
        if risk_error:
            st.warning(
                f"Pricing succeeded, but Greek calculation was unavailable: {risk_error}"
            )
        else:
            st.caption("The Greek risk snapshot is calculated when RUN PRICING succeeds.")
        return
    snapshot = cached[1]
    frame = pd.DataFrame(greek_table(snapshot))
    st.dataframe(frame, hide_index=True, width="stretch",
                 column_config={
                     "Model": st.column_config.NumberColumn(format="%.7f"),
                     "Black-Scholes": st.column_config.NumberColumn(format="%.7f"),
                     "Model - BS": st.column_config.NumberColumn(format="%+.7f"),
                 })
    st.caption(
        "Vega and Vanna: per +1 volatility percentage point. Vomma: per (1pp)²; "
        "Theta: 1-day roll-down; Rho: per +1 bp. "
        "Black–Scholes columns are available only for eligible European "
        "options without cash dividends."
    )
    # Preserve numerical warnings from both basic and second-order Greeks.
    problems = (*snapshot.base.issues, *snapshot.extra_issues)
    if problems:
        with st.expander(f"Numerical warnings · {len(problems)}", expanded=False):
            for warning in problems:
                st.warning(warning)
    _show_curve(result, snapshot)
    _show_delta_hedge(snapshot)
