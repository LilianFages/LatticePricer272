"""Reusable desk-style pricing outputs; no engine equations defined here."""

from __future__ import annotations

import csv
import io

import altair as alt
import pandas as pd
import streamlit as st

from frontend.pricing_service import (
    PricingRequest,
    PricingResult,
    SpotSensitivityPoint,
    compute_spot_sensitivity,
    ticket_csv,
)


def payoff_chart(request: PricingRequest) -> alt.Chart:
    """Terminal contractual payoff, not option present value or P&L."""
    middle = max(request.spot, request.strike, 1.0)
    lower = max(0.0, min(request.spot, request.strike) * 0.55)
    upper = middle * 1.6
    spots = [lower + (upper - lower) * i / 100 for i in range(101)]
    if request.option_type == "Call":
        payoffs = [max(s - request.strike, 0.0) for s in spots]
    else:
        payoffs = [max(request.strike - s, 0.0) for s in spots]
    frame = pd.DataFrame({"Spot at expiry (€)": spots, "Payoff (€)": payoffs})
    return (
        alt.Chart(frame)
        .mark_line(color="#84C8ED", strokeWidth=2.3)
        .encode(
            x=alt.X("Spot at expiry (€):Q", scale=alt.Scale(zero=False)),
            y=alt.Y("Payoff (€):Q", scale=alt.Scale(zero=True)),
            tooltip=[
                alt.Tooltip("Spot at expiry (€):Q", format=".2f"),
                alt.Tooltip("Payoff (€):Q", format=".4f"),
            ],
        )
        .properties(height=215)
    )


def spot_curve_chart(
    points: tuple[SpotSensitivityPoint, ...], current_spot: float,
) -> alt.LayerChart:
    """Current model premium versus varied initial spot; each point is repriced."""
    valid = [
        {"Spot (€)": row.spot, "Model price (€)": row.price}
        for row in points if row.price is not None
    ]
    frame = pd.DataFrame(valid)
    line = (
        alt.Chart(frame)
        .mark_line(color="#8AD1F8", point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("Spot (€):Q", scale=alt.Scale(zero=False)),
            y=alt.Y("Model price (€):Q", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("Spot (€):Q", format=".2f"),
                alt.Tooltip("Model price (€):Q", format=".6f"),
            ],
        )
    )
    # Render option metrics using consistent monetary units and precision.
    spot_rule = (
        alt.Chart(pd.DataFrame({"Spot (€)": [current_spot]}))
        .mark_rule(color="#C5A16B", strokeDash=[5, 4], opacity=0.85)
        .encode(x="Spot (€):Q")
    )
    return (line + spot_rule).properties(height=245)


def sensitivity_csv(points: tuple[SpotSensitivityPoint, ...]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["spot", "option_value", "error"])
    for point in points:
        writer.writerow([point.spot, point.price, point.error or ""])
    return output.getvalue()


# Render option metrics using consistent monetary units and precision.
def _render_cached_spot_revaluation(cached: tuple, result: PricingResult) -> None:
    points: tuple[SpotSensitivityPoint, ...] = cached[1]
    valid = [point for point in points if point.price is not None]
    if len(valid) >= 2:
        st.altair_chart(spot_curve_chart(points, result.request.spot), width="stretch")
    else:
        st.warning("Not enough valid spot points to display a sensitivity curve.")
    failures = [point for point in points if point.price is None]
    if failures:
        st.warning(
            f"{len(failures)} of {len(points)} scenarios could not be priced "
            "under the engine's existing assumptions. They are excluded from the curve."
        )
        with st.expander("Unavailable scenarios"):
            st.dataframe(pd.DataFrame([
                {"Spot (€)": row.spot, "Reason": row.error} for row in failures
            ]), hide_index=True, width="stretch")
    st.download_button(
        # Maintain the existing semantic hierarchy of cards and indicators.
        "Export spot scenarios · CSV", data=sensitivity_csv(points),
        file_name="lattice272_spot_revaluation.csv", mime="text/csv", width="stretch",
    )


def _sensitivity_panel(result: PricingResult) -> None:
    """On-demand spot repricing, separate from payoff and the pricing timer."""
    st.markdown('<div class="desk-section">Spot revaluation</div>', unsafe_allow_html=True)
    st.caption(
        "Reprice the same contract at nine spot levels (±20%). "
        "Other inputs remain fixed; these are model values today, not terminal payoffs or Greeks."
    )
    if st.button("Calculate spot value curve · 9 repricings", key="spot_revaluation_run", width="stretch"):
        try:
            with st.spinner("Repricing nine spot scenarios..."):
                points = compute_spot_sensitivity(result.request, points=9, span=0.20)
            st.session_state["spot_revaluation"] = (result.request, points)
        except Exception as exc:
            st.session_state.pop("spot_revaluation", None)
            st.error(f"Spot sensitivity unavailable: {exc}")

    cached = st.session_state.get("spot_revaluation")
    if cached is None or cached[0] != result.request:
        return
    # Keep reusable UI elements independent of the pricing engine.
    _render_cached_spot_revaluation(cached, result)


def _render_allocation_diagnostics(result: PricingResult) -> None:
    if result.allocated_nodes is not None:
        with st.expander("Allocation pruning diagnostics", expanded=False):
            if result.full_trinomial_reference_nodes is not None:
                a, b = st.columns(2)
                a.metric("Full trinomial · theoretical nodes", f"{result.full_trinomial_reference_nodes:,}")
                b.metric("Avoided allocations", f"{result.allocation_savings_pct:.1f}%")
                st.caption(
                    "Exact full-trinomial geometric node count for the no-dividend, "
                    "positive-volatility case. Reachable nodes and allocated nodes "
                    "can legitimately be equal in the optimized tree."
                )
            else:
                # Keep reusable UI elements independent of the pricing engine.
                st.caption(
                    "Full-tree allocation baseline not displayed with cash dividends "
                    "or zero volatility: dynamic column sizes require a separately "
                    "constructed comparable tree. This does not affect the quoted price."
                )
            if result.pruned_transitions is not None:
                st.metric("Simplified outgoing transitions", f"{result.pruned_transitions:,}")


def _render_pricing_benchmarks(result: PricingResult) -> None:
    if result.black_scholes is not None:
        st.markdown('<div class="desk-section">Analytical benchmark</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.metric("Black–Scholes", f"{result.black_scholes:,.6f} €")
        b.metric("Model − BS", f"{result.black_scholes_error:+.6f} €")
        rel = result.black_scholes_relative_error_pct
        # Render option metrics using consistent monetary units and precision.
        c.metric("Relative deviation", f"{rel:+.3f}%" if rel is not None else "N/A")
        st.caption("Reference for European vanilla options without cash dividends only.")
    else:
        st.caption("Black–Scholes analytical reference not applicable to this configuration.")

    _render_allocation_diagnostics(result)


def result_panel(result: PricingResult) -> None:
    req = result.request
    st.markdown('<div class="desk-section">Last priced ticket</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="desk-ticket">'
        '<div class="name">Calculated model premium · EUR</div>'
        f'<div class="price">{result.price:,.6f} €</div>'
        f'<div class="description">{req.option_type} · {req.exercise} · '
        f'{req.model} · {req.steps} steps</div></div>',
        unsafe_allow_html=True,
    )

    # Preserve the current HTML class names and CSS hooks.
    a, b, c = st.columns(3)
    a.metric("Pricing runtime", f"{result.elapsed_ms:.2f} ms", help="Constructor, build and pricing only.")
    b.metric("Reachable nodes", f"{result.reachable_nodes:,}", help="Unique nodes reached through the pricing links.")
    c.metric(
        "Allocated nodes" if result.allocated_nodes is not None else "Pruned transitions",
        (
            f"{result.allocated_nodes:,}" if result.allocated_nodes is not None
            else (f"{result.pruned_transitions:,}" if result.pruned_transitions is not None else "—")
        ),
    )

    _render_pricing_benchmarks(result)

    # The risk snapshot is cached by Pricing Terminal on a successful ticket.
    from frontend.trading_risk_views import render_desk_risk
    render_desk_risk(result)

    with st.expander("Terminal payoff (contractual)", expanded=False):
        st.altair_chart(payoff_chart(req), width="stretch")
        st.caption("Payoff at maturity versus terminal stock price; not present-value sensitivity.")

    _sensitivity_panel(result)
    st.download_button(
        "Export pricing ticket · CSV", data=ticket_csv(result),
        file_name="lattice272_pricing_ticket.csv", mime="text/csv", width="stretch",
    )
