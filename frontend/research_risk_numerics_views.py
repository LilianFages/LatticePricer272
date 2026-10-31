"""Streamlit Risk Numerics dashboard, from existing priced Greek observations.

No pricing occurs on view toggles, zoom or report exploration.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import altair as alt
import pandas as pd
import streamlit as st

from frontend.greeks_service import analytic_bs_greeks
from frontend.research_risk_numerics import StabilitySummary, observations, summarize_stability


GREEKS = ("Delta", "Gamma")
DIMENSIONS = (("spot bump", "Spot-bump width (%)"), ("steps", "Lattice depth N"))


def _risk_chart(rows: tuple, dimension: str, greek: str,
                analytical: float | None, deviation: bool) -> alt.Chart | None:
    """One observed curve, with transparent Gamma stencil and BS benchmark."""
    points = observations(rows, dimension=dimension, greek=greek,
                          analytical=analytical, deviation=deviation)
    if not points:
        return None
    # Plot measured Greeks only; failed stencils have already been filtered.
    data = pd.DataFrame([{
        "x": p.coordinate, "y": p.plotted, "measured": p.measured,
        "N": p.steps, "gamma_bump": p.gamma_bump_pct,
    } for p in points])
    xlabel = "Spot bump (%)" if dimension == "spot bump" else "Time steps N"
    ylabel = (f"{greek} minus Black–Scholes" if deviation else greek)
    x = alt.X("x:Q", title=xlabel, scale=alt.Scale(zero=False))
    y = alt.Y("y:Q", title=ylabel, scale=alt.Scale(zero=False))
    curve = (alt.Chart(data).mark_line(point=alt.OverlayMarkDef(size=66),
                                       color="#89CDF2", strokeWidth=2.4)
             .encode(x=x, y=y, tooltip=[
                 alt.Tooltip("x:Q", title=xlabel, format=".4f"),
                 alt.Tooltip("measured:Q", title=greek, format=".8f"),
                 alt.Tooltip("gamma_bump:Q", title="Gamma bump (%)", format=".3f"),
                 alt.Tooltip("N:Q", title="Lattice N"),
             ]))
    # Analytical lines are conditional on a valid Black–Scholes comparator.
    if analytical is None or not math.isfinite(analytical):
        return curve.properties(height=285).interactive()
    ref_value = 0.0 if deviation else analytical
    reference = (alt.Chart(pd.DataFrame({"benchmark": [ref_value]}))
                 .mark_rule(strokeDash=[5, 4], color="#EDBB72", opacity=.9)
                 .encode(y="benchmark:Q"))
    return alt.layer(curve, reference).properties(height=285).interactive()


def _fmt(value: float | None, digits: int = 6) -> str:
    return "N/A" if value is None else f"{value:.{digits}g}"


def _render_risk_overview(stats: StabilitySummary, analytic: dict[str, float] | None) -> bool:
    """Show numerical spans and select analytical deviation when available."""
    st.markdown('<div class="desk-section">Observed numerical robustness</div>',
                unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Valid configurations", f"{stats.successful}/{stats.total}")
    b.metric("Delta span across N", _fmt(stats.n_delta_span))
    # Show observed finite differences without assuming convergence.
    c.metric("Gamma span across N", _fmt(stats.n_gamma_span))
    if analytic:
        a, b = st.columns(2)
        a.metric("Max |Delta − BS| across N", _fmt(stats.n_delta_max_bs_gap))
        b.metric("Max |Gamma − BS| across N", _fmt(stats.n_gamma_max_bs_gap))
        mode = st.radio("Risk visualization", ("Observed Greek", "Deviation from Black–Scholes"),
                        horizontal=True, key="qa_risk_numerics_view")
        deviation = mode == "Deviation from Black–Scholes"
    else:
        deviation = False
        st.caption("Black–Scholes comparison is unavailable for this contract. "
                   "Observed variation is not a quantified error against the true value.")
    return deviation


def _render_risk_charts(rows: tuple, analytic: dict[str, float] | None, deviation: bool) -> None:
    """Show bump-width and lattice-depth Greeks without repricing."""
    st.caption("The bump sweep holds N fixed and uses equal Delta/Gamma spot shocks. "
               "The N sweep holds Delta's bump at ±1%; Gamma follows the selected "
               "adaptive or fixed scheme. Orange dashed line = Black–Scholes when valid.")
    # Show observed finite differences without assuming convergence.
    for dimension, label in DIMENSIONS:
        st.markdown(f'<div class="desk-section">{label}</div>', unsafe_allow_html=True)
        left, right = st.columns(2, gap="large")
        for target, greek in zip((left, right), GREEKS):
            with target:
                st.markdown(f"**{greek} · {label}**")
                chart = _risk_chart(rows, dimension, greek,
                                    analytic[greek] if analytic else None, deviation)
                if chart is None:
                    st.info(f"No valid {greek} observations in this sweep.")
                else:
                    st.altair_chart(chart, width="stretch")


def _render_risk_methodology(rows: tuple, stats: StabilitySummary, export_csv: Callable) -> None:
    """Explain numerical limits and export the measured sweeps."""
    if stats.near_zero_n_gamma:
        st.warning(f"Near-zero Gamma at {stats.near_zero_n_gamma} N settings "
                   "despite nonzero analytic Black–Scholes Gamma. "
                   "This may indicate finite-lattice stencil instability.")
    # Show observed finite differences without assuming convergence.
    failed = [r for r in rows if r.status != "ok" or r.delta is None or r.gamma is None]
    if failed:
        st.warning(f"{len(failed)} configurations did not yield complete finite-difference Greeks.")
    with st.expander("Numerical methodology, observations & CSV", expanded=False):
        st.caption("A small observed spread is not proof of convergence. "
                   "In the adaptive scheme Gamma's bump changes with N, so N-only "
                   "effects are not isolated. A fixed bump supports a cleaner N comparison, "
                   "although finite-difference bias remains. "
                   "Analytic Black–Scholes comparison applies only where valid.")
        # Export the original measured sweeps, including failed observations.
        table = pd.DataFrame([{
            "Sweep": p.dimension, "Coordinate": p.coordinate, "N": p.steps,
            "Delta": p.delta, "Gamma": p.gamma,
            "Delta bump (%)": p.spot_bump_pct, "Gamma bump (%)": p.gamma_bump_pct,
            "Status": p.status,
        } for p in rows])
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button("Export risk numerics · CSV", export_csv(rows),
                           file_name="lattice272_risk_numerics.csv", mime="text/csv",
                           width="stretch")


def render_risk_numerics(rows: tuple, request: object, export_csv: Callable) -> None:
    """Two coordinated sweeps, not a duplicate of the trading Greek snapshot."""
    try:
        analytic = analytic_bs_greeks(request)
    except (ValueError, OverflowError, ZeroDivisionError):
        analytic = None
    if not analytic or any(not math.isfinite(analytic[g]) for g in GREEKS):
        analytic = None
    # These charts summarize cached observations; no pricing is triggered here.
    stats = summarize_stability(rows, analytic)
    deviation = _render_risk_overview(stats, analytic)
    _render_risk_charts(rows, analytic, deviation)
    _render_risk_methodology(rows, stats, export_csv)
