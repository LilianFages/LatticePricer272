"""Quant Research: one interactive workspace, four distinct investigations.

Pricing and all numerical calculations are delegated to existing, tested
services. Historical experiments are strictly optional and never rerun here.
"""
from __future__ import annotations

import streamlit as st

STUDIES = (
    "Numerical accuracy",
    "Computational efficiency",
    "Optimal stopping",
    "Risk numerics",
)

# Only a secondary, opt-in read of existing archived experiments.
ARCHIVE_CHOICES = (
    "Data library",
    "Pruning robustness",
    "Accuracy vs cost",
    "Complexity",
    "Temporal exercise",
    "Overview",
)


def render_quant_research_page() -> None:
    """Four complementary questions, never six overlapping application tabs."""
    st.markdown('<div class="desk-kicker">Quant research / 03</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="desk-title">Quant Research · Interactive Lab</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="desk-subtitle">Numerical accuracy · computational efficiency · '
        'optimal stopping · risk numerics</div>',
        unsafe_allow_html=True,
    )

    accuracy, efficiency, stopping, numerics = st.tabs(STUDIES)

    with accuracy:
        from frontend.analytics_views import render_convergence_tab
        st.caption("How do discretization and model choice affect price accuracy "
                   "and computational cost? Black–Scholes is a reference only "
                   "when its assumptions apply.")
        render_convergence_tab()

    with efficiency:
        from frontend.research_efficiency_views import render_efficiency_tab
        render_efficiency_tab()

    with stopping:
        from frontend.exercise_boundary_views import render_exercise_boundary_tab
        render_exercise_boundary_tab()

    with numerics:
        from frontend.analytics_views import (render_greeks_stability_tab,
                                              render_sensitivity_tab)
        render_greeks_stability_tab()
        # A different question from bump stability: repricing under market moves.
        # Kept available without a fifth top-level tab or duplicate Greek snapshot.
        with st.expander("Additional analysis · Market scenario sweeps", expanded=False):
            st.caption("Reprice the same contract under controlled spot, volatility, "
                       "rate or maturity changes. This is not a Greek estimate.")
            render_sensitivity_tab()

    st.divider()
    st.caption("Live studies run only when submitted. Local numerical models · "
               "not market-calibrated quotes. The saved experiment archive is optional.")

    # toggle rather than expander: the archived CSVs are NOT loaded by default.
    if st.toggle("Browse historical experiment evidence · optional", value=False,
                 key="quant_research_archive_visible"):
        from frontend.research_lab import render_research_archive
        choice = st.selectbox("Archived evidence", ARCHIVE_CHOICES,
                              key="quant_research_archive_choice")
        render_research_archive(choice)
