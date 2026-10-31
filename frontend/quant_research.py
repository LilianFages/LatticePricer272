"""Three research questions, five on-demand investigations and archived evidence.

This module only orchestrates existing tested studies. It performs no pricing,
changes no financial convention and never loads historical CSVs unless asked.
"""
from __future__ import annotations

import streamlit as st

# One question per top-level tab; no duplicated Pricing Terminal Greek snapshot.
STUDIES = (
    "Numerical reliability",
    "Computational efficiency",
    "Exercise & dividends",
)

# Existing investigations retain their own widget/session keys and results.
RELIABILITY_VIEWS = (
    "Pricing convergence",
    "Greeks robustness",
    "Market scenarios",
)

# Keep exact legacy choice values and their session-state key for safe upgrades.
ARCHIVE_CHOICES = (
    "Data library",
    "Pruning robustness",
    "Accuracy vs cost",
    "Complexity",
    "Temporal exercise",
    "Overview",
)

# A presentation taxonomy, not a dataset migration or new CSV schema.
ARCHIVE_GROUPS = {
    "Numerical reliability": ("Accuracy vs cost",),
    "Computational efficiency": ("Pruning robustness", "Complexity"),
    "Exercise & dividends": ("Temporal exercise",),
    "Research library": ("Overview", "Data library"),
}


def _archive_label(choice: str) -> str:
    """Explain provenance in the selector without changing legacy dispatch keys."""
    for group, choices in ARCHIVE_GROUPS.items():
        if choice in choices:
            return f"{group} · {choice}"
    raise ValueError(f"Unregistered archived study: {choice}")


def _render_numerical_reliability() -> None:
    """Keep model convergence separate from finite-difference sensitivities."""
    st.caption(
        "Research question: where are lattice prices and finite-difference "
        "sensitivities numerically reliable? Analytical Black–Scholes is a "
        "reference only for eligible European contracts without cash dividends."
    )
    investigation = st.segmented_control(
        "Numerical investigation", RELIABILITY_VIEWS,
        selection_mode="single", default=RELIABILITY_VIEWS[0],
        key="quant_research_numerical_view", width="stretch",
    ) or RELIABILITY_VIEWS[0]

    # Each investigation is rendered on demand, without automatic repricing.
    if investigation == "Pricing convergence":
        from frontend.analytics_views import render_convergence_tab
        st.caption(
            "Inspect discrete-step convergence, numerical-reference gaps "
            "and the measured price-versus-computational-cost trade-off. "
            "A finite-N numerical reference is not a known true price."
        )
        render_convergence_tab()
    elif investigation == "Greeks robustness":
        from frontend.analytics_views import render_greeks_stability_tab
        st.caption(
            "Study Delta and Gamma stability across bump widths and lattice "
            "depths. These are finite-difference estimates, not exact derivatives "
            "of a smooth model when exercise or discrete dividends intervene."
        )
        render_greeks_stability_tab()
    else:
        from frontend.analytics_views import render_sensitivity_tab
        st.caption(
            "Supplementary experiment: independently reprice controlled market "
            "scenarios. A finite scenario sweep is not a Greek estimate."
        )
        render_sensitivity_tab()


def _render_computational_efficiency() -> None:
    """Evaluate actual allocation/runtime improvements on comparable contracts."""
    from frontend.research_efficiency_views import render_efficiency_tab

    st.caption(
        "Research question: how much computation does pruning save at a fixed "
        "lattice depth, and what observed price discrepancy accompanies it? "
        "Single-run live timings are not hardware-independent speed rankings."
    )
    render_efficiency_tab()


def _render_exercise_dividends() -> None:
    """Live stopping study or saved ex-date robustness, never both rerun."""
    st.caption(
        "Research question: how do early exercise decisions and cash-dividend "
        "events change the stopping region and measured exercise probabilities? "
        "Node exercise regions are not first-stopping distributions."
    )
    view = st.segmented_control(
        "Exercise investigation",
        ("Optimal stopping · live", "Dividend-grid stability · recorded"),
        selection_mode="single", default="Optimal stopping · live",
        key="quant_research_exercise_view", width="stretch",
    ) or "Optimal stopping · live"
    if view == "Optimal stopping · live":
        from frontend.exercise_boundary_views import render_exercise_boundary_tab
        render_exercise_boundary_tab()
    else:
        from frontend.research_grid_views import render_grid_robustness
        render_grid_robustness()


def _render_archived_evidence() -> None:
    """Display saved experiments only on explicit opt-in, never rerun them."""
    st.divider()
    st.caption(
        "All live studies run only when submitted. These are numerical model "
        "investigations, not market-calibrated executable quotes. "
        "Historical experimental CSVs are versioned, read-only evidence."
    )
    if st.toggle(
        "Browse historical experiment evidence · optional", value=False,
        key="quant_research_archive_visible",
    ):
        from frontend.research_lab import render_research_archive

        st.caption(
            "Select an archived experiment by research question. Recorded "
            "runtimes depend on the original machine and benchmark protocol; "
            "they are not directly comparable to live Streamlit timings."
        )
        choice = st.selectbox(
            "Archived evidence", ARCHIVE_CHOICES,
            format_func=_archive_label,
            key="quant_research_archive_choice",
        )
        render_research_archive(choice)


def render_quant_research_page() -> None:
    """Three focused research sections; keep all prior services and CSV views."""
    st.markdown(
        '<div class="desk-kicker">Quant research / 03</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="desk-title">Quant Research · Interactive Lab</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="desk-subtitle">Numerical reliability · computational '
        'efficiency · exercise & dividends</div>',
        unsafe_allow_html=True,
    )

    reliability, efficiency, stopping = st.tabs(STUDIES)
    with reliability:
        _render_numerical_reliability()
    with efficiency:
        _render_computational_efficiency()
    with stopping:
        _render_exercise_dividends()

    _render_archived_evidence()
