"""Live and interactive pruning efficiency study; heavy work is opt-in."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from frontend.analytics_views import _contract_fields, _get_request
from frontend.pricing_service import PricingInputError
from frontend.research_efficiency_service import (
    ALLOCATION, BRANCH, FULL, EFFICIENCY_MODELS,
    EfficiencyReport, efficiency_study,
)

N_CHOICES = (10, 20, 30, 40, 60, 80, 100, 125, 150)
THRESHOLDS = (1e-18, 1e-12, 1e-9, 1e-5, 1e-3)


def _form() -> tuple:
    with st.form("qa_efficiency_form"):
        fields = _contract_fields("qa_eff")
        st.markdown('<div class="desk-section">02 · Pruning experiment</div>',
                    unsafe_allow_html=True)
        st.caption("Compare the same full, branch-pruned and allocation-pruned "
                   "trinomial contract at each selected N.")
        depths = st.multiselect("Time-step grid N", N_CHOICES,
                                default=(20, 40, 80, 100), key="qa_eff_ns")
        a, b = st.columns(2)
        threshold = a.selectbox("Pruning threshold", THRESHOLDS, index=0,
                                key="qa_eff_threshold",
                                format_func=lambda x: f"{x:.0e}")
        repeats = b.selectbox("Timing repetitions", (1, 3, 5), index=1,
                              key="qa_eff_repeats",
                              help="Median of normal independent pricing runs.")
        measure_memory = st.checkbox("Measure peak Python memory (slower)", value=False,
                                     key="qa_eff_memory",
                                     help="Separate tracemalloc run, not process RSS. "
                                          "Not included in latency statistics.")
        submitted = st.form_submit_button("RUN PRUNING STUDY  →",
                                          type="primary", width="stretch")
    return fields, depths, threshold, repeats, measure_memory, submitted


def _frame(report: EfficiencyReport) -> pd.DataFrame:
    return pd.DataFrame([{
        "Model": p.model, "N": p.steps, "Status": p.status,
        "Premium (€)": p.price, "Median latency (ms)": p.median_ms,
        "Allocated nodes": p.allocated_nodes,
        "Engine linked nodes": p.linked_nodes,
        "Peak Python memory (MiB)": p.peak_python_mib,
        "Gap to full (€)": p.absolute_gap_to_full,
        "Allocation savings (%)": p.allocation_savings_pct,
    } for p in report.points])


# Keep the plots legible without enlarging the form or creating new tabs.
EFFICIENCY_CHART_HEIGHT = 305


def _chart(
    data: pd.DataFrame, x: str, y: str, *,
    height: int = EFFICIENCY_CHART_HEIGHT,
) -> alt.Chart:
    """Interactive pricing diagnostics with a compact bottom legend."""
    return (
        alt.Chart(data)
        .mark_line(point=alt.OverlayMarkDef(size=66), strokeWidth=2.3)
        .encode(
            x=alt.X(f"{x}:Q", title=x),
            y=alt.Y(f"{y}:Q", title=y, scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Model:N",
                legend=alt.Legend(
                    orient="bottom", direction="horizontal", columns=2,
                    title=None, labelLimit=240, labelFontSize=11,
                    symbolSize=85, offset=8,
                ),
            ),
            tooltip=["Model:N", "N:Q", alt.Tooltip(f"{y}:Q", format=".5f")],
        )
        .properties(height=height)
        .interactive()
    )


def _results(report: EfficiencyReport) -> None:
    frame = _frame(report)
    valid = frame[frame["Status"] == "ok"].copy()
    st.markdown('<div class="desk-section">Allocation & runtime scaling</div>',
                unsafe_allow_html=True)
    if len(valid) != len(frame):
        with st.expander(f"{len(frame)-len(valid)} unavailable configuration(s)",
                         expanded=True):
            st.dataframe(frame[frame["Status"] != "ok"][["Model", "N", "Status"]],
                         hide_index=True, width="stretch")
    if valid.empty:
        st.error("No model completed. Review the contract and dividend schedule.")
        return
    latest_n = int(valid["N"].max())
    optimal = valid[(valid["N"] == latest_n) & (valid["Model"] == ALLOCATION)]
    if len(optimal) == 1:
        point = optimal.iloc[0]
        a, b, c = st.columns(3)
        a.metric("Allocation saving at largest N", (
            f"{point['Allocation savings (%)']:+.1f}%"
            if pd.notna(point["Allocation savings (%)"]) else "N/A"))
        b.metric("Same-N price gap", (
            f"{point['Gap to full (€)']:.6f} €"
            if pd.notna(point["Gap to full (€)"]) else "N/A"))
        c.metric("Median model latency", f"{point['Median latency (ms)']:.1f} ms")
    a, b = st.columns(2, gap="large")
    with a:
        st.altair_chart(_chart(valid, "N", "Allocated nodes"), width="stretch")
    with b:
        st.altair_chart(_chart(valid, "N", "Median latency (ms)"), width="stretch")
    st.markdown('<div class="desk-section">Price preservation & resource use</div>',
                unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        available = valid.dropna(subset=["Gap to full (€)"])
        if not available.empty:
            st.altair_chart(_chart(available, "N", "Gap to full (€)"), width="stretch")
    with b:
        if report.measured_peak_memory:
            measured = valid.dropna(subset=["Peak Python memory (MiB)"])
            if not measured.empty:
                st.altair_chart(_chart(measured, "N", "Peak Python memory (MiB)"),
                                width="stretch")
        else:
            st.info("Enable peak Python memory in the form to compare memory footprints. "
                    "Measurement requires a separate, slower profiling run.")
    st.caption("Exact Node allocations come from instrumented lattice builds, "
               "not estimated complexity formulas. Runtime is the median of "
               f"{report.repetitions} normal, independent pricing runs; "
               "profiling is kept outside runtime measurements. "
               "Same-N full-trinomial gaps measure numerical disagreement, "
               "not error versus the true option value. These are local CPU measurements.")
    with st.expander("Detailed observations & CSV", expanded=False):
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button("Export efficiency · CSV", frame.to_csv(index=False),
                           file_name="lattice272_efficiency.csv", mime="text/csv",
                           width="stretch")


def render_efficiency_tab() -> None:
    st.markdown("**Pruning Efficiency Lab**")
    st.caption("Explore the computational savings of branch and allocation "
               "pruning without mixing numerical accuracy and cost claims.")
    left, right = st.columns([1.0, 1.85], gap="large")
    with left:
        fields, ns, threshold, repeats, memory, submitted = _form()
    with right:
        if submitted:
            st.session_state.pop("qa_efficiency_result", None)
            try:
                request = _get_request(fields, ALLOCATION,
                                       int(ns[0]) if ns else 20, float(threshold))
                with st.spinner("Pricing and counting actual lattice allocations..."):
                    st.session_state["qa_efficiency_result"] = efficiency_study(
                        request, tuple(int(n) for n in ns),
                        repetitions=int(repeats), measure_memory=bool(memory),
                    )
            except (PricingInputError, ValueError, RuntimeError, ArithmeticError,
                    OverflowError) as exc:
                st.error(f"Efficiency study unavailable: {exc}")
        report = st.session_state.get("qa_efficiency_result")
        if isinstance(report, EfficiencyReport):
            _results(report)
        elif not submitted:
            st.info("Choose the N grid and pruning threshold, then run the study. "
                    "No models are priced when adjusting display controls.")
