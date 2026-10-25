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


# Lot 4B.1: same-N benchmark ratios, separate from model pricing.
def _speedup_frame(valid: pd.DataFrame) -> pd.DataFrame:
    """Relative median latency at identical N; never divide by zero.

    A missing or ambiguous full-tree baseline makes the ratio unavailable.
    The figure is descriptive: medians do not establish a statistical speedup.
    """
    full = valid.loc[valid["Model"] == FULL, ["N", "Median latency (ms)"]].copy()
    full["Median latency (ms)"] = pd.to_numeric(
        full["Median latency (ms)"], errors="coerce"
    )
    full = full[full["Median latency (ms)"].map(
        lambda x: pd.notna(x) and 0 < x < float("inf")
    )]
    # A duplicated full measurement is ambiguous; use neither one.
    full = full.drop_duplicates(subset=["N"], keep=False)
    full = full.rename(columns={"Median latency (ms)": "Full latency (ms)"})

    pruned = valid.loc[valid["Model"].isin((BRANCH, ALLOCATION))].copy()
    pruned["Median latency (ms)"] = pd.to_numeric(
        pruned["Median latency (ms)"], errors="coerce"
    )
    pruned = pruned[pruned["Median latency (ms)"].map(
        lambda x: pd.notna(x) and 0 < x < float("inf")
    )]
    compared = pruned.merge(full, on="N", how="inner", validate="many_to_one")
    compared["Speed-up (×)"] = (
        compared["Full latency (ms)"] / compared["Median latency (ms)"]
    )
    return compared.loc[compared["Speed-up (×)"].map(
        lambda x: pd.notna(x) and float("-inf") < x < float("inf")
    )].copy()


def _price_gap_label(value: float | None) -> str:
    """Never round a tiny but nonzero same-N price difference to a false zero."""
    if value is None or pd.isna(value):
        return "N/A"
    if value == 0:
        return "0 €"
    if abs(value) < 1e-5:
        return f"{value:.2e} €"
    return f"{value:.6f} €"


# Lot 4B.2 — overlapping allocations and full-tree baseline

def _overlapping_allocation_ns(valid: pd.DataFrame) -> tuple[int, ...]:
    """Exact coincidences for real measured full/branch allocations at equal N."""
    full = valid.loc[valid["Model"] == FULL, ["N", "Allocated nodes"]]
    branch = valid.loc[valid["Model"] == BRANCH, ["N", "Allocated nodes"]]
    if full["N"].duplicated().any() or branch["N"].duplicated().any():
        return ()  # Ambiguous observations are never silently paired.
    matched = full.merge(branch, on="N", how="inner", suffixes=("_full", "_branch"))
    matched = matched.dropna(subset=["Allocated nodes_full", "Allocated nodes_branch"])
    coincident = matched.loc[
        matched["Allocated nodes_full"] == matched["Allocated nodes_branch"], "N"
    ]
    return tuple(sorted(int(n) for n in coincident))


def _allocation_nodes_chart(valid: pd.DataFrame) -> alt.LayerChart:
    """Keep coincident Full/Branch lines visible through width and dash contrast.

    The widest pink Full trace is rendered first; the narrower dashed Branch
    trace on top leaves the pink background visible when counts coincide.
    Individual points are layered with larger Full and smaller Branch marks.
    """
    colors = alt.Scale(
        domain=[FULL, BRANCH, ALLOCATION],
        range=["#F2A0A7", "#287FE2", "#88CDF2"],
    )
    x = alt.X("N:Q", title="N", scale=alt.Scale(zero=False))
    y = alt.Y("Allocated nodes:Q", title="Allocated nodes", scale=alt.Scale(zero=True))
    color = alt.Color(
        "Model:N", scale=colors,
        legend=alt.Legend(
            orient="bottom", direction="vertical", columns=1,
            title=None, labelLimit=300, labelFontSize=11, symbolSize=100,
        ),
    )
    layers = []
    for model, width, dash, point_size in (
        (FULL, 5.1, [1, 0], 135),
        (BRANCH, 2.3, [6, 5], 55),
        (ALLOCATION, 2.3, [1, 0], 70),
    ):
        sample = valid.loc[valid["Model"] == model].sort_values("N")
        if sample.empty:
            continue
        layers.append(
            alt.Chart(sample)
            .mark_line(
                point=alt.OverlayMarkDef(size=point_size),
                strokeWidth=width, strokeDash=dash,
            )
            .encode(
                x=x, y=y, color=color,
                tooltip=[
                    "Model:N", "N:Q",
                    alt.Tooltip("Allocated nodes:Q", format=",.0f"),
                ],
            )
        )
    return alt.layer(*layers).properties(height=EFFICIENCY_CHART_HEIGHT).interactive()


def _speedup_baseline_chart(data: pd.DataFrame) -> alt.LayerChart:
    """Real measured speed-up series with a clearly marked full-tree 1x level."""
    x = alt.X("N:Q", title="N", scale=alt.Scale(zero=False))
    y = alt.Y("Speed-up (×):Q", title="Speed-up (×)", scale=alt.Scale(zero=False))
    anchor = pd.DataFrame({
        "N": [float(data["N"].max())],
        "Full baseline": [1.0],
        "Label": ["Full = 1×"],
    })
    baseline = (
        alt.Chart(anchor).mark_rule(
            color="#F2A0A7", strokeDash=[6, 4], strokeWidth=2, opacity=.95,
        ).encode(
            y=alt.Y("Full baseline:Q", title="Speed-up (×)"),
            tooltip=[alt.Tooltip("Label:N", title="Reference")],
        )
    )
    baseline_label = (
        alt.Chart(anchor).mark_text(
            color="#F2A0A7", fontSize=11, align="right", dx=-5, dy=-9,
        ).encode(
            x=x, y=alt.Y("Full baseline:Q"), text="Label:N",
        )
    )
    pruned = (
        alt.Chart(data).mark_line(
            point=alt.OverlayMarkDef(size=66), strokeWidth=2.3,
        ).encode(
            x=x, y=y,
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(
                    domain=[BRANCH, ALLOCATION],
                    range=["#287FE2", "#88CDF2"],
                ),
                legend=alt.Legend(
                    orient="bottom", direction="vertical", columns=1,
                    title=None, labelLimit=300, labelFontSize=11, symbolSize=100,
                ),
            ),
            tooltip=[
                "Model:N", "N:Q",
                alt.Tooltip("Speed-up (×):Q", format=".3f"),
                alt.Tooltip("Full latency (ms):Q", format=".3f"),
                alt.Tooltip("Median latency (ms):Q", format=".3f"),
            ],
        )
    )
    return alt.layer(baseline, pruned, baseline_label).properties(
        height=EFFICIENCY_CHART_HEIGHT,
    ).interactive()


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
    maximum_gap = valid["Gap to full (€)"].dropna().max()
    if len(optimal) == 1:
        point = optimal.iloc[0]
        a, b, c = st.columns(3)
        a.metric("Allocation saving at largest N", (
            f"{point['Allocation savings (%)']:+.1f}%"
            if pd.notna(point["Allocation savings (%)"]) else "N/A"))
        b.metric("Max same-N price gap", _price_gap_label(maximum_gap))
        c.metric("Median model latency", f"{point['Median latency (ms)']:.1f} ms")
    else:
        st.caption(f"Max same-N price gap across valid scenarios: "
                   f"{_price_gap_label(maximum_gap)}")

    a, b = st.columns(2, gap="large")
    with a:
        st.altair_chart(_allocation_nodes_chart(valid), width="stretch")
        overlap_n = _overlapping_allocation_ns(valid)
        if overlap_n:
            st.caption("Full and Branch pruning allocate the same number of nodes at "
                       f"N={', '.join(map(str, overlap_n))}. "
                       "The thick pink Full trace remains behind the dashed Branch trace.")
    with b:
        st.altair_chart(_chart(valid, "N", "Median latency (ms)"), width="stretch")

    st.markdown('<div class="desk-section">Relative speed & resource use</div>',
                unsafe_allow_html=True)
    speedup = _speedup_frame(valid)
    memory = (valid.dropna(subset=["Peak Python memory (MiB)"])
              if report.measured_peak_memory else valid.iloc[:0])
    if not speedup.empty and not memory.empty:
        a, b = st.columns(2, gap="large")
        with a:
            st.altair_chart(_speedup_baseline_chart(speedup), width="stretch")
        with b:
            st.altair_chart(_chart(memory, "N", "Peak Python memory (MiB)"),
                            width="stretch")
    else:
        # Avoid an empty half-width panel when memory profiling was not chosen.
        if not speedup.empty:
            st.altair_chart(_speedup_baseline_chart(speedup), width="stretch")
        else:
            st.info("Speed-up is unavailable: a positive full-tree latency at the "
                    "same N is required for each comparison.")
        if not memory.empty:
            st.altair_chart(_chart(memory, "N", "Peak Python memory (MiB)"),
                            width="stretch")
        elif not report.measured_peak_memory:
            st.caption("Peak Python memory was not requested; enable profiling "
                       "in the form to add this diagnostic.")

    st.caption("Speed-up = median full-tree latency / median pruned-model latency "
               "at the same N. Above 1× means faster; below 1× means slower. "
               "Small timing differences are sensitive to local CPU noise.")
    st.caption("Actual node allocations come from instrumented lattice builds, "
               "not estimated complexity formulas. Runtime is the median of "
               f"{report.repetitions} independent pricing runs; "
               "memory profiling is measured separately using tracemalloc (not RSS). "
               "Same-N price gaps, retained in the summary and CSV, measure numerical "
               "disagreement with the full tree, not error versus the true value.")
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
