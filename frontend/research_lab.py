"""05 / Research Lab — interactive explorer of completed reproducible experiments.

No engines run automatically. All figures use repository CSV exports and expose
filters / downloadable rows. The source dataset is documented on each tab.
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from frontend.research_lab_service import (
    CATALOG, ROOT, DatasetError, as_csv, comparable_cases, coverage_summary,
    frontier_thresholds, load_dataset, mass_consistency, missingness_by_column,
    missingness_diagnostics, observed_pareto, robustness_by_step,
)

BLUE = "#79BCE8"
GOLD = "#E0AE74"
PALETTE = ["#79BCE8", "#C49BEB", "#E0AE74", "#7CC6B1", "#D78591"]
# Use recorded experiment exports without starting new pricing runs.
MODEL_NAMES = {
    "hybrid": "Hybrid",
    "trinomial_no_pruning": "Trinomial · Full",
    "trinomial_branch_pruning": "Trinomial · Branch",
    "trinomial_allocation_pruning": "Trinomial · Allocation",
}
SCENARIO_NAMES = {
    "american_cash_dividend": "American put · Cash dividend",
    "american_no_dividend": "American put · No dividend",
    "european_no_dividend": "European put · No dividend",
}
RESOURCE_NAMES = {
    "total_seconds": "Total runtime (seconds)",
    "build_seconds": "Build time (seconds)",
    "pricing_seconds": "Pricing time (seconds)",
    "allocated_nodes": "Allocated nodes",
    # Maintain the same section labels, tab order and filter widget keys.
    "peak_python_mib": "Peak Python memory (MiB)",
}
LIBRARY_COLUMNS = {
    "Robustness": ("scenario", "nb_steps", "strike", "kind", "exercise", "status",
                   "branch_status", "allocation_status", "absolute_gap", "node_savings_pct"),
    "Pruning benchmark": ("nb_steps", "model", "status", "total_seconds",
                          "allocated_nodes", "peak_python_mib"),
    "Accuracy/cost": ("model", "nb_steps", "worst_absolute_error",
                      "max_median_runtime_seconds", "allocated_nodes_per_tree"),
    "Temporal profiles": ("scenario", "time", "discounted_mean_option_value",
                          "exercise_region_probability", "first_exercise_probability"),
}



@st.cache_data(show_spinner=False)
def _cached_csv(name: str, last_modified_ns: int) -> pd.DataFrame:
    # The version argument invalidates the cache if an experiment is rerun.
    return load_dataset(name)


def _data(name: str) -> pd.DataFrame | None:
    location = ROOT / CATALOG[name].path
    try:
        return _cached_csv(name, location.stat().st_mtime_ns if location.is_file() else -1)
    except DatasetError as exc:
        st.info(f"{exc} — Run the corresponding experiment, then refresh this page.")
    return None


def _chart(chart: alt.Chart) -> None:
    st.altair_chart(chart, width="stretch")


def _download(frame: pd.DataFrame, name: str, label: str = "Export filtered data · CSV") -> None:
    # Distinguish observed results from a theoretical frontier.
    st.download_button(label, as_csv(frame), file_name=f"research_lab_{name}.csv",
                       mime="text/csv", width="stretch", key=f"research_{name}_download")


def _source(name: str) -> None:
    spec = CATALOG[name]
    st.caption(f"Source: `{spec.path}` · {spec.description}")


def _timeseries(
    data: pd.DataFrame, x: str, y: str, group: str, title: str,
    x_title: str, y_title: str, logx: bool = False, logy: bool = False,
    markers: bool = True, tight_y: bool = False, legend_bottom: bool = False,
) -> alt.Chart:
    axis_x = (alt.X(f"{x}:Q", title=x_title, scale=alt.Scale(type="log"))
              if logx else alt.X(f"{x}:Q", title=x_title))
    # Preserve downloaded CSV schemas and chart legends.
    axis_y = (alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(type="log"))
              if logy else alt.Y(f"{y}:Q", title=y_title,
                                 scale=alt.Scale(zero=False)) if tight_y
              else alt.Y(f"{y}:Q", title=y_title))
    selection = alt.selection_point(fields=[group], bind="legend")
    legend = alt.Legend(title=None if legend_bottom else "Series",
                        orient="bottom" if legend_bottom else "right", labelLimit=230)
    chart = alt.Chart(data).mark_line(point=markers, strokeWidth=2).encode(
        x=axis_x, y=axis_y,
        color=alt.Color(f"{group}:N", legend=legend,
                        scale=alt.Scale(range=PALETTE)),
        opacity=alt.condition(selection, alt.value(1), alt.value(.17)),
        tooltip=[alt.Tooltip(f"{x}:Q"), alt.Tooltip(f"{y}:Q", format=".6g"),
                 alt.Tooltip(f"{group}:N")],
    ).add_params(selection)
    return chart.properties(height=330, title=title).interactive()


# Separate recorded-data KPIs from the underlying experiment catalogue.
def _render_research_pareto(right: object, cost: pd.DataFrame | None) -> None:
    with right:
        if cost is not None:
            frontier = observed_pareto(cost)
            # Maintain the same section labels, tab order and filter widget keys.
            frontier = frontier[(frontier["max_median_runtime_seconds"] > 0)
                                & (frontier["worst_absolute_error"] > 0)].copy()
            if not frontier.empty:
                frontier["runtime_ms"] = frontier["max_median_runtime_seconds"] * 1000
                _chart(alt.Chart(frontier).mark_line(point=True,
                                                    color=GOLD, strokeWidth=2.5).encode(
                    x=alt.X("runtime_ms:Q", title="Worst median runtime (ms)",
                            scale=alt.Scale(type="log")),
                    y=alt.Y("worst_absolute_error:Q", title="Worst-strike error (€)",
                            scale=alt.Scale(type="log")),
                    tooltip=["model:N", "nb_steps:Q", "runtime_ms:Q", "worst_absolute_error:Q"],
                ).properties(title="Observed Pareto trade-off", height=330))


def _render_research_pruning_chart(left: object, robust: pd.DataFrame | None) -> None:
    with left:
        if robust is not None:
            # Use recorded experiment exports without starting new pricing runs.
            summary = robustness_by_step(robust)
            if not summary.empty:
                summary = summary.copy()
                summary["scenario_label"] = summary["scenario"].map({
                    "no_dividend": "No dividend", "single_dividend": "One dividend",
                    "two_dividends": "Two dividends",
                }).fillna(summary["scenario"])
                _chart(_timeseries(summary, "nb_steps", "mean_savings_pct", "scenario_label",
                                   "Allocation savings on comparable cases", "Steps N", "Saved allocations (%)",
                                   legend_bottom=True))


def _render_research_overview_charts(robust: pd.DataFrame | None, cost: pd.DataFrame | None) -> None:
    left, right = st.columns(2, gap="large")
    _render_research_pruning_chart(left, robust)
    _render_research_pareto(right, cost)
    st.caption("Both highlights use recorded comparable cases and the sampled error/cost grid. "
               "They are not cross-machine speed guarantees or universal optimality claims.")
    st.markdown('<div class="desk-section">Research library</div>', unsafe_allow_html=True)
    st.markdown(
        # Distinguish observed results from a theoretical frontier.
        "**Complexity & pruning** — Node counts, runtime, peak Python memory, "
        "truncated branches and robustness across dividend schedules.\n\n"
        "**Accuracy vs cost** — Error against Black–Scholes across three observed strikes, "
        "empirical Pareto frontiers and achieved target tolerances.\n\n"
        "**Temporal stopping** — Discounted expected option values, exercise-region mass, "
        "and first-exercise distributions, including stability across lattice depths."
    )
    st.caption(
        "All results are local snapshots of prior experiments, not live pricing or market quotes. "
        "Runtime comparisons are machine- and run-dependent; mixed collection sessions are not a "
        "controlled benchmark. Use Model Comparison or Quant Analytics for fresh contract calculations."
    )


def _render_research_highlights(robust: pd.DataFrame | None, cost: pd.DataFrame | None) -> None:
    st.markdown('<div class="desk-section">Experimental evidence · Highlights</div>', unsafe_allow_html=True)
    comparable = comparable_cases(robust) if robust is not None else pd.DataFrame()
    # Maintain the same section labels, tab order and filter widget keys.
    a, b, c = st.columns(3)
    if not comparable.empty:
        gaps = pd.to_numeric(comparable["absolute_gap"], errors="coerce")
        savings = pd.to_numeric(comparable["node_savings_pct"], errors="coerce")
        a.metric("Max comparable price gap", f"{gaps.max():.3g} €" if gaps.notna().any() else "—")
        b.metric("Mean allocated-node savings", f"{savings.mean():.1f}%" if savings.notna().any() else "—")
    else:
        a.metric("Max comparable price gap", "—")
        b.metric("Mean allocated-node savings", "—")
    if cost is not None:
        errors = pd.to_numeric(cost["worst_absolute_error"], errors="coerce")
        c.metric("Best recorded worst-strike error", f"{errors.min():.5g} €" if errors.notna().any() else "—")
    else:
        c.metric("Best recorded worst-strike error", "—")


def _overview() -> None:
    # Separate recorded-data KPIs from the underlying experiment catalogue.
    st.markdown('<div class="desk-section">Research coverage · Recorded runs</div>', unsafe_allow_html=True)
    robust = _data("Robustness")
    cost = _data("Accuracy/cost")
    temporal = _data("Temporal sensitivity")
    a, b, c = st.columns(3)
    a.metric("Robustness cases", f"{len(robust):,}" if robust is not None else "—")
    b.metric("Model × discretization points", f"{len(cost):,}" if cost is not None else "—")
    c.metric("Temporal scenarios", f"{len(temporal):,}" if temporal is not None else "—")
    if robust is not None:
        coverage = coverage_summary(robust)
        st.caption(
            f"Robustness: {coverage['compared']} genuine price comparisons, "
            f"{coverage['allocation_only']} allocation-only successes, "
            f"{coverage['failed']} remaining outcomes. Allocation-only runs are NOT price-gap validations."
        )

    # Separate view-specific data preparation from chart display.
    _render_research_highlights(robust, cost)

    # Use recorded experiment exports without starting new pricing runs.
    _render_research_overview_charts(robust, cost)


def _render_robustness_comparisons(filtered: pd.DataFrame, stats: dict) -> None:
    stats_data = pd.DataFrame({"status": ["Compared", "Allocation only", "Other"],
                              "cases": [stats["compared"], stats["allocation_only"], stats["failed"]]})
    _chart(alt.Chart(stats_data).mark_bar(size=24).encode(
        y=alt.Y("status:N", title=None, sort=["Compared", "Allocation only", "Other"],
                axis=alt.Axis(labelAngle=0, labelLimit=170)),
        x=alt.X("cases:Q", title="Number of cases", axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("status:N", scale=alt.Scale(range=PALETTE), legend=None),
        tooltip=["status:N", "cases:Q"],
    ).properties(height=135))
    comp = comparable_cases(filtered)
    if not comp.empty:
        gap = pd.to_numeric(comp["absolute_gap"], errors="coerce")
        a, b = st.columns(2)
        # Separate view-specific data preparation from chart display.
        a.metric("Max observed comparable price gap", f"{gap.max():.3g} €" if gap.notna().any() else "—")
        savings = pd.to_numeric(comp["node_savings_pct"], errors="coerce")
        b.metric("Mean measured node savings", f"{savings.mean():.1f}%" if savings.notna().any() else "—")
        summary = robustness_by_step(filtered)
        if not summary.empty:
            _chart(_timeseries(summary, "nb_steps", "mean_savings_pct", "scenario",
                               "Allocation reduction on comparable cases only",
                               "Time steps N", "Allocated-node savings (%)"))
    st.caption("A baseline failure cannot establish a pricing gap. Missing gaps are never replaced by zero.")
    st.dataframe(filtered[["scenario", "nb_steps", "strike", "kind", "exercise", "status",
                           "branch_status", "allocation_status", "absolute_gap", "node_savings_pct"]],
                 hide_index=True, width="stretch")
    _download(filtered, "robustness")
    _source("Robustness")


def _render_pruning_kpis(stats: dict) -> None:
    a, b, c, d = st.columns(4)
    a.metric("Cases selected", stats["total"])
    b.metric("Compared prices", stats["compared"])
    c.metric("Allocation-only", stats["allocation_only"])
    # Keep missing datasets visible rather than reporting empty success.
    d.metric("Other outcomes", stats["failed"])


def _pruning() -> None:
    st.markdown('<div class="desk-section">Robustness · Two distinct outcomes</div>', unsafe_allow_html=True)
    frame = _data("Robustness")
    # Display pruning comparisons only for outcomes with a valid reference.
    if frame is None:
        return
    scenario_choices = sorted(frame["scenario"].dropna().unique().tolist())
    scenario = st.multiselect("Dividend scenarios", scenario_choices, default=scenario_choices,
                              key="lab_rob_scenarios")
    n_options = sorted(frame["nb_steps"].dropna().astype(int).unique().tolist())
    ns = st.multiselect("Time steps", n_options, default=n_options, key="lab_rob_steps")
    option_types = sorted(frame["kind"].dropna().unique().tolist())
    kinds = st.multiselect("Option kinds", option_types, default=option_types,
                           key="lab_rob_kinds")
    filtered = frame[frame["scenario"].isin(scenario) & frame["nb_steps"].isin(ns)
                     & frame["kind"].isin(kinds)].copy()
    stats = coverage_summary(filtered)
    _render_pruning_kpis(stats)
    if filtered.empty:
        st.warning("No cases match these filters.")
        return
    # Display pruning comparisons only for outcomes with a valid reference.
    _render_robustness_comparisons(filtered, stats)


def _render_accuracy_target(filtered: pd.DataFrame, frontier: pd.DataFrame) -> None:
    tolerance = st.select_slider("Worst-strike error target (€)",
                                 options=[0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
                                 value=0.005, key="lab_target")
    targets = frontier_thresholds(filtered, (float(tolerance),))
    best = targets.iloc[0]
    if best["status"] == "reached":
        st.success(f"Fastest selected configuration achieving ≤{tolerance:g} €: "
                   f"{best['model']}, N={int(best['nb_steps'])}, "
                   f"{best['runtime_ms']:.2f} ms per valuation, "
                   f"worst-strike error {best['error_eur']:.6g} €.")
    else:
        st.info("No configuration in the selected dataset meets this target.")
    a, b = st.columns(2)
    with a:
        st.markdown("**Filtered observations**")
        st.dataframe(filtered[["model", "nb_steps", "worst_absolute_error",
                               "max_median_runtime_seconds", "allocated_nodes_per_tree"]],
                     # Keep missing datasets visible rather than reporting empty success.
                     hide_index=True, width="stretch", height=300)
    with b:
        st.markdown("**Observed selected-model frontier**")
        st.dataframe(frontier[["model", "nb_steps", "worst_absolute_error",
                                "max_median_runtime_seconds"]], hide_index=True,
                     width="stretch", height=300)
    _download(filtered, "accuracy_cost")
    _source("Accuracy/cost")


def _render_accuracy_scatter(positive: pd.DataFrame, frontier: pd.DataFrame) -> None:
    if not positive.empty:
        scatter = alt.Chart(positive).mark_circle(size=90, opacity=.8).encode(
            x=alt.X("runtime_ms:Q", title="Worst median runtime (ms / valuation)", scale=alt.Scale(type="log")),
            y=alt.Y("worst_error_eur:Q", title="Worst absolute error across K=80,100,120 (€)",
                    scale=alt.Scale(type="log")),
            color=alt.Color("model:N", scale=alt.Scale(range=PALETTE), title="Architecture"),
            tooltip=["model:N", "nb_steps:Q", alt.Tooltip("runtime_ms:Q", format=".3f"),
                     alt.Tooltip("worst_error_eur:Q", format=".6g")],
        )
        line = frontier.copy()
        line["runtime_ms"] = line["max_median_runtime_seconds"] * 1000
        line = line[(line["runtime_ms"] > 0) & (line["worst_absolute_error"] > 0)]
        boundary_axes = {
            "x": alt.X("runtime_ms:Q", title="Worst median runtime (ms / valuation)",
                       scale=alt.Scale(type="log")),
            "y": alt.Y("worst_absolute_error:Q", title="Worst absolute error across K=80,100,120 (€)",
                       # Distinguish observed results from a theoretical frontier.
                       scale=alt.Scale(type="log")),
        }
        boundary = alt.Chart(line).mark_line(strokeDash=[5, 4], strokeWidth=2.7,
                                                  color=GOLD).encode(**boundary_axes)
        highlights = alt.Chart(line).mark_point(filled=True, size=170, color=GOLD,
                                                stroke="#0C1623", strokeWidth=1.5).encode(
            **boundary_axes,
            tooltip=["model:N", "nb_steps:Q",
                     alt.Tooltip("runtime_ms:Q", format=".3f"),
                     alt.Tooltip("worst_absolute_error:Q", format=".6g")],
        )
        _chart((scatter + boundary + highlights).properties(height=465,
                title="Observed error–cost trade-off · Log–log · Gold: selected-model Pareto points").interactive())
    st.caption(
        "The frontier is computed from the SELECTED measured points. Log plots exclude zero errors "
        "(zero-valued observations remain in the CSV and the frontier calculation). "
        "The dataset includes measurements from distinct sessions; only the sampled grid is covered."
    )


def _accuracy() -> None:
    # Keep the measured trade-off axes and filters synchronized.
    st.markdown('<div class="desk-section">Precision · Computational cost</div>', unsafe_allow_html=True)
    frame = _data("Accuracy/cost")
    if frame is None:
        return
    models = sorted(frame["model"].dropna().unique().tolist())
    selected = st.multiselect("Architectures", models, default=models, key="lab_acc_models")
    filtered = frame[frame["model"].isin(selected)].copy()
    if filtered.empty:
        st.warning("Select at least one architecture.")
        return
    filtered["runtime_ms"] = pd.to_numeric(filtered["max_median_runtime_seconds"], errors="coerce") * 1000
    filtered["worst_error_eur"] = pd.to_numeric(filtered["worst_absolute_error"], errors="coerce")
    positive = filtered[(filtered["runtime_ms"] > 0) & (filtered["worst_error_eur"] > 0)].copy()
    frontier = observed_pareto(filtered)
    c1, c2, c3 = st.columns(3)
    c1.metric("Observed points", len(filtered))
    c2.metric("Non-dominated points", len(frontier))
    # Maintain the same section labels, tab order and filter widget keys.
    c3.metric("Min recorded worst-strike error", f"{filtered['worst_error_eur'].min():.5f} €")
    _render_accuracy_scatter(positive, frontier)
    # Preserve downloaded CSV schemas and chart legends.
    _render_accuracy_target(filtered, frontier)


def _render_recursive_scaling(runtime: pd.DataFrame | None) -> None:
    if runtime is not None:
        melted = runtime.melt(id_vars="nb_steps", value_vars=["recursive_seconds", "backward_seconds"],
                              var_name="strategy", value_name="seconds")
        melted = melted[(melted["seconds"] > 0) & (melted["nb_steps"] > 0)].copy()
        melted["strategy"] = melted["strategy"].map({
            "recursive_seconds": "Recursive", "backward_seconds": "Backward induction",
        })
        if not melted.empty:
            _chart(_timeseries(melted, "nb_steps", "seconds", "strategy", "Measured runtime",
                               "Steps N", "Seconds", logx=True, logy=True))
        _source("Runtime scaling")


def _render_pruning_scaling(frame: pd.DataFrame | None) -> None:
    if frame is not None:
        good = frame[frame["status"].eq("success")].copy()
        labels = sorted(good["model"].dropna().unique().tolist())
        selected = st.multiselect("Models", labels, default=labels, key="lab_scale_models")
        # Keep runtime scaling summaries connected to the archived sample.
        good = good[good["model"].isin(selected)].copy()
        if not good.empty:
            metric = st.selectbox("Resource", list(RESOURCE_NAMES), key="lab_scale_metric",
                                  format_func=lambda value: RESOURCE_NAMES[value])
            good[metric] = pd.to_numeric(good[metric], errors="coerce")
            good = good[(good[metric] > 0) & (good["nb_steps"] > 0)]
            if not good.empty:
                labelled = good.copy()
                labelled["model_label"] = labelled["model"].map(MODEL_NAMES).fillna(labelled["model"])
                _chart(_timeseries(labelled, "nb_steps", metric, "model_label",
                                   "Observed resource scaling (log–log)", "Steps N",
                                   RESOURCE_NAMES[metric], logx=True, logy=True, legend_bottom=True))
            st.dataframe(good[["nb_steps", "model", metric, "allocated_nodes"]].loc[:,
                               lambda x: ~x.columns.duplicated()], hide_index=True, width="stretch")
            _download(good, "pruning_scaling")
        _source("Pruning benchmark")


def _scaling() -> None:
    st.markdown('<div class="desk-section">Algorithmic scaling · Allocations and time</div>', unsafe_allow_html=True)
    frame = _data("Pruning benchmark")
    _render_pruning_scaling(frame)
    st.markdown('<div class="desk-section">Recursive versus backward traversal</div>', unsafe_allow_html=True)
    # Use recorded experiment exports without starting new pricing runs.
    runtime = _data("Runtime scaling")
    _render_recursive_scaling(runtime)
    st.caption("The timing exports reflect their original machines and run conditions; "
               "they are not standardized latency guarantees.")


def _render_temporal_stability(shown: pd.DataFrame, selected: list) -> None:
    with st.expander("Stability across N=50, 100, 200", expanded=True):
        sensitivity = _data("Temporal sensitivity")
        if sensitivity is not None:
            sensitivity = sensitivity[sensitivity["scenario"].isin(selected)].copy()
            sensitivity["scenario_label"] = sensitivity["scenario"].map(SCENARIO_NAMES).fillna(sensitivity["scenario"])
            if not sensitivity.empty:
                left, right = st.columns(2)
                with left:
                    _chart(_timeseries(sensitivity, "nb_steps", "initial_option_price", "scenario_label",
                                       "Premium stability", "Steps N", "Premium (€)",
                                       tight_y=True, legend_bottom=True))
                with right:
                    sensitivity["early_pct"] = sensitivity["early_exercise_probability"] * 100
                    _chart(_timeseries(sensitivity, "nb_steps", "early_pct", "scenario_label",
                                       "Early exercise probability", "Steps N", "Probability (%)",
                                       # Keep missing datasets visible rather than reporting empty success.
                                       legend_bottom=True))
                _download(sensitivity, "temporal_stability")


def _render_temporal_studies(shown: pd.DataFrame, selected: list) -> None:
    vtab, region, first = st.tabs(["Discounted option values", "Exercise-region mass", "First-exercise mass"])
    # Distinguish observed results from a theoretical frontier.
    with vtab:
        _chart(_timeseries(shown, "time", "discounted_mean_option_value", "scenario_label",
                           "Discounted average option value", "Years", "Discounted value (€)",
                           tight_y=True, legend_bottom=True))
        st.caption("Non-zero-based vertical axis deliberately magnifies small variations. "
                   "European expected discounted values are martingales; "
                   "for American claims the Snell envelope is a supermartingale.")
    with region:
        shown["region_pct"] = shown["exercise_region_probability"] * 100
        _chart(_timeseries(shown, "time", "region_pct", "scenario_label",
                           "Unconditional mass in exercise region", "Years", "Mass (%)",
                           legend_bottom=True))
        st.caption("These are risk-neutral UNSTOPPED reach probabilities. Do NOT sum over dates.")
    with first:
        shown["first_pct"] = shown["first_exercise_probability"] * 100
        _chart(_timeseries(shown, "time", "first_pct", "scenario_label",
                           "First-exercise distribution (not cumulatively summed)",
                           # Preserve downloaded CSV schemas and chart legends.
                           "Years", "First exercise at date (%)", legend_bottom=True))
        st.caption("Stopped-path mass: each path contributes at most once; maturity includes only ITM exercise.")
    _render_temporal_stability(shown, selected)
    _download(shown, "temporal_profiles")
    _source("Temporal profiles")
    st.caption("The original experiment equalizes forwards across scenarios by adjusting no-dividend "
               "starting spots; it does not impose equal initial spots. Cash dividends are discrete.")


def _temporal() -> None:
    # Separate value profiles, exercise mass and stopping distributions.
    st.markdown('<div class="desk-section">Exercise decisions · Time profiles</div>', unsafe_allow_html=True)
    frame = _data("Temporal profiles")
    if frame is None:
        return
    scenarios = sorted(frame["scenario"].dropna().unique().tolist())
    selected = st.multiselect("Comparison scenarios", scenarios, default=scenarios,
                              key="lab_temp_scenarios")
    shown = frame[frame["scenario"].isin(selected)].copy()
    if shown.empty:
        st.warning("Select a scenario.")
        return
    shown["scenario_label"] = shown["scenario"].map(SCENARIO_NAMES).fillna(shown["scenario"])
    a, b, c = st.columns(3)
    a.metric("Dates per scenario", int(shown.groupby("scenario")["time"].nunique().max()))
    b.metric("Maximum mass error", f"{mass_consistency(shown):.2g}")
    c.metric("Initial scenarios", int(shown["scenario"].nunique()))
    # Separate value profiles, exercise mass and stopping distributions.
    _render_temporal_studies(shown, selected)


def _render_research_library_status(name: str, frame: pd.DataFrame, missing: dict) -> None:
    if name == "Robustness":
        st.caption(
            f"Of {missing['total']:,} empty fields: "
            f"{missing['empty_error_fields']:,} are blank error-message fields, "
            f"{missing['unavailable_comparisons']:,} are unavailable baseline/comparison "
            f"measurements from allocation-only runs, and "
            f"{missing['unclassified']:,} are not explained by those two categories. "
            "No missing comparison is interpreted as zero."
        )
    else:
        st.caption("Empty cells are preserved from the original dataset. "
                   "Inspect their meaning before comparing or aggregating observations.")


def _render_research_library_data(frame: pd.DataFrame, name: str, missing: dict) -> None:
    _render_research_library_status(name, frame, missing)
    # Use recorded experiment exports without starting new pricing runs.
    if missing["total"]:
        with st.expander("Missing-cell details · by column", expanded=False):
            st.dataframe(missingness_by_column(frame), width="stretch", hide_index=True)
    suggested = [col for col in LIBRARY_COLUMNS.get(name, tuple(frame.columns)) if col in frame.columns]
    columns = st.multiselect("Columns to display", list(frame.columns),
                             default=suggested, key=f"lab_columns_{name}")
    if not columns:
        st.info("Select at least one column for the preview.")
        _download(frame, f"catalog_all_{name.lower().replace(' ', '_')}",
                  "Export all source columns · CSV")
        return
    preview = frame[columns].copy()
    st.dataframe(preview, hide_index=True, width="stretch", height=405)
    left, right = st.columns(2)
    with left:
        _download(preview, f"catalog_selected_{name.lower().replace(' ', '_')}",
                  "Export selected columns · CSV")
    # Maintain the same section labels, tab order and filter widget keys.
    with right:
        _download(frame, f"catalog_all_{name.lower().replace(' ', '_')}",
                  "Export all source columns · CSV")
    st.caption("Raw files are versioned in GitHub and loaded locally. "
               "No automated background experiment or network request is triggered by this page.")


def _library() -> None:
    st.markdown('<div class="desk-section">Provenance · Inspect and export</div>', unsafe_allow_html=True)
    names = list(CATALOG.keys())
    name = st.selectbox("Saved dataset", names, key="lab_library_choice")
    frame = _data(name)
    if frame is None:
        return
    missing = missingness_diagnostics(name, frame)
    a, b, c = st.columns(3)
    # Support repeatable filtering and unmodified CSV downloads.
    a.metric("Rows", len(frame))
    b.metric("Columns", len(frame.columns))
    c.metric("Empty CSV cells", f"{missing['total']:,}",
             help="Blank source fields are preserved as missing, not replaced with zero.")
    _source(name)
    _render_research_library_data(frame, name, missing)


def render_research_lab_page() -> None:
    st.markdown('<div class="desk-kicker">Research lab / 05</div>', unsafe_allow_html=True)
    st.markdown('<div class="desk-title">Research Lab · Empirical Evidence</div>', unsafe_allow_html=True)
    st.markdown('<div class="desk-subtitle">Saved experimental results · reproducible filters · '
                'explicit numerical limitations</div>', unsafe_allow_html=True)
    overview, pruning, accuracy, scaling, temporal, library = st.tabs([
        "Overview", "Pruning robustness", "Accuracy vs cost", "Complexity",
        "Temporal exercise", "Data library",
    ])
    with overview:
        # Keep research tabs in the established order and avoid recomputation.
        _overview()
    with pruning:
        _pruning()
    with accuracy:
        _accuracy()
    with scaling:
        _scaling()
    with temporal:
        _temporal()
    with library:
        _library()
