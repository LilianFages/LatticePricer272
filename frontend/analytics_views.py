"""Streamlit Quant Analytics panels for sensitivity, convergence, Greek stability.

All heavy computation is button-triggered. These panels do not modify src/.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from frontend.research_accuracy_service import AccuracyCostPoint

from frontend.analytics_service import (
    ConvergenceResult, PricingObservation, StabilityObservation,
    convergence_study, greeks_stability, observations_csv,
    sensitivity_sweep, stability_csv,
)
from frontend.pricing_service import (
    DEFAULT_THRESHOLD, MAX_STEPS, MODELS, PricingInputError, PricingRequest,
)


# Preserve existing internal call sites while moving form ownership out of this view.
from frontend.research_contract import (
    build_research_request as _get_request,
    render_research_contract_fields as _contract_fields,
)




def _model_select(key: str, *, default: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(st.multiselect("Models to compare", MODELS, default=default, key=key))


def _threshold(key: str) -> float:
    return st.selectbox("Pruning threshold", (0.0, 1e-18, 1e-15, 1e-12, 1e-5),
                        index=1, key=key,
                        # Keep chart axes, tooltips and color grouping consistent.
                        format_func=lambda x: "off" if not x else f"{x:.0e}")


def _line_chart(
    data: pd.DataFrame, *, x: str, y: str, group: str,
    x_label: str, y_label: str, log_y: bool = False, height: int = 310,
) -> alt.Chart:
    scale = alt.Scale(type="log", zero=False) if log_y else alt.Scale(zero=False)
    return (alt.Chart(data).mark_line(point=alt.OverlayMarkDef(size=55), strokeWidth=2.2)
            .encode(x=alt.X(f"{x}:Q", title=x_label, scale=alt.Scale(zero=False)),
                    y=alt.Y(f"{y}:Q", title=y_label, scale=scale),
                    color=alt.Color(f"{group}:N", title="Model / group"),
                    tooltip=[alt.Tooltip(f"{group}:N"),
                             alt.Tooltip(f"{x}:Q", format=".4f"),
                             alt.Tooltip(f"{y}:Q", format=".7f")])
            .properties(height=height))


def _pricing_frame(points: tuple[PricingObservation, ...]) -> pd.DataFrame:
    # Separate statistical results from the rendering of warning states.
    return pd.DataFrame([{
        "model": p.model, "x": p.coordinate, "price": p.price,
        "runtime_ms": p.runtime_ms, "status": p.status,
        "black_scholes": p.black_scholes,
    } for p in points])


def _status(data: pd.DataFrame) -> None:
    failures = data[data["status"] != "ok"]
    if not failures.empty:
        st.warning(f"{len(failures)} scenarios were not priced. They are excluded from charts, not set to zero.")
        with st.expander("Unpriced scenarios", expanded=False):
            st.dataframe(failures[["model", "x", "status"]], hide_index=True,
                         width="stretch")


def _render_sensitivity_results(saved: tuple) -> None:
    # Collect model-sensitivity inputs through the current Streamlit form.
    parameter, rows = saved
    frame = _pricing_frame(rows)
    successful = frame.dropna(subset=["price"])
    st.metric("Successful valuations", f"{len(successful)} / {len(frame)}")
    _status(frame)
    if not successful.empty:
        st.altair_chart(_line_chart(successful, x="x", y="price", group="model",
                                   x_label=parameter.capitalize(), y_label="Premium (€)"),
                        width="stretch")
        st.caption('Each point is a fresh model valuation. Changes in maturity keep '
            'ex-dividend dates fixed relative to today; invalid shortened contracts are reported separately.')
    st.dataframe(frame, hide_index=True, width="stretch")
    st.download_button("Export sensitivity · CSV", observations_csv(rows),
                       file_name="lattice272_sensitivity.csv", mime="text/csv", width="stretch")


def _sensitivity_input_form() -> tuple:
    # Preserve on-demand execution to avoid unnecessary valuations.
    with st.form("analytics_sensitivity_form"):
        fields = _contract_fields("qa_sens")
        st.markdown('<div class="desk-section">02 · Sweep setup</div>', unsafe_allow_html=True)
        models = _model_select("qa_sens_models", default=("Binomial", "Trinomial · Allocation pruning"))
        c1, c2 = st.columns(2)
        parameter = c1.selectbox("Variable", ("spot", "volatility", "rate", "maturity"),
                                 key="qa_sens_parameter")
        n = c2.number_input("Steps N", min_value=1, max_value=MAX_STEPS, value=100,
                            key="qa_sens_n")
        c1, c2 = st.columns(2)
        multiplier = c1.selectbox("Sweep range multiplier", (0.5, 1.0, 1.5),
                                   index=1, key="qa_sens_multiplier")
        count = c2.selectbox("Grid points", (5,9,13), index=1, key="qa_sens_count")
        st.caption("Base half-widths (× range multiplier): spot ±20%, vol ±5 pp, "
                   "rate ±2 pp, maturity ±25%.")
        # Preserve on-demand execution to avoid unnecessary valuations.
        threshold = _threshold("qa_sens_threshold")
        submitted = st.form_submit_button("RUN SENSITIVITY  →", type="primary", width="stretch")
    return (fields, models, parameter, n, multiplier, count, threshold, submitted)


def _run_sensitivity_on_submit(submitted: bool, fields: dict, models: tuple, n: int,
    threshold: float, parameter: str, multiplier: float, count: int) -> None:
    if submitted:
        st.session_state.pop("qa_sens_result", None)
        try:
            request = _get_request(fields, models[0] if models else MODELS[0], int(n), threshold)
            from frontend.analytics_service import DEFAULT_AMPLITUDE
            width = DEFAULT_AMPLITUDE[parameter]*float(multiplier)
            with st.spinner("Repricing independent market scenarios..."):
                rows = sensitivity_sweep(request, parameter, models, count=count,
                                         amplitude=width)
            st.session_state["qa_sens_result"] = (parameter, rows)
        except (PricingInputError, ValueError, RuntimeError, OverflowError) as exc:
            st.error(f"Sensitivity unavailable: {exc}")


def render_sensitivity_tab() -> None:
    # Execute sensitivity pricing only after the user submits the form.
    left, right = st.columns([1.0, 1.45], gap="large")
    with left:
        fields, models, parameter, n, multiplier, count, threshold, submitted = _sensitivity_input_form()
    with right:
        st.markdown('<div class="desk-section">Repricing sweep</div>', unsafe_allow_html=True)
        _run_sensitivity_on_submit(submitted, fields, models, n, threshold, parameter, multiplier, count)
        saved = st.session_state.get("qa_sens_result")
        # Keep chart axes, tooltips and color grouping consistent.
        if saved is None:
            st.info("Choose a market variable and run independent repricings. No background calculations.")
            return
        _render_sensitivity_results(saved)


# Compare pricing curves with a fixed analytic benchmark.
def _render_convergence_charts(successful: pd.DataFrame, outcome: ConvergenceResult) -> None:
    if not successful.empty:
        st.altair_chart(_line_chart(successful, x="x", y="price", group="model",
                                   x_label="Time steps N", y_label="Premium (€)"),
                        width="stretch")
        if outcome.reference_price is not None:
            errors = successful.copy()
            errors["gap"] = (errors["price"] - outcome.reference_price).abs()
            positive = errors[errors["gap"] > 0]
            if not positive.empty:
                name = ("Absolute Black–Scholes error (€)" if outcome.reference_type == "analytic"
                        else "Absolute gap to finite-N reference (€)")
                st.altair_chart(_line_chart(positive, x="x", y="gap", group="model",
                                           x_label="Time steps N", y_label=name, log_y=True),
                                # Keep chart axes, tooltips and color grouping consistent.
                                width="stretch")
                st.caption("Zero gaps are omitted from the log plot and retained in the CSV.")
        st.altair_chart(_line_chart(successful, x="x", y="runtime_ms", group="model",
                                   x_label="Time steps N", y_label="Single-run runtime (ms)"),
                        width="stretch")
        st.caption('Runtime is a single end-to-end valuation per point, not a replicated '
            'benchmark. For robust latency comparison, use Model Comparison or the research experiments.')


def _render_convergence_results(outcome: ConvergenceResult) -> None:
    frame = _pricing_frame(outcome.points)
    successful = frame.dropna(subset=["price"])
    _status(frame)
    st.caption(f"Reference: {outcome.reference_label} ({outcome.reference_type}).")
    # Preserve on-demand execution to avoid unnecessary valuations.
    if outcome.reference_type == "numerical":
        st.warning('The numerical reference is a selected finite-N price, NOT an exact '
            'option value. Deviations below are price gaps, not accuracy guarantees.')
    elif outcome.reference_error:
        st.warning(f"Reference unavailable: {outcome.reference_error}")
    _render_convergence_charts(successful, outcome)
    _render_accuracy_cost_tradeoff(outcome)
    with st.expander("Convergence observations & CSV", expanded=False):
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button("Export convergence · CSV",
                           observations_csv(outcome.points, reference=outcome.reference_price,
                                            label=outcome.reference_label),
                           file_name="lattice272_convergence.csv", mime="text/csv",
                           width="stretch")



# Lot 4B.1: visible global Pareto envelope, plotting only.
def _pareto_path_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Ordered, globally nondominated empirical Pareto vertices.

    Different pricing models can alternate on the frontier. Duplicates with
    identical coordinates are one graphical vertex but stay in the data table.
    """
    return (rows.loc[rows["Pareto candidate"]]
            .sort_values(["Runtime (ms)", "Absolute gap (€)"], kind="mergesort")
            .drop_duplicates(subset=["Runtime (ms)", "Absolute gap (€)"]))


def _accuracy_cost_tooltips() -> list:
    """Use identical tooltip fields for candidate points and Pareto highlights."""
    return ["Model:N", "N:Q", alt.Tooltip("Premium (€):Q", format=".6f"),
            alt.Tooltip("Runtime (ms):Q", format=".3f"),
            alt.Tooltip("Absolute gap (€):Q", format=".6f"),
            "Pareto candidate:N"]


def _accuracy_cost_chart(rows: pd.DataFrame, *, pareto_only: bool,
                         gap_label: str) -> alt.LayerChart:
    """Plot all observed points plus the *global* nondominated envelope.

    The dashed segments are visual guides between sampled settings; they are
    not evidence of an attainable intermediate configuration.
    """
    displayed = rows.loc[rows["Pareto candidate"]] if pareto_only else rows
    # The envelope is global across models, not per-series interpolation.
    frontier = _pareto_path_frame(rows)
    x = alt.X("Runtime (ms):Q", title="Single-run valuation time (ms)",
              scale=alt.Scale(zero=True))
    y = alt.Y("Absolute gap (€):Q", title=gap_label,
              scale=alt.Scale(zero=True))
    tips = _accuracy_cost_tooltips()
    envelope = (alt.Chart(frontier)
                .mark_line(color="#F5C46D", strokeDash=[6, 4], strokeWidth=2.7)
                .encode(x=x, y=y, order=alt.Order("Runtime (ms):Q",
                                                 sort="ascending")))
    points = (alt.Chart(displayed).mark_circle(size=105, opacity=.92)
              .encode(x=x, y=y,
                      color=alt.Color("Model:N", title=None,
                                      legend=alt.Legend(orient="bottom",
                                                        direction="horizontal",
                                                        columns=2, labelLimit=220)),
                      tooltip=tips))
    highlighted = (alt.Chart(frontier)
                   .mark_point(filled=False, color="#F5C46D",
                               size=205, strokeWidth=2.2)
                   .encode(x=x, y=y, tooltip=tips))
    return alt.layer(envelope, points, highlighted).properties(height=325).interactive()


def _render_accuracy_target_result(points: tuple, target: float) -> None:
    """Render fastest observed qualifying setting, not an interpolated optimum."""
    from frontend.research_accuracy_service import best_observed_under_target
    winner = best_observed_under_target(points, float(target))
    if winner is None:
        st.info(f"No sampled configuration achieved an absolute gap ≤ {target:g} €.")
    else:
        a, b, c = st.columns(3)
        # Keep full, unabridged model names in the data while avoiding clipped KPI text.
        display_model = (winner.model.replace("Trinomial · Allocation pruning", "Allocation pruning")
                         .replace("Trinomial · Branch pruning", "Branch pruning")
                         .replace("Trinomial · Full", "Full trinomial"))
        a.metric("Fastest eligible model", display_model)
        b.metric("Time steps", f"N={winner.steps}")
        c.metric("Observed runtime", f"{winner.runtime_ms:.2f} ms")
        st.caption(f"Observed gap: {winner.absolute_gap:.6f} €. "
                   "The target is checked only against this run's reference. "
                   f"Full model name: {winner.model}.")


def _observed_accuracy_cost_frame(
    points: tuple[AccuracyCostPoint, ...],
) -> pd.DataFrame:
    """Keep the observed price/runtime/Pareto schema stable for charts and CSV."""
    return pd.DataFrame([{
        "Model": p.model, "N": p.steps, "Premium (€)": p.premium,
        "Runtime (ms)": p.runtime_ms, "Absolute gap (€)": p.absolute_gap,
        "Pareto candidate": p.pareto,
    } for p in points])


def _render_accuracy_cost_tradeoff(outcome: ConvergenceResult) -> None:
    """Observed Pareto candidates, a global frontier and on-demand target search."""
    from frontend.research_accuracy_service import accuracy_cost_points

    st.markdown('<div class="desk-section">Accuracy vs computational cost</div>',
                unsafe_allow_html=True)
    points = accuracy_cost_points(outcome)
    if not points:
        st.info("No complete runtime/price/reference observations for a tradeoff chart.")
        return
    analytic = outcome.reference_type == "analytic"
    gap_label = ("Absolute Black–Scholes error (€)" if analytic else
                 "Absolute gap to finite-N reference (€)")
    if not analytic:
        st.caption("Numerical-reference gaps are not true pricing errors. "
                   "The selected finite-N comparator can itself be inaccurate.")
    rows = _observed_accuracy_cost_frame(points)
    a, b = st.columns([1.1, 1.0])
    pareto_only = a.checkbox("Show observed Pareto candidates only", value=False,
                              key="qa_conv_pareto_only")
    target = b.selectbox("Target absolute gap (€)",
                         (0.0005, 0.001, 0.005, 0.01, 0.05, 0.1),
                         index=3, key="qa_conv_gap_target",
                         format_func=lambda x: f"{x:g} €")
    st.altair_chart(_accuracy_cost_chart(rows, pareto_only=pareto_only,
                                        gap_label=gap_label), width="stretch")
    st.caption("Gold dashed line = observed global Pareto frontier across all selected "
               "models; outlined points are the sampled nondominated candidates. "
               "Connecting segments are a visual guide, not an attainable continuous "
               "set of model configurations.")
    _render_accuracy_target_result(points, float(target))
    st.caption("Pareto candidates are nondominated among the sampled settings "
               "for runtime and absolute price gap. Each runtime is a single "
               "measurement; rankings may change with CPU load. This is not a "
               "statistical benchmark or an accuracy guarantee.")
    with st.expander("All observed configurations"):
        st.dataframe(rows, hide_index=True, width="stretch")

def _convergence_input_form() -> tuple:
    with st.form("analytics_convergence_form"):
        fields = _contract_fields("qa_conv")
        st.markdown('<div class="desk-section">02 · Discretization</div>', unsafe_allow_html=True)
        # Keep chart axes, tooltips and color grouping consistent.
        models = _model_select("qa_conv_models", default=("Binomial", "Trinomial · Allocation pruning"))
        options = (10,20,40,60,80,100,125,150,175,200,250,300,350)
        ns = st.multiselect("Time-step grid N", options,
                            default=(20,40,80,100,150,200), key="qa_conv_grid")
        reference = st.selectbox("Numerical reference for non-BS cases", MODELS,
                                 index=3, key="qa_conv_reference")
        threshold = _threshold("qa_conv_threshold")
        submitted = st.form_submit_button("RUN CONVERGENCE  →", type="primary", width="stretch")
    return (fields, models, ns, reference, threshold, submitted)


def render_convergence_tab() -> None:
    left, right = st.columns([1.0, 1.45], gap="large")
    with left:
        fields, models, ns, reference, threshold, submitted = _convergence_input_form()
    with right:
        st.markdown('<div class="desk-section">Discretization & computational cost</div>', unsafe_allow_html=True)
        if submitted:
            # Collect model-sensitivity inputs through the current Streamlit form.
            st.session_state.pop("qa_conv_result", None)
            try:
                request = _get_request(fields, models[0] if models else MODELS[0], 100, threshold)
                with st.spinner("Pricing each model and time-step configuration..."):
                    outcome = convergence_study(request, models, tuple(sorted(ns)),
                                                numerical_reference_model=reference)
                st.session_state["qa_conv_result"] = outcome
            except (PricingInputError, ValueError, RuntimeError, OverflowError) as exc:
                st.error(f"Convergence unavailable: {exc}")
        outcome: ConvergenceResult | None = st.session_state.get("qa_conv_result")
        if outcome is None:
            st.info("Select a discretization grid; errors are measured against Black–Scholes only when applicable.")
            return
        _render_convergence_results(outcome)




def _render_stability_results(saved: tuple) -> None:
    """Retain the existing session data while rendering the focused research view."""
    from frontend.research_risk_numerics_views import render_risk_numerics
    rows, request = saved
    render_risk_numerics(rows, request, stability_csv)


def _greeks_stability_input_form() -> tuple:
    """Risk Numerics: vary spot stencil separately from lattice depth."""
    from frontend.research_risk_numerics import GAMMA_N_MODES
    with st.form("analytics_greeks_stability_form"):
        fields = _contract_fields("qa_stab")
        st.markdown('<div class="desk-section">02 · Numerical robustness</div>',
                    unsafe_allow_html=True)
        model = st.selectbox("Pricing engine", MODELS, index=3, key="qa_stab_model")
        bump_grid = st.multiselect(
            "Spot shocks · % (Delta and Gamma)",
            (0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0),
            default=(0.5, 1.0, 3.5, 5.0), key="qa_stab_bumps",
        )
        n_grid = st.multiselect(
            "Discretization steps N", (20, 40, 60, 80, 100, 125, 150, 175, 200, 250),
            default=(40, 80, 100, 150), key="qa_stab_grid",
        )
        gamma_mode = st.selectbox(
            "Gamma bump for N sweep", GAMMA_N_MODES, index=0,
            key="qa_stab_gamma_n_mode",
            help="Grid-aware adapts Gamma's spot bump to N; fixed bump isolates "
                 "the N effect more cleanly, but still has stencil bias.",
        )
        threshold = _threshold("qa_stab_threshold")
        st.caption("Each shock builds and prices a fresh lattice. "
                   "Run only when the experiment settings are ready.")
        submitted = st.form_submit_button(
            "RUN RISK NUMERICS  →", type="primary", width="stretch",
        )
    return fields, model, bump_grid, n_grid, threshold, gamma_mode, submitted


def render_greeks_stability_tab() -> None:
    """One deliberate repricing campaign; view interactions never reprice."""
    from frontend.research_risk_numerics import n_sweep_gamma_bump

    left, right = st.columns([1.0, 1.85], gap="large")
    with left:
        fields, model, bump_grid, n_grid, threshold, gamma_mode, submitted = (
            _greeks_stability_input_form()
        )
    with right:
        st.markdown('<div class="desk-section">Risk numerics · Delta/Gamma stability</div>',
                    unsafe_allow_html=True)
        if submitted:
            st.session_state.pop("qa_stab_result", None)
            try:
                request = _get_request(fields, model, 100, threshold)
                gamma_bump = n_sweep_gamma_bump(gamma_mode)
                with st.spinner("Repricing risk stencils across bumps and N..."):
                    rows = greeks_stability(
                        request, tuple(sorted(bump_grid)), tuple(sorted(n_grid)),
                        gamma_bump_pct=gamma_bump,
                    )
                st.session_state["qa_stab_result"] = (rows, request)
            except (PricingInputError, ValueError, RuntimeError, OverflowError) as exc:
                st.error(f"Risk numerics unavailable: {exc}")
        saved = st.session_state.get("qa_stab_result")
        if saved is None:
            st.info("Select spot-bump widths and N, then run the study. "
                    "Results are explored without additional pricing.")
            return
        _render_stability_results(saved)
