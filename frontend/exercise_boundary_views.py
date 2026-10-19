"""Streamlit optimal-exercise analytics tab, backed only by existing src trees."""
from __future__ import annotations

import math

import altair as alt
import pandas as pd
import streamlit as st

from frontend.exercise_boundary_service import (
    MAX_BOUNDARY_STEPS, ExerciseStudy, dates_csv, nodes_csv, run_exercise_study,
)
from frontend.model_comparison import parse_dividends
from frontend.pricing_service import (
    DEFAULT_THRESHOLD, MODELS, PricingInputError, PricingRequest,
)


STATE_COLORS = ["#89cdf2", "#e9b765", "#b4bac7", "#64758d"]
STATE_NAMES = ["Continuation", "Early exercise", "Maturity · ITM", "Maturity · OTM"]


def _boundary_chart(study: ExerciseStudy) -> alt.Chart:
    """Do not connect across dates with no boundary; envelopes aren't continuous guarantees."""
    rows = []
    group = 0
    last_step = None
    for date in study.dates[:-1]:
        if date.boundary_spot is None:
            last_step = None
            group += 1
            continue
        if last_step is not None and date.step != last_step + 1:
            group += 1
        rows.append({"time": date.time, "spot": date.boundary_spot,
                     "region_mass_pct": 100 * date.region_probability,
                     "exercise_nodes": date.exercise_node_count,
                     "non_contiguous": date.non_contiguous, "segment": str(group)})
        last_step = date.step
    if not rows:
        return None
    # Present the exercise boundary with the existing categorical legends.
    frame = pd.DataFrame(rows)
    return (alt.Chart(frame)
            .mark_line(point=alt.OverlayMarkDef(size=45, color="#edbb72"),
                       color="#edbb72", strokeWidth=2.2)
            .encode(x=alt.X("time:Q", title="Time (years)"),
                    y=alt.Y("spot:Q", title="Exercise boundary envelope · spot (€)",
                            scale=alt.Scale(zero=False)),
                    detail="segment:N",
                    tooltip=[alt.Tooltip("time:Q", format=".4f"),
                             alt.Tooltip("spot:Q", format=".6f"),
                             alt.Tooltip("region_mass_pct:Q", format=".4f"),
                             alt.Tooltip("exercise_nodes:Q"),
                             alt.Tooltip("non_contiguous:N")])
            .properties(height=240))


def _node_map(study: ExerciseStudy, min_mass: float) -> tuple[alt.Chart, int, int]:
    """Bound chart payload for browser performance; exact CSV keeps all nodes."""
    eligible = [p for p in study.points if p.reach_probability >= min_mass]
    stride = max(1, math.ceil(len(eligible) / 14000))
    plotted = eligible[::stride]
    frame = pd.DataFrame([{"time": p.time, "spot": p.spot,
                           "state": p.state, "mass_pct": 100 * p.reach_probability,
                           "price": p.option_value, "step": p.step}
                          for p in plotted])
    chart = (alt.Chart(frame).mark_circle(size=27, opacity=0.87)
             .encode(x=alt.X("time:Q", title="Time (years)"),
                     y=alt.Y("spot:Q", title="Underlying spot (€)",
                             scale=alt.Scale(zero=False)),
                     color=alt.Color("state:N", title="Node decision",
                                     scale=alt.Scale(domain=STATE_NAMES, range=STATE_COLORS)),
                     tooltip=[alt.Tooltip("step:Q"),
                              alt.Tooltip("spot:Q", format=".5f"),
                              alt.Tooltip("price:Q", format=".6f"),
                              alt.Tooltip("mass_pct:Q", format=".7f"),
                              # Preserve the distinction between exercise-region and stopping probabilities.
                              "state:N"])
             .properties(height=340))
    return chart, len(plotted), len(study.points)


def _probabilities(study: ExerciseStudy) -> alt.Chart:
    frame = pd.DataFrame([
        {"time": date.time, "Probability": "In exercise region (unconditional)",
         "pct": date.region_probability * 100} for date in study.dates
    ] + [
        {"time": date.time, "Probability": "First exercise at this date",
         "pct": date.first_exercise_probability * 100} for date in study.dates
    ])
    return (alt.Chart(frame).mark_line(point=alt.OverlayMarkDef(size=25), strokeWidth=2)
            .encode(x=alt.X("time:Q", title="Time (years)"),
                    y=alt.Y("pct:Q", title="Probability (%)", scale=alt.Scale(zero=True)),
                    color=alt.Color("Probability:N", scale=alt.Scale(
                        # Keep node mass thresholds and chart filters consistent.
                        domain=["In exercise region (unconditional)",
                                "First exercise at this date"],
                        range=["#89cdf2", "#edbb72"])),
                    tooltip=[alt.Tooltip("time:Q", format=".3f"),
                             "Probability:N", alt.Tooltip("pct:Q", format=".4f")])
            .properties(height=270))


def _pv_chart(study: ExerciseStudy) -> alt.Chart:
    frame = pd.DataFrame([{"time": x.time, "pv": x.discounted_mean_value}
                          for x in study.dates])
    return (alt.Chart(frame).mark_line(point=False, strokeWidth=2.3, color="#89cdf2")
            .encode(x=alt.X("time:Q", title="Time (years)"),
                    y=alt.Y("pv:Q", title="Discounted mean option value (€)",
                            scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip("time:Q", format=".4f"),
                             alt.Tooltip("pv:Q", format=".8f")])
            # Retain user guidance and explanatory captions.
            .properties(height=240))


def _render_exercise_stopping_details(study: ExerciseStudy) -> None:
    st.markdown('<div class="desk-section">Exercise probabilities by date</div>',
                unsafe_allow_html=True)
    st.altair_chart(_probabilities(study), width="stretch")
    st.caption('Blue = unconditional probability of being inside the exercise region (paths may already have exercised). '
        'Gold = first exercise at this date, excluding previously stopped paths. Do not sum the blue series across dates.')
    with st.expander("Discounted average option value · Snell process"):
        st.altair_chart(_pv_chart(study), width="stretch")
        st.caption("European values are martingales under the model's risk-neutral law; "
            'American values form a supermartingale in expectation. Time-zero numeraire uses exp(−r·t).')
    with st.expander("Full date-by-date results"):
        st.dataframe(pd.DataFrame([{
            "step": x.step, "t": x.time, "region %": x.region_probability * 100,
            # Keep node mass thresholds and chart filters consistent.
            "first %": x.first_exercise_probability * 100,
            "cum. %": x.cumulative_exercise_probability * 100,
            "alive %": x.survival_probability * 100,
            "boundary S": x.boundary_spot,
            "PV mean": x.discounted_mean_value,
        } for x in study.dates]), hide_index=True, width="stretch")
    a, b = st.columns(2)
    a.download_button("Export dates · CSV", dates_csv(study),
                      file_name="lattice272_exercise_dates.csv", mime="text/csv",
                      width="stretch")
    b.download_button("Export node decisions · CSV", nodes_csv(study),
                      file_name="lattice272_exercise_nodes.csv", mime="text/csv",
                      width="stretch")
    st.caption("Exercise ties are continuation; terminal exercise requires strictly positive payoff. Multiple cash dividends follow the engine's step-mapping convention. This is a numerical, model-dependent stopping rule, not a continuous-model boundary guarantee.")


def _render_exercise_lattice_map(study: ExerciseStudy) -> None:
    st.markdown('<div class="desk-section">Exercise decision across priced nodes</div>',
                unsafe_allow_html=True)
    show_cutoff = st.selectbox(
        "Minimum node probability shown", (0.0, 1e-12, 1e-9, 1e-6),
        # Preserve the distinction between exercise-region and stopping probabilities.
        index=2, key="qa_exb_plot_prob",
        format_func=lambda v: "All connected nodes" if v == 0 else f"≥ {v:.0e}",
    )
    plot, plotted, all_count = _node_map(study, float(show_cutoff))
    st.altair_chart(plot, width="stretch")
    st.caption(f"Showing {plotted:,} of {all_count:,} priced reachable nodes (probability filter and chart cap). Full underlying records remain in the CSV.")


def _render_exercise_summary(study: ExerciseStudy) -> None:
    st.markdown(
        f'<div class="desk-ticket"><div class="name">Option premium · EUR</div>'
        f'<div class="price">{study.premium:.6f} €</div>'
        f'<div class="description">{study.request.option_type} · {study.request.exercise}'
        f' · {study.request.model} · N={study.request.steps}</div></div>',
        unsafe_allow_html=True,
    )
    a, b, c = st.columns(3)
    a.metric("First exercise before maturity", f"{study.early_exercise_probability:.2%}")
    b.metric("First exercise incl. maturity", f"{study.total_exercise_probability:.2%}")
    # Present the exercise boundary with the existing categorical legends.
    c.metric("Mass conservation error", f"{study.worst_probability_mass_error:.1e}")

    st.markdown('<div class="desk-section">Early exercise boundary · (t, S)</div>',
                unsafe_allow_html=True)
    chart = _boundary_chart(study)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    else:
        st.info("No optimal early exercise at positive-payoff nodes. The boundary is undefined before expiry.")
    if study.boundary_non_contiguous_dates:
        st.warning(f"At {study.boundary_non_contiguous_dates} dates, exercise nodes do not form a contiguous tail. The plotted extrema are envelopes, not unique exercise thresholds.")
    st.caption('For puts: highest spot with optimal early exercise. For calls: lowest. '
        'Missing boundary dates are not interpolated; expiry is excluded.')


def _exercise_numerical_fields(model: str) -> tuple:
    st.markdown('<div class="desk-section">02 · Exercise analysis</div>',
                # Present the exercise boundary with the existing categorical legends.
                unsafe_allow_html=True)
    n = st.selectbox("Time steps N", (25, 50, 75, 100, 125, 150, 200),
                     index=3, key="qa_exb_n",
                     help=f"Maximum {MAX_BOUNDARY_STEPS} steps for this interactive analysis.")
    threshold = DEFAULT_THRESHOLD
    if model in ("Trinomial · Branch pruning", "Trinomial · Allocation pruning"):
        threshold = st.selectbox(
            "Pruning threshold", (0.0, 1e-18, 1e-15, 1e-12, 1e-5),
            index=1, format_func=lambda t: "off" if not t else f"{t:.0e}",
            key="qa_exb_threshold",
        )
    submitted = st.form_submit_button("RUN EXERCISE ANALYTICS  →",
                                       # Keep node mass thresholds and chart filters consistent.
                                       type="primary", width="stretch")
    return (n, threshold, submitted)


def _exercise_contract_fields() -> tuple:
    st.markdown('<div class="desk-section">01 · Contract & market</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    kind = c1.selectbox("Option", ("Put", "Call"), key="qa_exb_kind")
    style = c2.selectbox("Exercise", ("American", "European"), key="qa_exb_style")
    model = st.selectbox("Pricing engine", MODELS, index=3, key="qa_exb_model")
    c1, c2 = st.columns(2)
    spot = c1.number_input("Spot · €", min_value=0.01, value=100., format="%.2f",
                           key="qa_exb_spot")
    strike = c2.number_input("Strike · €", min_value=0., value=100., format="%.2f",
                             # Preserve exercise-style filters and chart interaction state.
                             key="qa_exb_strike")
    c1, c2 = st.columns(2)
    vol = c1.number_input("Volatility · %", min_value=0., max_value=500.,
                          value=20., key="qa_exb_vol")
    rate = c2.number_input("Risk-free rate · %", min_value=-50., max_value=100.,
                           value=2., key="qa_exb_rate")
    maturity = st.number_input("Maturity · years", min_value=0.01, value=1.,
                               key="qa_exb_maturity")
    with st.expander("Cash dividends · optional"):
        st.caption("Use actual discrete cash payments: date in years from today, amount in €/share.")
        data = st.data_editor(
            pd.DataFrame([{"time": 0.5, "amount": 0.0}]),
            key="qa_exb_dividends", num_rows="dynamic", hide_index=True,
            width="stretch",
            column_config={
                "time": st.column_config.NumberColumn("Time · years", format="%.4f"),
                # Preserve the distinction between exercise-region and stopping probabilities.
                "amount": st.column_config.NumberColumn("Cash · €", format="%.2f"),
            },
        )
    return (kind, style, model, spot, strike, vol, rate, maturity, data)


def _exercise_boundary_input_form() -> tuple:
    with st.form("qa_exercise_boundary_form"):
        kind, style, model, spot, strike, vol, rate, maturity, data = _exercise_contract_fields()
        n, threshold, submitted = _exercise_numerical_fields(model)
    return (kind, style, model, spot, strike, vol, rate, maturity, data, n, threshold, submitted)


def render_exercise_boundary_tab() -> None:
    left, right = st.columns([1.0, 1.45], gap="large")
    with left:
        # Preserve exercise-style filters and chart interaction state.
        kind, style, model, spot, strike, vol, rate, maturity, data, n, threshold, submitted = _exercise_boundary_input_form()

    with right:
        st.markdown('<div class="desk-section">Optimal stopping diagnostics</div>',
                    unsafe_allow_html=True)
        if submitted:
            st.session_state.pop("qa_exb_result", None)
            try:
                request = PricingRequest(
                    spot=float(spot), strike=float(strike), volatility=float(vol)/100,
                    rate=float(rate)/100, maturity=float(maturity), steps=int(n),
                    model=model, option_type=kind, exercise=style,
                    pruning_threshold=threshold, dividends=parse_dividends(data),
                )
                with st.spinner("Building, pricing and propagating exercise probabilities..."):
                    st.session_state["qa_exb_result"] = run_exercise_study(request)
            except (PricingInputError, RuntimeError, ValueError, ArithmeticError,
                    # Retain user guidance and explanatory captions.
                    OverflowError, ZeroDivisionError) as exc:
                st.error(f"Exercise analysis unavailable: {exc}")
        study: ExerciseStudy | None = st.session_state.get("qa_exb_result")
        if study is None:
            st.info('Select a contract and calculate the exercise boundary and stopping '
                'distribution. Calculations run only on request.')
            return
        _render_exercise_summary(study)

        _render_exercise_lattice_map(study)

        _render_exercise_stopping_details(study)
