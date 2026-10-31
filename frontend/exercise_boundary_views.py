"""Streamlit optimal-exercise analytics tab, backed only by existing src trees."""
from __future__ import annotations

import math

import altair as alt
import pandas as pd
import streamlit as st

from frontend.exercise_boundary_service import (
    MAX_BOUNDARY_STEPS, ExerciseStudy, dates_csv, nodes_csv, run_exercise_study,
)
from frontend.research_stopping_service import (
    DividendExerciseComparison, compare_dividend_exercise,
)
from frontend.dividend_input import parse_dividends
from frontend.pricing_service import (
    DEFAULT_THRESHOLD, MODELS, PricingInputError, PricingRequest,
)


STATE_COLORS = ["#89cdf2", "#e9b765", "#b4bac7", "#64758d"]
STATE_NAMES = ["Continuation", "Early exercise", "Maturity · ITM", "Maturity · OTM"]


# A boundary is an envelope of exercised nodes; gaps must not be interpolated.
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
                     color=alt.Color("state:N", legend=alt.Legend(
                                     orient="bottom", columns=2, title=None),
                                     scale=alt.Scale(domain=STATE_NAMES, range=STATE_COLORS)),
                     tooltip=[alt.Tooltip("step:Q"),
                              alt.Tooltip("spot:Q", format=".5f"),
                              alt.Tooltip("price:Q", format=".6f"),
                              alt.Tooltip("mass_pct:Q", format=".7f"),
                              # Preserve the distinction between exercise-region and stopping probabilities.
                              "state:N"])
             .properties(height=345))
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
                    color=alt.Color("Probability:N", legend=alt.Legend(
                        orient="bottom", direction="horizontal", title=None),
                        scale=alt.Scale(
                        # Keep node mass thresholds and chart filters consistent.
                        domain=["In exercise region (unconditional)",
                                "First exercise at this date"],
                        range=["#89cdf2", "#edbb72"])),
                    tooltip=[alt.Tooltip("time:Q", format=".3f"),
                             "Probability:N", alt.Tooltip("pct:Q", format=".4f")])
            .properties(height=345))


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
    """Node-level probability chart in the right half of the diagnostics grid."""
    st.markdown("**Exercise probabilities · by date**")
    st.altair_chart(_probabilities(study), width="stretch")
    st.caption(
        "Blue: unconditional mass in the exercise region · "
        "Gold: first exercise at a node. Do not sum the blue series over time."
    )
    if study.request.exercise == "American" and study.request.dividends:
        st.caption("Dividend-event exercises outside priced nodes are not included "
                   "in these probabilities; see Methodology above.")


def _render_exercise_lattice_map(study: ExerciseStudy) -> None:
    """Compact node map for the left half of the diagnostics grid."""
    st.markdown("**Exercise decision map**")
    show_cutoff = st.selectbox(
        "Minimum node probability shown", (0.0, 1e-12, 1e-9, 1e-6),
        index=2, key="qa_exb_plot_prob",
        format_func=lambda v: "All connected nodes" if v == 0 else f"≥ {v:.0e}",
    )
    plot, plotted, all_count = _node_map(study, float(show_cutoff))
    st.altair_chart(plot, width="stretch")
    st.caption(f"{plotted:,} / {all_count:,} priced nodes visible. "
               "Probability filter and chart cap affect the plot, not the exported data.")


def _render_exercise_summary(study: ExerciseStudy) -> None:
    """Primary contract, pricing and exercise summary, before comparisons."""
    st.markdown('<div class="desk-section">01 · Core stopping summary</div>',
                unsafe_allow_html=True)
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
    c.metric("Mass conservation error", f"{study.worst_probability_mass_error:.1e}")

    st.markdown("**Exercise boundary · primary contract**")
    chart = _boundary_chart(study)
    if chart is None:
        st.info("No optimal early exercise at positive-payoff nodes before expiry.")
    else:
        st.altair_chart(chart.properties(height=270), width="stretch")
    if study.boundary_non_contiguous_dates:
        st.caption(
            f"Non-contiguous exercise region on {study.boundary_non_contiguous_dates} dates: "
            "the plotted boundary is an envelope, not necessarily a unique threshold."
        )
    st.caption("Put: highest exercised spot · Call: lowest exercised spot · "
               "undefined dates are not interpolated; expiry is excluded.")


# Connected segments require adjacent valid ex-date boundary observations.
def _dividend_comparison_frame(comparison: DividendExerciseComparison) -> pd.DataFrame:
    """Discontinuous boundary envelopes must never be connected across gaps."""
    rows: list[dict] = []
    for scenario, study in (
        ("Cash dividends", comparison.with_dividends),
        ("No cash dividends", comparison.without_dividends),
    ):
        segment = 0
        previous = None
        for point in study.dates[:-1]:
            if point.boundary_spot is None:
                previous = None
                segment += 1
                continue
            if previous is not None and point.step != previous + 1:
                segment += 1
            rows.append({
                "time": point.time,
                "spot": point.boundary_spot,
                "scenario": scenario,
                "curve": f"{scenario}-{segment}",
                "non_contiguous": point.non_contiguous,
            })
            previous = point.step
    return pd.DataFrame(rows, columns=[
        "time", "spot", "scenario", "curve", "non_contiguous",
    ])


def _dividend_comparison_chart(comparison: DividendExerciseComparison, kind: str) -> alt.Chart:
    """Live comparison of actual nodal boundaries or nodal exercise mass."""
    if kind == "boundary":
        frame = _dividend_comparison_frame(comparison)
        chart = (alt.Chart(frame)
                 .mark_line(point=alt.OverlayMarkDef(size=34), strokeWidth=2.2)
                 .encode(
                     x=alt.X("time:Q", title="Time (years)"),
                     y=alt.Y("spot:Q", title="Nodal exercise envelope (€)",
                             scale=alt.Scale(zero=False)),
                     color=alt.Color("scenario:N", legend=alt.Legend(
                         orient="bottom", direction="horizontal", title=None),
                         scale=alt.Scale(domain=["Cash dividends", "No cash dividends"],
                                         range=["#e9b765", "#89cdf2"])),
                     detail="curve:N",
                     tooltip=[alt.Tooltip("time:Q", format=".4f"),
                              alt.Tooltip("spot:Q", format=".5f"),
                              "scenario:N", "non_contiguous:N"],
                 ))
    elif kind == "node_first_exercise":
        frame = pd.DataFrame([
            {"time": date.time, "probability": date.first_exercise_probability * 100,
             "scenario": scenario}
            for scenario, study in (
                ("Cash dividends", comparison.with_dividends),
                ("No cash dividends", comparison.without_dividends),
            )
            for date in study.dates
        ])
        chart = (alt.Chart(frame).mark_line(strokeWidth=2.2)
                 .encode(
                     x=alt.X("time:Q", title="Time (years)"),
                     y=alt.Y("probability:Q", title="First exercise at a lattice node (%)",
                             scale=alt.Scale(zero=True)),
                     color=alt.Color("scenario:N", legend=alt.Legend(
                         orient="bottom", direction="horizontal", title=None),
                         scale=alt.Scale(domain=["Cash dividends", "No cash dividends"],
                                         range=["#e9b765", "#89cdf2"])),
                     tooltip=[alt.Tooltip("time:Q", format=".4f"),
                              alt.Tooltip("probability:Q", format=".4f"),
                              "scenario:N"],
                 ))
    else:
        raise ValueError("Unknown stopping comparison chart.")
    event_frame = pd.DataFrame([
        {"time": dividend.time, "amount": dividend.amount}
        for dividend in comparison.with_dividends.request.dividends
    ])
    events = (alt.Chart(event_frame).mark_rule(color="#B58A5F", strokeDash=[4, 4],
                                                opacity=.65)
              .encode(x="time:Q", tooltip=[alt.Tooltip("time:Q", format=".4f"),
                                            alt.Tooltip("amount:Q", format=".2f")]))
    return alt.layer(chart, events).properties(height=290).interactive()


# Same market inputs isolate the cash-dividend effect, not an equal-forward effect.
def _render_dividend_comparison(comparison: DividendExerciseComparison) -> None:
    """An explicit two-panel comparison, not nested inside nodal diagnostics."""
    st.markdown('<div class="desk-section">02 · Dividend impact</div>',
                unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Premium · with dividends", f"{comparison.with_dividends.premium:.6f} €")
    b.metric("Premium · without dividends", f"{comparison.without_dividends.premium:.6f} €")
    c.metric("Premium difference", f"{comparison.premium_difference:+.6f} €")

    boundary_col, first_col = st.columns(2, gap="large")
    with boundary_col:
        st.markdown("**Exercise boundary · with / without dividends**")
        if _dividend_comparison_frame(comparison).empty:
            st.info("Neither contract has an early-exercise boundary at a priced node.")
        else:
            st.altair_chart(_dividend_comparison_chart(comparison, "boundary"),
                            width="stretch")
    with first_col:
        st.markdown("**First exercise · node dates only**")
        st.altair_chart(_dividend_comparison_chart(comparison, "node_first_exercise"),
                        width="stretch")

    st.caption("Gold: cash dividends · Blue: no cash dividends · "
               "vertical dashed rules: dividend ex-dates. Same initial spot and "
               "market inputs; this is not an equal-forward comparison.")
    with st.expander("Methodology · comparison and dividend-event exercise", expanded=False):
        st.caption(
            "Contracts share spot, strike, rate, volatility, maturity, engine, "
            "steps and pruning. Only cash dividends change. Exercise boundaries "
            "connect adjacent dates only when a nodal exercise region exists."
        )
        if comparison.with_dividends.request.exercise == "American":
            st.warning(
                "First-exercise probabilities count lattice-node exercise only. "
                "Exercise around dividend events, including off-grid events, "
                "affects option values but is not counted in these stopping-time curves. "
                "They must not be interpreted as the complete first-exercise distribution."
            )


# Editorial layout; financial analytics are unchanged.

# Keep date-level stopping mass and full-node exports separate from chart sampling.
def _render_exercise_secondary_details(study: ExerciseStudy) -> None:
    """Detailed data and Snell diagnostics outside the primary chart grid."""
    with st.expander("Snell process · discounted mean option value", expanded=False):
        st.altair_chart(_pv_chart(study).properties(height=265), width="stretch")
        st.caption(
            "Under the model's risk-neutral law, discounted European values are "
            "martingales and discounted American values are supermartingales in "
            "expectation. Discount convention: exp(−r·t)."
        )
    with st.expander("Full date-by-date diagnostics & CSV exports", expanded=False):
        frame = pd.DataFrame([{
            "step": item.step,
            "t": item.time,
            "region %": item.region_probability * 100,
            "first %": item.first_exercise_probability * 100,
            "cum. %": item.cumulative_exercise_probability * 100,
            "alive %": item.survival_probability * 100,
            "boundary S": item.boundary_spot,
            "PV mean": item.discounted_mean_value,
        } for item in study.dates])
        st.dataframe(frame, hide_index=True, width="stretch")
        a, b = st.columns(2)
        a.download_button("Export dates · CSV", dates_csv(study),
                          file_name="lattice272_exercise_dates.csv", mime="text/csv",
                          width="stretch")
        b.download_button("Export node decisions · CSV", nodes_csv(study),
                          file_name="lattice272_exercise_nodes.csv", mime="text/csv",
                          width="stretch")
        st.caption("Ties imply continuation. Maturity exercise requires a strictly "
                   "positive payoff. Dividend-event mapping follows the engine's "
                   "grid convention; boundaries are numerical, not continuous-time guarantees.")


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
    comparison = st.checkbox(
        "Compare with identical contract without cash dividends",
        value=False, key="qa_exb_dividend_comparison",
        help="Run two models at the same S0, K, rate, volatility, N and T; "
             "only dividends differ. Calculations start on submit.",
    )
    submitted = st.form_submit_button("RUN EXERCISE ANALYTICS  →",
                                       # Keep node mass thresholds and chart filters consistent.
                                       type="primary", width="stretch")
    return (n, threshold, comparison, submitted)


def _exercise_market_fields() -> tuple:
    """Collect the contract and market without changing widget keys."""
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
    return kind, style, model, spot, strike, vol, rate, maturity


def _exercise_dividend_editor() -> pd.DataFrame:
    """Keep the discrete-dividend editor isolated from contract fields."""
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
    return data


def _exercise_contract_fields() -> tuple:
    st.markdown('<div class="desk-section">01 · Contract & market</div>',
                unsafe_allow_html=True)
    kind, style, model, spot, strike, vol, rate, maturity = _exercise_market_fields()
    data = _exercise_dividend_editor()
    return (kind, style, model, spot, strike, vol, rate, maturity, data)


def _exercise_boundary_input_form() -> tuple:
    with st.form("qa_exercise_boundary_form"):
        kind, style, model, spot, strike, vol, rate, maturity, data = _exercise_contract_fields()
        n, threshold, comparison, submitted = _exercise_numerical_fields(model)
    return (kind, style, model, spot, strike, vol, rate, maturity, data, n, threshold, comparison, submitted)


def _submit_exercise_analysis(inputs: tuple, submitted: bool) -> None:
    """Run the same validation and engine path only on submit."""
    kind, style, model, spot, strike, vol, rate, maturity, data, n, threshold, comparison = inputs
    if submitted:
        st.session_state.pop("qa_exb_result", None)
        st.session_state.pop("qa_exb_comparison", None)
        try:
            request = PricingRequest(
                spot=float(spot), strike=float(strike), volatility=float(vol)/100,
                rate=float(rate)/100, maturity=float(maturity), steps=int(n),
                model=model, option_type=kind, exercise=style,
                pruning_threshold=threshold, dividends=parse_dividends(data),
            )
            with st.spinner("Building, pricing and propagating exercise probabilities..."):
                if comparison:
                    pair = compare_dividend_exercise(request)
                    st.session_state["qa_exb_result"] = pair.with_dividends
                    st.session_state["qa_exb_comparison"] = pair
                else:
                    st.session_state["qa_exb_result"] = run_exercise_study(request)
        except (PricingInputError, RuntimeError, ValueError, ArithmeticError,
                # Retain user guidance and explanatory captions.
                OverflowError, ZeroDivisionError) as exc:
            st.error(f"Exercise analysis unavailable: {exc}")


def _show_exercise_results() -> None:
    """Show the cached stopping study, comparisons and node charts."""
    study: ExerciseStudy | None = st.session_state.get("qa_exb_result")
    if study is None:
        st.info('Select a contract and calculate the exercise boundary and stopping '
            'distribution. Calculations run only on request.')
        return
    _render_exercise_summary(study)
    pair = st.session_state.get("qa_exb_comparison")
    if pair is not None:
        _render_dividend_comparison(pair)

    # Node exercise markers and the stopping distribution use one priced study.
    st.markdown('<div class="desk-section">03 · Node-level diagnostics</div>',
                unsafe_allow_html=True)
    left_chart, right_chart = st.columns([1.10, 1.0], gap="large")
    with left_chart:
        _render_exercise_lattice_map(study)
    with right_chart:
        _render_exercise_stopping_details(study)
    _render_exercise_secondary_details(study)


def render_exercise_boundary_tab() -> None:
    left, right = st.columns([1.0, 1.72], gap="large")
    with left:
        # Preserve exercise-style filters and chart interaction state.
        (kind, style, model, spot, strike, vol, rate, maturity,
         data, n, threshold, comparison, submitted) = _exercise_boundary_input_form()

    with right:
        st.markdown('<div class="desk-section">Optimal stopping diagnostics</div>',
                    unsafe_allow_html=True)
        _submit_exercise_analysis(
            (kind, style, model, spot, strike, vol, rate, maturity, data,
             n, threshold, comparison), submitted,
        )
        _show_exercise_results()
