"""Streamlit Quant Analytics panels for sensitivity, convergence, Greek stability.

All heavy computation is button-triggered. These panels do not modify src/.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from frontend.analytics_service import (
    ConvergenceResult, PricingObservation, StabilityObservation,
    convergence_study, greeks_stability, observations_csv,
    sensitivity_sweep, stability_csv,
)
from frontend.model_comparison import parse_dividends
from frontend.pricing_service import (
    DEFAULT_THRESHOLD, MAX_STEPS, MODELS, PricingInputError, PricingRequest,
)


def _contract_fields(prefix: str) -> dict:
    """Reusable but independently keyed input fields inside the calling form."""
    st.markdown('<div class="desk-section">01 · Contract & market</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    option = a.selectbox("Option", ("Call", "Put"), key=f"{prefix}_option")
    exercise = b.selectbox("Exercise", ("European", "American"), key=f"{prefix}_exercise")
    a, b = st.columns(2)
    spot = a.number_input("Spot · €", min_value=0.01, value=100.0,
                          key=f"{prefix}_spot", format="%.2f")
    strike = b.number_input("Strike · €", min_value=0.0, value=100.0,
                            key=f"{prefix}_strike", format="%.2f")
    a, b = st.columns(2)
    vol = a.number_input("Volatility · %", min_value=0.0, max_value=500.0,
                         value=20.0, key=f"{prefix}_vol")
    rate = b.number_input("Risk-free rate · %", min_value=-50.0, max_value=100.0,
                          value=2.0, key=f"{prefix}_rate")
    maturity = st.number_input("Maturity · years", min_value=0.01, value=1.0,
                               key=f"{prefix}_maturity")
    with st.expander("Cash dividends · optional"):
        st.caption("Discrete cash payments from today, expressed in years and € per share.")
        data = st.data_editor(
            pd.DataFrame([{"time": 0.5, "amount": 0.0}]),
            key=f"{prefix}_dividends", num_rows="dynamic", hide_index=True,
            width="stretch",
            column_config={
                "time": st.column_config.NumberColumn("Payment time · years", format="%.4f"),
                "amount": st.column_config.NumberColumn("Cash amount · €", format="%.2f"),
            },
        )
    return dict(spot=spot, strike=strike, rate=rate/100.0,
                volatility=vol/100.0, maturity=maturity, option_type=option,
                exercise=exercise, dividends_frame=data)


def _get_request(values: dict, model: str, n: int, threshold: float) -> PricingRequest:
    return PricingRequest(
        spot=float(values["spot"]), strike=float(values["strike"]),
        rate=float(values["rate"]), volatility=float(values["volatility"]),
        maturity=float(values["maturity"]), option_type=values["option_type"],
        exercise=values["exercise"],
        dividends=parse_dividends(values["dividends_frame"]),
        model=model, steps=n, pruning_threshold=threshold,
    )


def _model_select(key: str, *, default: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(st.multiselect("Models to compare", MODELS, default=default, key=key))


def _threshold(key: str) -> float:
    return st.selectbox("Pruning threshold", (0.0, 1e-18, 1e-15, 1e-12, 1e-5),
                        index=1, key=key,
                        format_func=lambda x: "off" if not x else f"{x:.0e}")


def _line_chart(data: pd.DataFrame, *, x: str, y: str, group: str,
                x_label: str, y_label: str, log_y: bool = False, height: int = 310):
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
    return pd.DataFrame([{
        "model": p.model, "x": p.coordinate, "price": p.price,
        "runtime_ms": p.runtime_ms, "status": p.status,
        "black_scholes": p.black_scholes,
    } for p in points])


def _status(data: pd.DataFrame):
    failures = data[data["status"] != "ok"]
    if not failures.empty:
        st.warning(f"{len(failures)} scenarios were not priced. They are excluded from charts, not set to zero.")
        with st.expander("Unpriced scenarios", expanded=False):
            st.dataframe(failures[["model", "x", "status"]], hide_index=True,
                         width="stretch")


def render_sensitivity_tab() -> None:
    left, right = st.columns([1.0, 1.45], gap="large")
    with left:
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
            threshold = _threshold("qa_sens_threshold")
            submitted = st.form_submit_button("RUN SENSITIVITY  →", type="primary", width="stretch")
    with right:
        st.markdown('<div class="desk-section">Repricing sweep</div>', unsafe_allow_html=True)
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
        saved = st.session_state.get("qa_sens_result")
        if saved is None:
            st.info("Choose a market variable and run independent repricings. No background calculations.")
            return
        parameter, rows = saved
        frame = _pricing_frame(rows)
        successful = frame.dropna(subset=["price"])
        st.metric("Successful valuations", f"{len(successful)} / {len(frame)}")
        _status(frame)
        if not successful.empty:
            st.altair_chart(_line_chart(successful, x="x", y="price", group="model",
                                       x_label=parameter.capitalize(), y_label="Premium (€)"),
                            width="stretch")
            st.caption("Each point is a fresh model valuation. Changes in maturity keep ex-dividend dates fixed relative to today; invalid shortened contracts are reported separately.")
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button("Export sensitivity · CSV", observations_csv(rows),
                           file_name="lattice272_sensitivity.csv", mime="text/csv", width="stretch")


def render_convergence_tab() -> None:
    left, right = st.columns([1.0, 1.45], gap="large")
    with left:
        with st.form("analytics_convergence_form"):
            fields = _contract_fields("qa_conv")
            st.markdown('<div class="desk-section">02 · Discretization</div>', unsafe_allow_html=True)
            models = _model_select("qa_conv_models", default=("Binomial", "Trinomial · Allocation pruning"))
            options = (10,20,40,60,80,100,125,150,175,200,250,300,350)
            ns = st.multiselect("Time-step grid N", options,
                                default=(20,40,80,100,150,200), key="qa_conv_grid")
            reference = st.selectbox("Numerical reference for non-BS cases", MODELS,
                                     index=3, key="qa_conv_reference")
            threshold = _threshold("qa_conv_threshold")
            submitted = st.form_submit_button("RUN CONVERGENCE  →", type="primary", width="stretch")
    with right:
        st.markdown('<div class="desk-section">Discretization & computational cost</div>', unsafe_allow_html=True)
        if submitted:
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
        frame = _pricing_frame(outcome.points)
        successful = frame.dropna(subset=["price"])
        _status(frame)
        st.caption(f"Reference: {outcome.reference_label} ({outcome.reference_type}).")
        if outcome.reference_type == "numerical":
            st.warning("The numerical reference is a selected finite-N price, NOT an exact option value. Deviations below are price gaps, not accuracy guarantees.")
        elif outcome.reference_error:
            st.warning(f"Reference unavailable: {outcome.reference_error}")
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
                                    width="stretch")
                    st.caption("Zero gaps are omitted from the log plot and retained in the CSV.")
            st.altair_chart(_line_chart(successful, x="x", y="runtime_ms", group="model",
                                       x_label="Time steps N", y_label="Single-run runtime (ms)"),
                            width="stretch")
            st.caption("Runtime is a single end-to-end valuation per point, not a replicated benchmark. For robust latency comparison, use Model Comparison or the research experiments.")
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button("Export convergence · CSV",
                           observations_csv(outcome.points, reference=outcome.reference_price,
                                            label=outcome.reference_label),
                           file_name="lattice272_convergence.csv", mime="text/csv", width="stretch")


def render_greeks_stability_tab() -> None:
    left, right = st.columns([1.0,1.45], gap="large")
    with left:
        with st.form("analytics_greeks_stability_form"):
            fields = _contract_fields("qa_stab")
            st.markdown('<div class="desk-section">02 · Differencing & grid study</div>', unsafe_allow_html=True)
            model = st.selectbox("Pricing engine", MODELS, index=3, key="qa_stab_model")
            bump_grid = st.multiselect("Spot shocks · % (Delta and Gamma)",
                                       (0.5,1.0,2.0,3.5,5.0,7.5,10.0),
                                       default=(0.5,1.0,3.5,5.0), key="qa_stab_bumps")
            n_grid = st.multiselect("Discretization steps N", (20,40,60,80,100,125,150,175,200,250),
                                    default=(40,80,100,150), key="qa_stab_grid")
            threshold = _threshold("qa_stab_threshold")
            st.caption("Bump sweep: same spot bump for Delta and Gamma. N sweep: Delta ±1%; Gamma grid-aware. Repeated repricings can take time.")
            submitted = st.form_submit_button("RUN GREEKS STABILITY  →", type="primary", width="stretch")
    with right:
        st.markdown('<div class="desk-section">Risk stability / numerical diagnostics</div>', unsafe_allow_html=True)
        if submitted:
            st.session_state.pop("qa_stab_result", None)
            try:
                request = _get_request(fields, model, 100, threshold)
                with st.spinner("Repricing spot shocks across bumps and lattice sizes..."):
                    rows = greeks_stability(request, tuple(sorted(bump_grid)), tuple(sorted(n_grid)))
                st.session_state["qa_stab_result"] = (rows, request)
            except (PricingInputError, ValueError, RuntimeError, OverflowError) as exc:
                st.error(f"Greek stability unavailable: {exc}")
        saved = st.session_state.get("qa_stab_result")
        if saved is None:
            st.info("Run this diagnostic to inspect the dependence of Delta and Gamma on the spot bump and the lattice discretization.")
            return
        rows, request = saved
        frame = pd.DataFrame([{
            "dimension": r.dimension, "x": r.coordinate, "N": r.steps,
            "Delta": r.delta, "Gamma": r.gamma, "Gamma bump %": r.gamma_bump_pct,
            "status": r.status,
        } for r in rows])
        successes = frame[frame["status"] == "ok"]
        st.metric("Successful risk configurations", f"{len(successes)} / {len(frame)}")
        failures = frame[frame["status"] != "ok"]
        if not failures.empty:
            st.warning("Some risk stencils could not be priced; failures remain explicit in CSV.")
            with st.expander("Failed stencils"):
                st.dataframe(failures, hide_index=True, width="stretch")
        analytic = None
        try:
            from frontend.greeks_service import analytic_bs_greeks
            analytic = analytic_bs_greeks(request)
        except (ValueError, OverflowError, ZeroDivisionError):
            pass
        for dimension, title in (("spot bump", "Spot-bump sensitivity"),
                                 ("steps", "Stability with N")):
            subset = successes[successes["dimension"] == dimension]
            if subset.empty:
                continue
            st.markdown(f'<div class="desk-section">{title}</div>', unsafe_allow_html=True)
            for greek in ("Delta", "Gamma"):
                st.altair_chart(_line_chart(subset.assign(series=greek), x="x", y=greek,
                                           group="series", x_label=("Spot bump (%)" if dimension=="spot bump" else "N"),
                                           y_label=greek, height=190), width="stretch")
                if analytic:
                    st.caption(f"Black–Scholes {greek}: {analytic[greek]:.6f} (analytic comparator, not used for bumps).")
        st.caption("N-dependent Gamma uses a new grid-aware bump at each N; the two dimensions vary together. To isolate N, rerun the analysis with a fixed manual Gamma bump in the Greeks snapshot.")
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button("Export Greek stability · CSV", stability_csv(rows),
                           file_name="lattice272_greek_stability.csv", mime="text/csv", width="stretch")
