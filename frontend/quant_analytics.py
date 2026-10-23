"""Quant Analytics workspace: finite-difference Greeks and BS diagnostics.

The numerical engines and trade validation remain in frontend.pricing_service.
"""

from __future__ import annotations

import math

import altair as alt
import pandas as pd
import streamlit as st

from frontend.greeks_service import GreekBumps, GreeksResult, calculate_greeks, greeks_csv
from frontend.dividend_input import parse_dividends
from frontend.pricing_service import (
    DEFAULT_THRESHOLD, MAX_STEPS, MODELS,
    PricingInputError, PricingRequest,
)


GREEK_METADATA = (
    ("Delta", "€ premium / € spot", "Central spot derivative"),
    ("Gamma", "€ premium / €² spot", "Central spot curvature"),
    ("Vega / 1pp", "€ per +1 volatility pp", "Derivative rescaled to 1 vol percentage point"),
    ("Theta / 1d", "€ for one calendar day", "Fixed spot; reduce maturity and shift dividend dates"),
    ("Rho / 1bp", "€ per +1 rate bp", "Derivative rescaled to 1 basis point"),
)


def _display_num(value: float | None, *, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if value != 0.0 and abs(value) < 10 ** (-digits):
        return f"{value:+.3e}"
    return f"{value:+.{digits}f}"



def _greek_contract_fields() -> tuple:
    """Collect option exercise and engine settings."""
    st.markdown('<div class="desk-section">01 · Contract</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    kind = c1.selectbox("Option", ("Call", "Put"), key="greeks_option")
    exercise = c2.selectbox("Exercise", ("European", "American"), key="greeks_exercise")
    model = st.selectbox("Pricing engine", MODELS, index=3, key="greeks_model")
    return kind, exercise, model



def _greek_market_fields() -> tuple:
    """Collect spot strike volatility rates and maturity."""
    st.markdown('<div class="desk-section">02 · Market data</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    spot = c1.number_input("Spot · €", min_value=0.01, max_value=1e7,
                           value=100.0, format="%.2f", key="greeks_spot")
    strike = c2.number_input("Strike · €", min_value=0.0, max_value=1e7,
                             value=100.0, format="%.2f", key="greeks_strike")
    c1, c2 = st.columns(2)
    vol_pct = c1.number_input("Volatility · %", min_value=0.0, max_value=500.0,
                              value=20.0, format="%.2f", key="greeks_vol")
    rate_pct = c2.number_input("Risk-free rate · %", min_value=-50.0,
                               # Keep the Greek bump controls and input validation inside one ticket.
                               max_value=100.0, value=2.0, format="%.2f",
                               key="greeks_rate")
    maturity = st.number_input("Maturity · years", min_value=0.01,
                               max_value=100.0, value=1.0, format="%.2f",
                               key="greeks_maturity")
    return spot, strike, vol_pct, rate_pct, maturity



def _greek_lattice_fields(model: str) -> tuple:
    """Collect numerical depth and pruning settings."""
    st.markdown('<div class="desk-section">03 · Lattice & perturbations</div>', unsafe_allow_html=True)
    steps = st.number_input("Time steps (N)", min_value=1, max_value=MAX_STEPS,
                            value=100, step=25, key="greeks_steps")
    threshold = DEFAULT_THRESHOLD
    if model in ("Trinomial · Branch pruning", "Trinomial · Allocation pruning"):
        threshold = st.selectbox(
            "Pruning threshold", (0.0, 1e-18, 1e-15, 1e-12, 1e-5),
            index=1, key="greeks_threshold",
            format_func=lambda x: "0 (off)" if x == 0 else f"{x:.0e}",
        )
    return steps, threshold



def _greek_bump_fields() -> tuple:
    """Collect unchanged finite difference perturbation controls."""
    with st.expander("Greek perturbations · advanced", expanded=False):
        # Preserve numerical bump selection and advanced controls.
        st.caption("Perturbations control numerical differentiation; Greek reporting units are fixed.")
        c1, c2, c3 = st.columns(3)
        spot_bump_pct = c1.selectbox("Delta spot ± %", (0.5, 1.0, 2.0), index=1)
        gamma_choice = c2.selectbox(
            "Gamma spot ± %", ("Auto · grid-aware", 1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 7.5, 10.0),
            index=0, help="Default Gamma shock spans approximately one lattice level; "
            "Delta keeps its separate small spot bump.",
        )
        c1, c2 = st.columns(2)
        vol_bump_pp = c1.selectbox("Vol ± pp", (0.25, 0.5, 1.0), index=1)
        rate_bump_bp = c3.selectbox("Rate ± bp", (5.0, 10.0, 25.0), index=1)
    return spot_bump_pct, vol_bump_pp, rate_bump_bp, gamma_choice



def _greek_dividend_fields() -> object:
    """Collect optional cash dividend schedule."""
    with st.expander("Cash dividends · optional"):
        st.caption("Dates in years from pricing date; amounts in € per share. Add rows as needed.")
        dividend_table = st.data_editor(
            pd.DataFrame([{"time": 0.5, "amount": 0.0}]),
            num_rows="dynamic", hide_index=True, width="stretch",
            # Maintain stable form keys across Streamlit reruns.
            key="greeks_dividends",
            column_config={
                "time": st.column_config.NumberColumn(
                    "Payment time · years", min_value=0.0, max_value=100.0,
                    format="%.4f", required=False,
                ),
                "amount": st.column_config.NumberColumn(
                    "Cash amount · €", min_value=0.0, max_value=1e6,
                    format="%.2f", required=False,
                ),
            },
        )
    return dividend_table


def _greeks_ticket():
    """Collect form values; return (submitted, request, bumps, error)."""
    with st.form("greeks_ticket"):
        kind, exercise, model = _greek_contract_fields()

        spot, strike, vol_pct, rate_pct, maturity = _greek_market_fields()

        steps, threshold = _greek_lattice_fields(model)
        spot_bump_pct, vol_bump_pp, rate_bump_bp, gamma_choice = _greek_bump_fields()

        dividend_table = _greek_dividend_fields()
        submitted = st.form_submit_button("CALCULATE GREEKS  →", type="primary", width="stretch")

    if not submitted:
        return False, None, None, None
    try:
        # Render Greeks with their correct dimensions and sign conventions.
        request = PricingRequest(
            spot=float(spot), strike=float(strike),
            volatility=float(vol_pct)/100, rate=float(rate_pct)/100,
            maturity=float(maturity), steps=int(steps),
            option_type=kind, exercise=exercise, model=model,
            pruning_threshold=threshold, dividends=parse_dividends(dividend_table),
        )
        bumps = GreekBumps(
            spot_pct=float(spot_bump_pct), vol_pp=float(vol_bump_pp),
            rate_bp=float(rate_bump_bp),
            gamma_spot_pct=(
                None if gamma_choice == "Auto · grid-aware" else float(gamma_choice)
            ),
        )
        return True, request, bumps, None
    except (ValueError, PricingInputError) as exc:
        return True, None, None, str(exc)


def _shock_chart(result: GreeksResult, field: str, baseline: float, h: float):
    """Plot already-computed prices; no extra pricing calls on UI rerun."""
    labels = (f"{field}-", "base", f"{field}+")
    x_vals = (baseline-h, baseline, baseline+h)
    rows = [
        {"parameter": x, "price": result.scenario_prices[name], "scenario": name}
        for x, name in zip(x_vals, labels)
        if name in result.scenario_prices
    ]
    if len(rows) < 2:
        return None
    frame = pd.DataFrame(rows)
    return (
        alt.Chart(frame)
        .mark_line(color="#89CDF2", point=alt.OverlayMarkDef(size=85, color="#89CDF2"), strokeWidth=2.3)
        .encode(
            x=alt.X("parameter:Q", title=field.capitalize(), scale=alt.Scale(zero=False)),
            y=alt.Y("price:Q", title="Option value (€)", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("scenario:N"),
                     # Keep the existing desk-style visual design unchanged.
                     alt.Tooltip("parameter:Q", format=".6f"),
                     alt.Tooltip("price:Q", format=".7f")],
        )
        .properties(height=220)
    )



def _render_greek_header(result: GreeksResult) -> None:
    """Keep premium and runtime overview identical."""
    st.markdown('<div class="desk-section">Last Greek snapshot</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="desk-ticket"><div class="name">Base option premium · EUR</div>'
        f'<div class="price">{result.price:.6f} €</div>'
        f'<div class="description">{result.request.option_type} · {result.request.exercise}'
        f' · {result.request.model} · N={result.request.steps}</div></div>',
        unsafe_allow_html=True,
    )
    a, b, c = st.columns(3)
    # Keep market-input validation separate from risk diagnostics.
    a.metric("Engine pricings", str(result.evaluations))
    b.metric("Summed engine runtime", f"{result.total_runtime_ms:.1f} ms")
    c.metric("Reference", "Black–Scholes" if result.analytic else "Numerical only")



def _render_greek_metrics(result: GreeksResult) -> object:
    """Show the existing Greek sensitivity metrics."""
    st.markdown('<div class="desk-section">Risk sensitivities</div>', unsafe_allow_html=True)
    st.caption(
        f"Spot shocks: Delta ±{result.bumps.spot_pct:g}% · "
        f"Gamma ±{result.gamma_spot_bump_pct:.3f}% ({result.gamma_scheme}). "
        "Gamma is sensitive to grid resolution; compare across N and bumps."
    )
    greeks = result.measured()
    main = [*st.columns(3), *st.columns(2)]
    for (name, unit, _), target in zip(GREEK_METADATA, main):
        with target:
            st.metric(name, _display_num(greeks[name]))
            st.caption(unit)
    return greeks



def _render_greek_table(result: GreeksResult, greeks: dict) -> None:
    """Compare measured Greeks against analytic Black Scholes."""
    data = []
    for name, unit, formula in GREEK_METADATA:
        value = greeks[name]
        bs = result.analytic.get(name) if result.analytic is not None else None
        data.append({
            "Greek": name, "Model": None if value is None else round(value, 9),
            "Black–Scholes": None if bs is None else round(bs, 9),
            "Model − BS": None if value is None or bs is None else round(value-bs, 9),
            "Unit": unit,
        })
    with st.expander("Detailed risk table · numerical vs analytical", expanded=True):
        st.dataframe(pd.DataFrame(data), hide_index=True, width="stretch")
        st.caption("Black–Scholes Greeks only for European, no cash dividends, positive volatility and strike.")



def _render_greek_scenarios(result: GreeksResult) -> None:
    """Render the current shock visualization ladder."""
    with st.expander("Perturbation ladder · repriced scenarios", expanded=False):
        request, bump = result.request, result.bumps
        a, b = st.columns(2, gap="large")
        with a:
            # Maintain stable form keys across Streamlit reruns.
            st.caption(f"Spot ±{bump.spot_pct:g}% · Delta (local)")
            chart = _shock_chart(result, "spot", request.spot, request.spot*bump.spot_pct/100)
            if chart is not None:
                st.altair_chart(chart, width="stretch")
            if not math.isclose(result.gamma_spot_bump_pct, bump.spot_pct, rel_tol=1e-12):
                st.caption(f"Gamma uses a separate ±{result.gamma_spot_bump_pct:.3f}% shock")
                gamma_chart = _shock_chart(
                    result, "spot_gamma", request.spot,
                    request.spot*result.gamma_spot_bump_pct/100,
                )
                if gamma_chart is not None:
                    st.altair_chart(gamma_chart, width="stretch")
        with b:
            st.caption(f"Volatility ±{bump.vol_pp:g}pp · Vega")
            chart = _shock_chart(result, "volatility", request.volatility, bump.vol_pp/100)
            if chart is not None:
                st.altair_chart(chart, width="stretch")
        # Render Greeks with their correct dimensions and sign conventions.
        st.caption(
            "Fixed steps N, fresh tree for every shock. For small volatility a second-order "
            "forward bump replaces the infeasible negative-volatility shock."
        )



def _render_greek_methodology(result: GreeksResult) -> None:
    """Explain finite differences and retain warning order."""
    with st.expander("Methodology & warnings", expanded=bool(result.issues)):
        st.markdown(
            "**Delta:** local central spot bump. **Gamma:** independent central spot bump, "
            "grid-aware by default to avoid spurious zero curvature in finite lattices. "
            "**Vega:** central volatility bumps "
            "(or forward second-order near zero). **Rho:** central rate bumps. "
            "**Theta:** price tomorrow minus price today with the *same calendar expiry*, "
            "fixed spot and ex-dates shifted one day closer. "
            "Near a dividend ex-date or expiry, Theta is unavailable rather than misleading. "
            "Discrete-tree Greeks are bump-dependent and can oscillate with N."
        )
        # Keep the existing desk-style visual design unchanged.
        st.caption(f"Vega scheme: {result.volatility_scheme}. "
                   f"Bumps: Delta spot ±{result.bumps.spot_pct:g}%, "
                   f"Gamma spot ±{result.gamma_spot_bump_pct:.3f}%, "
                   f"vol ±{result.bumps.vol_pp:g} pp, rate ±{result.bumps.rate_bp:g} bp.")
        for issue in result.issues:
            st.warning(issue)


def _result_panel(result: GreeksResult) -> None:
    _render_greek_header(result)

    greeks = _render_greek_metrics(result)

    # Preserve numerical bump selection and advanced controls.
    _render_greek_table(result, greeks)
    _render_greek_scenarios(result)
    _render_greek_methodology(result)

    st.download_button(
        "Export Greek risk snapshot · CSV", greeks_csv(result),
        file_name="lattice272_greeks.csv", mime="text/csv", width="stretch",
    )




def _update_quant_greeks(
    submitted: bool,
    request: PricingRequest | None,
    bumps: GreekBumps | None,
    error: str | None,
) -> None:
    """Run the same Greek repricing only after form submission."""
    if submitted:
        st.session_state["quant_last_greeks"] = None
        if error:
            st.error(error)
        elif request is not None and bumps is not None:
            try:
                with st.spinner("Repricing Greek scenarios..."):
                    st.session_state["quant_last_greeks"] = calculate_greeks(request, bumps)
            except (PricingInputError, ValueError, RuntimeError,
                    OverflowError, ArithmeticError) as exc:
                st.error(f"Greek calculation unavailable: {exc}")

def _render_quant_greeks_tab(greek_tab: object) -> None:
    """Render risk ticket and Greek results in same tab."""
    with greek_tab:
        left, right = st.columns([1.0, 1.45], gap="large")
        # Keep market-input validation separate from risk diagnostics.
        with left:
            submitted, request, bumps, error = _greeks_ticket()
        with right:
            _update_quant_greeks(submitted, request, bumps, error)
            result = st.session_state.get("quant_last_greeks")
            if result is not None:
                _result_panel(result)
            # Preserve numerical bump selection and advanced controls.
            elif not submitted:
                st.markdown('<div class="desk-section">Quantitative risk output</div>',
                            unsafe_allow_html=True)
                st.info("Select **CALCULATE GREEKS** to reprice the contract under controlled shocks.")
                st.caption("Delta · Gamma · Vega / 1 vol point · Theta / day · Rho / 1bp")


def render_quant_analytics_page() -> None:
    """Five explicit, independently triggered Quant Analytics workflows."""
    from frontend.analytics_views import (
        render_convergence_tab, render_greeks_stability_tab,
        render_sensitivity_tab,
    )

    st.markdown('<div class="desk-kicker">Quant analytics / 04</div>', unsafe_allow_html=True)
    st.markdown('<div class="desk-title">Quant Analytics · Pricing & Risk</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="desk-subtitle">Bump-transparent Greeks · repricing sensitivities · '
        'convergence & computational cost</div>', unsafe_allow_html=True,
    )
    greek_tab, sensitivity_tab, convergence_tab, efficiency_tab, stability_tab, exercise_tab = st.tabs(
        ["Greeks snapshot", "Market sensitivities", "Numerical accuracy",
         "Computational efficiency", "Risk numerics", "Optimal stopping"]
    )
    _render_quant_greeks_tab(greek_tab)
    with sensitivity_tab:
        render_sensitivity_tab()
    with convergence_tab:
        render_convergence_tab()
    with efficiency_tab:
        from frontend.research_efficiency_views import render_efficiency_tab
        render_efficiency_tab()
    with stability_tab:
        render_greeks_stability_tab()
    with exercise_tab:
        from frontend.exercise_boundary_views import render_exercise_boundary_tab
        render_exercise_boundary_tab()
    st.divider()
    st.caption("Research-grade local model diagnostics, not executable quotes or market-calibrated hedges.")
