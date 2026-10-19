"""Model Comparison desk workspace (module 02).

Launch the existing Streamlit application from the repository root:
    python -m streamlit run frontend/app.py
"""

from __future__ import annotations

import math

import altair as alt
import pandas as pd
import streamlit as st

from frontend.comparison_service import (
    ComparisonReport,
    compare_models,
    comparison_csv,
    display_rows,
)
from frontend.pricing_service import (
    DEFAULT_THRESHOLD,
    MAX_STEPS,
    MODELS,
    DividendInput,
    PricingInputError,
    PricingRequest,
)

# Build comparable contracts from a shared market-input ticket.
DEFAULT_MODELS = (
    "Binomial",
    "Trinomial · Full",
    "Trinomial · Branch pruning",
    "Trinomial · Allocation pruning",
    "Hybrid",
)


def parse_dividends(frame: pd.DataFrame) -> tuple[DividendInput, ...]:
    """Read UI input only; engine validation occurs in comparison_service."""
    dividends = []
    for row_number, record in enumerate(frame.to_dict("records"), 1):
        raw_time = record.get("time")
        raw_amount = record.get("amount")
        if pd.isna(raw_amount):
            if not pd.isna(raw_time):
                raise PricingInputError(
                    f"Dividend row {row_number}: supply an amount or delete the row."
                )
            continue
        try:
            amount = float(raw_amount)
        except (TypeError, ValueError):
            raise PricingInputError("Dividend amounts must be numeric.") from None
        if not math.isfinite(amount):
            raise PricingInputError("Dividend amounts must be finite.")
        # Retain model-order and display labels in side-by-side comparisons.
        if amount == 0.0:
            continue
        if pd.isna(raw_time):
            raise PricingInputError(f"Dividend row {row_number}: supply a payment time.")
        try:
            dividend_time = float(raw_time)
        except (TypeError, ValueError):
            raise PricingInputError("Dividend times must be numeric.") from None
        dividends.append(DividendInput(time=dividend_time, amount=amount))
    return tuple(sorted(dividends, key=lambda d: d.time))



def _comparison_contract_fields() -> tuple:
    """Collect the common contract and selected pricing engines."""
    st.markdown('<div class="desk-section">01 · Contract</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    kind = a.selectbox("Option", ("Call", "Put"), key="cmp_kind")
    exercise = b.selectbox("Exercise", ("European", "American"), key="cmp_exercise")
    models = tuple(st.multiselect(
        "Models to compare", MODELS, default=DEFAULT_MODELS,
        key="cmp_models", help="All selected models price the same contract."
    ))
    return kind, exercise, models



def _comparison_market_fields() -> tuple:
    """Collect common market parameters."""
    st.markdown('<div class="desk-section">02 · Market data</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    spot = a.number_input("Spot · €", min_value=0.01, max_value=1e7,
                          value=100.0, step=1.0, format="%.2f", key="cmp_spot")
    strike = b.number_input("Strike · €", min_value=0.0, max_value=1e7,
                            value=100.0, step=1.0, format="%.2f", key="cmp_strike")
    a, b = st.columns(2)
    # Preserve dividend-editing behavior and Streamlit widget keys.
    vol_pct = a.number_input("Volatility · %", min_value=0.0, max_value=500.0,
                             value=20.0, step=1.0, format="%.2f", key="cmp_vol")
    rate_pct = b.number_input("Risk-free rate · %", min_value=-50.0,
                              max_value=100.0, value=2.0, step=0.25,
                              format="%.2f", key="cmp_rate")
    maturity = st.number_input("Maturity · years", min_value=0.01,
                               max_value=100.0, value=1.0, step=0.25,
                               format="%.2f", key="cmp_maturity")
    return spot, strike, vol_pct, rate_pct, maturity



def _comparison_benchmark_fields() -> tuple:
    """Set independent timing runs and price reference."""
    st.markdown('<div class="desk-section">03 · Benchmark setup</div>', unsafe_allow_html=True)
    steps = st.number_input("Time steps (N)", min_value=1, max_value=MAX_STEPS,
                            value=100, step=25, key="cmp_steps")
    a, b = st.columns(2)
    threshold = a.selectbox(
        "Pruning threshold", (0.0, 1e-18, 1e-15, 1e-12),
        index=1, format_func=lambda x: "0 (off)" if x == 0 else f"{x:.0e}",
        # Separate chart presentation from result selection.
        key="cmp_threshold",
    )
    repetitions = b.selectbox(
        "Timing repetitions", (1, 3, 5), index=1, key="cmp_repetitions",
        help="Median of independent runs. A fresh tree is built each time.",
    )
    reference = st.selectbox(
        "Price reference", ("Auto", "Black–Scholes", *MODELS),
        index=0, key="cmp_reference",
        help="Auto uses Black–Scholes when valid, otherwise the first "
             "successful selected model. A requested unavailable reference "
             "is reported as unavailable; not silently replaced.",
    )
    return steps, threshold, repetitions, reference



def _comparison_dividend_fields() -> object:
    """Collect the same cash dividend schedule."""
    with st.expander("Cash dividends · optional"):
        st.caption("Discrete payments: time in years, amount in EUR per share. "
                   "Multiple rows supported. Binomial will be marked unsupported.")
        frame = st.data_editor(
            # Build comparable contracts from a shared market-input ticket.
            pd.DataFrame([{"time": 0.5, "amount": 0.0}]),
            num_rows="dynamic", hide_index=True, width="stretch",
            key="cmp_dividend_table",
            column_config={
                "time": st.column_config.NumberColumn(
                    "Payment time · years", min_value=0.0,
                    max_value=100.0, format="%.4f", required=False,
                ),
                "amount": st.column_config.NumberColumn(
                    "Cash amount · €", min_value=0.0,
                    max_value=1e6, format="%.2f", required=False,
                ),
            },
        )
    return frame


def comparison_form() -> tuple[
    bool, PricingRequest | None, tuple[str, ...], int, str, str | None
]:
    """Collect one identical contract for all selected pricing models."""
    with st.form("model_comparison_ticket"):
        kind, exercise, models = _comparison_contract_fields()

        spot, strike, vol_pct, rate_pct, maturity = _comparison_market_fields()

        steps, threshold, repetitions, reference = _comparison_benchmark_fields()
        frame = _comparison_dividend_fields()
        submitted = st.form_submit_button(
            "RUN MODEL COMPARISON  →", type="primary", width="stretch"
        )
    # Retain model-order and display labels in side-by-side comparisons.
    if not submitted:
        return False, None, models, int(repetitions), reference, None
    try:
        dividends = parse_dividends(frame)
        request = PricingRequest(
            spot=float(spot), strike=float(strike),
            rate=float(rate_pct) / 100,
            volatility=float(vol_pct) / 100,
            maturity=float(maturity), steps=int(steps),
            option_type=kind, exercise=exercise,
            model="Trinomial · Allocation pruning",
            dividends=dividends, pruning_threshold=threshold,
        )
        return True, request, models, int(repetitions), reference, None
    except PricingInputError as exc:
        return True, None, models, int(repetitions), reference, str(exc)


def _bar_chart(rows: list[dict[str, object]], field: str) -> alt.Chart:
    """Interactive bar chart for runtime or signed price deviation."""
    frame = pd.DataFrame([row for row in rows if row[field] is not None])
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=3, color="#79C4EB")
        .encode(
            x=alt.X(field + ":Q", title=field, scale=alt.Scale(zero=True)),
            y=alt.Y("Model:N", sort=list(frame["Model"]), title=None),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip(field + ":Q", format="+.6f"),
            ],
        )
        .properties(height=max(170, len(frame) * 39))
    )



def _render_comparison_head(report: ComparisonReport, valid: list, failed: list) -> None:
    """Present aggregate model diagnostics and pricing failures."""
    a, b, c = st.columns(3)
    a.metric("Models priced", f"{len(valid)} / {len(report.lines)}")
    b.metric("Reference", report.reference_label or "Unavailable")
    c.metric("Reference premium", (
        f"{report.reference_price:.6f} €"
        if report.reference_price is not None else "N/A"
    ))
    if failed:
        with st.expander(f"{len(failed)} unavailable / failed model(s)", expanded=True):
            for line in failed:
                st.warning(f"**{line.model}** — {line.status}: {line.reason}")
    if report.reference_price is None and valid:
        st.warning("Requested price reference is unavailable. Price gaps are not computed.")



def _render_comparison_table(rows: list) -> None:
    """Keep detailed pricing table and format options unchanged."""
    st.markdown('<div class="desk-section">Pricing & performance</div>', unsafe_allow_html=True)
    table = pd.DataFrame(rows)[[
        "Model", "Status", "Price (€)", "Δ vs reference (€)",
        "Δ vs reference (%)", "Median time (ms)", "Reachable nodes",
    ]]
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            # Build comparable contracts from a shared market-input ticket.
            "Price (€)": st.column_config.NumberColumn(format="%.6f €"),
            "Δ vs reference (€)": st.column_config.NumberColumn(format="%+.6f €"),
            "Δ vs reference (%)": st.column_config.NumberColumn(format="%+.3f %%"),
            "Median time (ms)": st.column_config.NumberColumn(format="%.2f ms"),
            "Reachable nodes": st.column_config.NumberColumn(format="%d"),
        },
    )



def _render_comparison_charts(report: ComparisonReport, rows: list) -> None:
    """Render price deviation and computation time panels."""
    tab_price, tab_runtime = st.tabs(["Price deviation", "Runtime comparison"])
    with tab_price:
        if report.reference_price is not None:
            st.caption(f"Signed deviation from {report.reference_label}. Zero = reference price.")
            st.altair_chart(_bar_chart(rows, "Δ vs reference (€)"), width="stretch")
        else:
            st.caption("No usable price reference; showing independent model premiums.")
            st.altair_chart(_bar_chart(rows, "Price (€)"), width="stretch")
    with tab_runtime:
        st.altair_chart(_bar_chart(rows, "Median time (ms)"), width="stretch")
    # Retain model-order and display labels in side-by-side comparisons.
    st.caption(
        "Runtime: median of separate fresh-tree runs, timing construction and "
        "pricing. Differences under a few milliseconds can reflect CPU load. "
        "Displayed model prices are NOT executable quotes."
    )



def _render_comparison_diagnostics(rows: list) -> None:
    """Show model allocation and timing diagnostics."""
    with st.expander("Detailed diagnostics · timing range, pruning and nodes"):
        detail = pd.DataFrame(rows)[[
            "Model", "Time min (ms)", "Time max (ms)",
            "Reachable nodes", "Allocated nodes*", "Pruned transitions",
            "Avoided allocations (%)*",
        ]]
        st.dataframe(detail, hide_index=True, width="stretch")
        st.caption(
            "*Allocated nodes and allocation savings are reported only where "
            "the existing engine exposes exact allocations and a valid "
            "no-dividend full-trinomial baseline. Reachable is not allocated."
        )


def _comparison_outputs(report: ComparisonReport) -> None:
    # Preserve dividend-editing behavior and Streamlit widget keys.
    st.markdown('<div class="desk-section">Last model comparison</div>', unsafe_allow_html=True)
    rows = display_rows(report)
    valid = report.successful
    failed = report.unsuccessful
    _render_comparison_head(report, valid, failed)
    # Separate chart presentation from result selection.
    if not valid:
        st.error("No selected engine completed pricing. Check model/dividend compatibility.")
        st.download_button(
            "Export comparison · CSV", comparison_csv(report),
            file_name="lattice272_model_comparison.csv", mime="text/csv",
            width="stretch",
        )
        return

    _render_comparison_table(rows)
    _render_comparison_charts(report, rows)

    _render_comparison_diagnostics(rows)
    st.download_button(
        "Export comparison · CSV", comparison_csv(report),
        file_name="lattice272_model_comparison.csv", mime="text/csv",
        width="stretch",
    )




def _update_comparison_result(
    submitted: bool,
    error: str | None,
    request: PricingRequest | None,
    models: tuple,
    repetitions: int,
    reference: str,
) -> None:
    """Price requested models and update the current session report."""
    if submitted:
        st.session_state["last_comparison_report"] = None
        if error:
            st.error(error)
        elif request is not None:
            try:
                with st.spinner("Repricing the contract across selected models..."):
                    st.session_state["last_comparison_report"] = compare_models(
                        request, models, repetitions, reference
                    )
            except (PricingInputError, ValueError) as exc:
                st.error(str(exc))

def _render_comparison_output_column(
    right: object,
    submitted: bool,
    request: PricingRequest | None,
    models: tuple,
    repetitions: int,
    reference: str,
    error: str | None,
) -> None:
    """Update session comparison and render selected results."""
    with right:
        # Preserve dividend-editing behavior and Streamlit widget keys.
        _update_comparison_result(submitted, error, request, models, repetitions, reference)
        report = st.session_state["last_comparison_report"]
        if report is not None:
            _comparison_outputs(report)
        elif not submitted:
            st.markdown('<div class="desk-section">Comparison output</div>',
                        # Separate chart presentation from result selection.
                        unsafe_allow_html=True)
            st.info("Select engines and choose **RUN MODEL COMPARISON**.")
            st.caption(
                "Reference: Black–Scholes for European options without dividends; "
                "otherwise compare against a selected numerical engine."
            )
            st.caption(
                "The geometric lattice explorer (small N, real nodes and "
                "transitions) will be implemented next in module 2.2."
            )


def render_comparison_page() -> None:
    """Page 02; rendered through the existing frontend/app.py entrypoint."""
    if "last_comparison_report" not in st.session_state:
        st.session_state["last_comparison_report"] = None

    st.markdown('<div class="desk-kicker">Model comparison / 02</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="desk-title">Cross-Model · Pricing & Performance</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="desk-subtitle">One contract · multiple pricing engines · '
        'explicit reference and numerical diagnostics</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.0, 1.35], gap="large")
    with left:
        submitted, request, models, repetitions, reference, error = comparison_form()
    _render_comparison_output_column(right, submitted, request, models, repetitions, reference, error)

    st.divider()
    st.caption(
        "LatticePricer272 · University quant project · User-supplied prices. "
        "No live market data, calibration or trading connectivity."
    )
