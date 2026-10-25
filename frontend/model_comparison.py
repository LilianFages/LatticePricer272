"""Model Comparison desk workspace (module 02).

Launch the existing Streamlit application from the repository root:
    python -m streamlit run frontend/app.py
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


import altair as alt
import pandas as pd
import streamlit as st

from frontend.diagnostics_fields import (
    render_market_fields, render_option_pair, render_reference_import,
)
from frontend.comparison_service import (
    ComparisonReport,
    compare_models,
    comparison_csv,
    display_rows,
)
from frontend.diagnostics_contract import import_saved_reference
from frontend.workspace_policy import COMPARISON_THRESHOLDS
from frontend.dividend_input import parse_dividends
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


def apply_reference_to_comparison(
    state: MutableMapping[str, Any],
) -> tuple[bool, tuple[str, ...]]:
    """Explicit import without pricing or changes to benchmark choices."""
    return import_saved_reference(state, "Compare models")


def comparison_dividend_rows(seed: PricingRequest | None) -> list[dict[str, float]]:
    """Present every saved cash dividend; preserve the default empty UI row."""
    if seed is None or not seed.dividends:
        return [{"time": 0.5, "amount": 0.0}]
    return [{"time": div.time, "amount": div.amount} for div in seed.dividends]


def _comparison_contract_key(base: str) -> str:
    """Recreate only imported contract widgets, never benchmark selections."""
    generation = st.session_state.get("cmp_import_nonce", 0)
    return f"{base}_{generation}" if generation else base


def _comparison_default(field: str, fallback: Any) -> Any:
    seed = st.session_state.get("cmp_reference_seed")
    return getattr(seed, field) if isinstance(seed, PricingRequest) else fallback


def _render_comparison_import() -> None:
    """Explicit ticket prefill; pricing remains button-triggered."""
    render_reference_import("Compare models")





def _comparison_contract_fields() -> tuple:
    """Contract-specific choices; the option editor is shared with the explorer."""
    st.markdown('<div class="desk-section">01 · Contract</div>', unsafe_allow_html=True)
    kind, exercise = render_option_pair(
        _comparison_contract_key, _comparison_default,
        widget_names=("cmp_kind", "cmp_exercise"),
    )
    models = tuple(st.multiselect(
        "Models to compare", MODELS, default=DEFAULT_MODELS,
        key="cmp_models", help="All selected models price the same contract."
    ))
    return kind, exercise, models


def _comparison_market_fields() -> tuple:
    """Common financial fields; preserve comparison widget IDs and limits."""
    return render_market_fields(
        _comparison_contract_key, _comparison_default,
        widget_names=("cmp_spot", "cmp_strike", "cmp_vol", "cmp_rate", "cmp_maturity"),
        heading="02 · Market data", spot_max=1e7,
        strike_max=1e7, maturity_max=100.0,
    )


def _comparison_benchmark_fields() -> tuple:
    """Set independent timing runs and price reference."""
    st.markdown('<div class="desk-section">03 · Benchmark setup</div>', unsafe_allow_html=True)
    steps = st.number_input("Time steps (N)", min_value=1, max_value=MAX_STEPS,
                            value=int(_comparison_default("steps", 100)),
                            step=25, key=_comparison_contract_key("cmp_steps"))
    a, b = st.columns(2)
    threshold = a.selectbox(
        "Pruning threshold", COMPARISON_THRESHOLDS,
        index=COMPARISON_THRESHOLDS.index(
            _comparison_default("pruning_threshold", DEFAULT_THRESHOLD)),
        format_func=lambda x: "0 (off)" if x == 0 else f"{x:.0e}",
        # Separate chart presentation from result selection.
        key=_comparison_contract_key("cmp_threshold"),
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
            pd.DataFrame(comparison_dividend_rows(
                st.session_state.get("cmp_reference_seed"))),
            num_rows="dynamic", hide_index=True, width="stretch",
            key=_comparison_contract_key("cmp_dividend_table"),
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
                    st.session_state["diagnostics_last_contract"] = request
                    st.session_state["diagnostics_last_source"] = "Compare models"
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
                "Select Explore lattice in Model Diagnostics to inspect "
                "actual priced nodes on a small grid."
            )


def render_comparison_page(*, embedded: bool = False) -> None:
    """Render cross-model analysis, optionally inside Model Diagnostics."""
    if "last_comparison_report" not in st.session_state:
        st.session_state["last_comparison_report"] = None

    if not embedded:
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
        _render_comparison_import()
        submitted, request, models, repetitions, reference, error = comparison_form()
    _render_comparison_output_column(right, submitted, request, models, repetitions, reference, error)

    if not embedded:
        st.divider()
        st.caption(
            "LatticePricer272 · University quant project · User-supplied prices. "
            "No live market data, calibration or trading connectivity."
        )
