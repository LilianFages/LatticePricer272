"""Read-only presentation of the independently executed dividend-grid study.

No pricing calls, no implicit experiment launch on Streamlit reruns. Missing
study exports produce an informative empty state, not generated demo numbers.
"""
from __future__ import annotations

import math
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "results" / "data" / "dividend_grid_robustness.csv"
REQUIRED = frozenset({
    "panel", "model", "steps", "date_shift_days", "dividend_time",
    "status", "european_put", "american_put", "european_call",
    "early_exercise_premium", "parity_residual", "gap_to_finest_sample",
    "finest_reference_steps", "date_shift_put", "failure",
})


def validate_study_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject incompatible exports without filling failed/missing observations."""
    # Use recorded observations only; failed configurations are never imputed.
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing phase 6C columns: {', '.join(sorted(missing))}")
    if not frame["panel"].isin(("grid", "date")).all():
        raise ValueError("Unknown phase 6C study panel.")
    if not frame["status"].isin(("success", "failed")).all():
        raise ValueError("Unknown phase 6C run status.")
    numeric = ("steps", "date_shift_days", "dividend_time", "european_put",
               "american_put", "european_call", "early_exercise_premium",
               "parity_residual", "gap_to_finest_sample", "finest_reference_steps",
               "date_shift_put")
    verified = frame.copy()
    for field in numeric:
        verified[field] = pd.to_numeric(verified[field], errors="coerce")
    good = verified["status"].eq("success")
    if (verified.loc[good, ["steps", "dividend_time", "european_put",
                            "american_put", "european_call", "parity_residual",
                            "early_exercise_premium"]].isna().any().any()):
        raise ValueError("Successful phase 6C case has missing mandatory measures.")
    # Missing failed prices remain NaN, but infinite successful results or
    # malformed grid coordinates cannot be treated as recorded observations.
    if (verified["steps"].isna().any() or verified["date_shift_days"].isna().any()):
        raise ValueError("Phase 6C export has invalid grid coordinates.")
    _validate_grid_coordinates(verified, numeric)
    _validate_option_premiums(verified, good)
    return verified


def _validate_grid_coordinates(
    verified: pd.DataFrame, numeric: tuple[str, ...]
) -> None:
    """Require finite, integral grid indices and valid ex-dividend dates."""
    steps = verified["steps"]
    shifts = verified["date_shift_days"]
    if (not steps.map(math.isfinite).all() or not shifts.map(math.isfinite).all()
            or not steps.between(1, 350).all()
            or not steps.eq(steps.round()).all()
            or not shifts.eq(shifts.round()).all()):
        raise ValueError("Phase 6C export has invalid lattice steps or day shifts.")
    dates = verified["dividend_time"].dropna()
    if not dates.map(math.isfinite).all() or not dates.between(0, 1, inclusive="right").all():
        # Use recorded observations only; failed configurations are never imputed.
        raise ValueError("Phase 6C export contains invalid ex-dates.")
    for field in numeric:
        nonmissing = verified[field].dropna()
        if not nonmissing.map(math.isfinite).all():
            raise ValueError(f"Phase 6C export contains non-finite {field}.")
    if verified.duplicated(subset=["panel", "model", "steps", "date_shift_days"]).any():
        raise ValueError("Duplicate Phase 6C configuration in recorded evidence.")


def _validate_option_premiums(verified: pd.DataFrame, good: pd.Series) -> None:
    """Preserve parity and early-exercise consistency for successful rows."""
    if (verified.loc[good, ["european_put", "american_put", "european_call"]]
            .lt(-1e-10).any().any()):
        raise ValueError("Negative successful option premium in Phase 6C evidence.")
    premium_gap = (verified.loc[good, "american_put"]
                   - verified.loc[good, "european_put"]
                   - verified.loc[good, "early_exercise_premium"])
    if premium_gap.abs().gt(1e-8).any():
        raise ValueError("Inconsistent Phase 6C early-exercise premium.")


def _fixed_date_chart(grid: pd.DataFrame) -> alt.Chart:
    """Preserve the recorded-series encodings and financial chart semantics."""
    # Use recorded observations only; failed configurations are never imputed.
    chart = (alt.Chart(grid).mark_line(point=True)
             .encode(x=alt.X("steps:Q", title="Lattice steps N"),
                     y=alt.Y("european_put:Q", title="European put (€)",
                             scale=alt.Scale(zero=False)),
                     color=alt.Color(
                         "model:N", title=None,
                         scale=alt.Scale(
                             domain=["Hybrid", "Trinomial · Allocation pruning"],
                             range=["#79BCE8", "#E0AE74"],
                         ),
                         legend=alt.Legend(
                             orient="bottom", direction="horizontal",
                             columns=2, labelLimit=330,
                         ),
                     ),
                     tooltip=["model:N", "steps:Q", "dividend_time:Q",
                              alt.Tooltip("european_put:Q", format=".7f"),
                              alt.Tooltip("gap_to_finest_sample:Q", format="+.7f")]))
    return chart.properties(height=325)


def _ex_date_shift_chart(dates: pd.DataFrame) -> alt.Chart:
    """Preserve the recorded-series encodings and financial chart semantics."""
    # Use recorded observations only; failed configurations are never imputed.
    chart = (alt.Chart(dates).mark_line(point=True)
             .encode(x=alt.X("date_shift_days:Q", title="Ex-date shift (days)"),
                     y=alt.Y("date_shift_put:Q", title="European put change (€)",
                             scale=alt.Scale(zero=False)),
                     color=alt.Color(
                         "model:N", title=None,
                         scale=alt.Scale(
                             domain=["Hybrid", "Trinomial · Allocation pruning"],
                             range=["#79BCE8", "#E0AE74"],
                         ),
                         legend=alt.Legend(
                             orient="bottom", direction="horizontal",
                             columns=2, labelLimit=330,
                         ),
                     ),
                     tooltip=["model:N", "date_shift_days:Q", "steps:Q",
                              alt.Tooltip("date_shift_put:Q", format="+.7f")]))
    return chart.properties(height=325)


def _render_fixed_date_panel(good: pd.DataFrame) -> None:
    """Compare numerical grids for the same cash-dividend contract."""
    # Use recorded observations only; failed configurations are never imputed.
    st.markdown("**A · Fixed ex-date, different N**")
    st.caption("Same spot, strike, rate, volatility, maturity and cash ex-date. "
               "Connecting the measured points is a visual guide, not a smooth "
               "interpolated convergence curve.")
    grid = good.loc[good["panel"].eq("grid")]
    if not grid.empty:
        st.altair_chart(_fixed_date_chart(grid), width="stretch")
        st.caption("Gap to finest sampled N uses the largest *successful* N for "
                   "each model; this is not a proven reference or true error.")
    else:
        st.warning("No successful fixed-ex-date grid observations.")


def _render_ex_date_shift_panel(good: pd.DataFrame) -> None:
    """Compare genuinely shifted ex-dates at a fixed lattice size."""
    st.markdown("**B · Fixed N, actual ex-date moved**")
    st.caption("Moving the ex-date changes the contract and its discounted "
               "dividend. The resulting price shift cannot be interpreted as "
               "discretization error.")
    dates = good.loc[good["panel"].eq("date") & good["date_shift_put"].notna()]
    # Use recorded observations only; failed configurations are never imputed.
    if not dates.empty:
        st.altair_chart(_ex_date_shift_chart(dates), width="stretch")
    else:
        st.warning("No comparable ex-date-shift observations.")


def _render_financial_consistency(good: pd.DataFrame) -> None:
    """Display observed put-call parity and American exercise consistency."""
    with st.expander("Financial consistency · Parity and early exercise", expanded=False):
        st.caption("Put–call parity is checked only for European options with "
                   "deterministic cash dividends. The American put premium "
                   "should be nonnegative; departures are reported rather than hidden.")
        if not good.empty:
            summary = good.groupby("model", as_index=False).agg(
                maximum_absolute_parity_residual=("parity_residual", lambda x: x.abs().max()),
                minimum_american_premium=("early_exercise_premium", "min"),
                observations=("status", "size"),
            )
            st.dataframe(summary, hide_index=True, width="stretch")


def _render_failed_cases(failures: pd.DataFrame) -> None:
    """Expose every failed experiment instead of dropping its row."""
    # Use recorded observations only; failed configurations are never imputed.
    if not failures.empty:
        with st.expander(f"Failed configurations · {len(failures)}", expanded=False):
            st.dataframe(failures[["model", "panel", "steps", "date_shift_days", "failure"]],
                         hide_index=True, width="stretch")


def _render_recorded_export(frame: pd.DataFrame) -> None:
    """Offer the unchanged validated frame and its UTF-8 BOM CSV export."""
    with st.expander("All measured observations and CSV", expanded=False):
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button("Export grid robustness · CSV",
                           frame.to_csv(index=False).encode("utf-8-sig"),
                           file_name="lattice272_dividend_grid_robustness.csv",
                           mime="text/csv", width="stretch")


def _load_grid_study() -> pd.DataFrame | None:
    """Read the archived dataset only, rejecting missing or incompatible exports."""
    if not DATASET.is_file():
        st.info("No full Phase 6C dataset yet. From the repository root, run "
                "`python -m experiments.dividends.dividend_grid_robustness` after the quick smoke test. "
                "This view never runs pricing automatically.")
        return None
    # Use recorded observations only; failed configurations are never imputed.
    try:
        frame = validate_study_frame(pd.read_csv(DATASET))
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        st.error(f"Phase 6C dataset is unavailable or incompatible: {exc}")
        return None
    if frame.empty:
        st.info("The Phase 6C dataset contains no observations.")
        return None
    return frame


def render_grid_robustness() -> None:
    """Display verified, previously saved experimental evidence on demand."""
    st.markdown('<div class="desk-section">Dividend ex-date · Grid robustness</div>',
                unsafe_allow_html=True)
    st.caption(
        "Recorded study, not live pricing. Separate fixed-contract discretization "
        "changes from genuine ex-date changes. Numerical reference gaps are not "
        "errors against an analytical price."
    )
    # Use recorded observations only; failed configurations are never imputed.
    frame = _load_grid_study()
    if frame is None:
        return

    good = frame.loc[frame["status"].eq("success")].copy()
    failures = frame.loc[frame["status"].eq("failed")]
    a, b, c = st.columns(3)
    a.metric("Recorded configurations", len(frame))
    b.metric("Successful price triplets", len(good))
    c.metric("Unpriced triplets", len(failures))

    # Display only recorded and validated observations; do not launch pricing.
    _render_fixed_date_panel(good)
    _render_ex_date_shift_panel(good)
    _render_financial_consistency(good)
    _render_failed_cases(failures)
    _render_recorded_export(frame)
