"""Read-only presentation of the independently executed dividend-grid study.

No pricing calls, no implicit experiment launch on Streamlit reruns. Missing
study exports produce an informative empty state, not generated demo numbers.
"""
from __future__ import annotations

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
    if (verified["steps"].isna().any() or verified["date_shift_days"].isna().any()):
        raise ValueError("Phase 6C export has invalid grid coordinates.")
    return verified


def render_grid_robustness() -> None:
    """Display verified, previously saved experimental evidence on demand."""
    st.markdown('<div class="desk-section">Dividend ex-date · Grid robustness</div>',
                unsafe_allow_html=True)
    st.caption(
        "Recorded study, not live pricing. Separate fixed-contract discretization "
        "changes from genuine ex-date changes. Numerical reference gaps are not "
        "errors against an analytical price."
    )
    if not DATASET.is_file():
        st.info("No full Phase 6C dataset yet. From the repository root, run "
                "`python -m experiments.dividend_grid_robustness` after the quick smoke test. "
                "This view never runs pricing automatically.")
        return
    try:
        frame = validate_study_frame(pd.read_csv(DATASET))
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        st.error(f"Phase 6C dataset is unavailable or incompatible: {exc}")
        return
    if frame.empty:
        st.info("The Phase 6C dataset contains no observations.")
        return

    good = frame.loc[frame["status"].eq("success")].copy()
    failures = frame.loc[frame["status"].eq("failed")]
    a, b, c = st.columns(3)
    a.metric("Recorded configurations", len(frame))
    b.metric("Successful price triplets", len(good))
    c.metric("Unpriced triplets", len(failures))

    st.markdown("**A · Fixed ex-date, different N**")
    st.caption("Same spot, strike, rate, volatility, maturity and cash ex-date. "
               "Connecting the measured points is a visual guide, not a smooth "
               "interpolated convergence curve.")
    grid = good.loc[good["panel"].eq("grid")]
    if not grid.empty:
        chart = (alt.Chart(grid).mark_line(point=True)
                 .encode(x=alt.X("steps:Q", title="Lattice steps N"),
                         y=alt.Y("european_put:Q", title="European put (€)",
                                 scale=alt.Scale(zero=False)),
                         color=alt.Color("model:N", title="Engine"),
                         tooltip=["model:N", "steps:Q", "dividend_time:Q",
                                  alt.Tooltip("european_put:Q", format=".7f"),
                                  alt.Tooltip("gap_to_finest_sample:Q", format="+.7f")]))
        st.altair_chart(chart.properties(height=305), width="stretch")
        st.caption("Gap to finest sampled N uses the largest *successful* N for "
                   "each model; this is not a proven reference or true error.")
    else:
        st.warning("No successful fixed-ex-date grid observations.")

    st.markdown("**B · Fixed N, actual ex-date moved**")
    st.caption("Moving the ex-date changes the contract and its discounted "
               "dividend. The resulting price shift cannot be interpreted as "
               "discretization error.")
    dates = good.loc[good["panel"].eq("date") & good["date_shift_put"].notna()]
    if not dates.empty:
        chart = (alt.Chart(dates).mark_line(point=True)
                 .encode(x=alt.X("date_shift_days:Q", title="Ex-date shift (days)"),
                         y=alt.Y("date_shift_put:Q", title="European put change (€)",
                                 scale=alt.Scale(zero=False)),
                         color=alt.Color("model:N", title="Engine"),
                         tooltip=["model:N", "date_shift_days:Q", "steps:Q",
                                  alt.Tooltip("date_shift_put:Q", format="+.7f")]))
        st.altair_chart(chart.properties(height=305), width="stretch")
    else:
        st.warning("No comparable ex-date-shift observations.")

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
    if not failures.empty:
        with st.expander(f"Failed configurations · {len(failures)}", expanded=False):
            st.dataframe(failures[["model", "panel", "steps", "date_shift_days", "failure"]],
                         hide_index=True, width="stretch")
    with st.expander("All measured observations and CSV", expanded=False):
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button("Export grid robustness · CSV",
                           frame.to_csv(index=False).encode("utf-8-sig"),
                           file_name="lattice272_dividend_grid_robustness.csv",
                           mime="text/csv", width="stretch")
