"""Phase 6C: archived research view rejects malformed numerical evidence."""
from __future__ import annotations

import pandas as pd
import pytest

from frontend.research_grid_views import REQUIRED, validate_study_frame


def _valid() -> pd.DataFrame:
    return pd.DataFrame([{
        "panel": "grid", "model": "Hybrid", "steps": 25,
        "date_shift_days": 0, "dividend_time": .5, "status": "success",
        "european_put": 5., "american_put": 5.2, "european_call": 6.,
        "early_exercise_premium": .2, "parity_residual": .01,
        "gap_to_finest_sample": .02, "finest_reference_steps": 40,
        "date_shift_put": None, "failure": None,
    }])


def test_schema_and_null_optional_observations_allowed() -> None:
    frame = _valid()
    result = validate_study_frame(frame)
    assert result.loc[0, "status"] == "success"
    assert pd.isna(result.loc[0, "date_shift_put"])
    pd.testing.assert_frame_equal(frame, _valid())
    assert set(REQUIRED).issubset(result.columns)


def test_missing_scientific_column_rejected() -> None:
    with pytest.raises(ValueError, match="Missing phase 6C columns"):
        validate_study_frame(_valid().drop(columns=["parity_residual"]))


def test_malformed_success_not_silently_imputed() -> None:
    frame = _valid()
    frame.loc[0, "european_put"] = None
    with pytest.raises(ValueError, match="missing mandatory"):
        validate_study_frame(frame)


def test_invalid_panel_and_status_rejected() -> None:
    frame = _valid()
    frame.loc[0, "panel"] = "fictional"
    with pytest.raises(ValueError, match="panel"):
        validate_study_frame(frame)
    frame = _valid()
    frame.loc[0, "status"] = "almost"
    with pytest.raises(ValueError, match="status"):
        validate_study_frame(frame)


def test_failed_rows_may_have_unpriced_measures() -> None:
    frame = _valid()
    frame.loc[0, "status"] = "failed"
    frame.loc[0, "european_put"] = None
    assert validate_study_frame(frame).loc[0, "status"] == "failed"

# Final-release evidence checks reject malformed observations rather than
# letting Altair silently hide non-finite values or duplicate measurements.
@pytest.mark.parametrize("field,value,reason", [
    ("steps", 25.5, "lattice steps"),
    ("steps", float("inf"), "lattice steps"),
    ("date_shift_days", 0.25, "day shifts"),
    ("dividend_time", 1.5, "ex-dates"),
    ("european_put", float("inf"), "non-finite"),
    ("european_call", -0.3, "Negative"),
    ("early_exercise_premium", .5, "inconsistent"),
])
def test_phase6d_rejects_corrupted_evidence(field, value, reason) -> None:
    frame = _valid()
    if field in ("steps", "date_shift_days"):
        frame[field] = frame[field].astype(float)
    frame.loc[0, field] = value
    with pytest.raises(ValueError, match=f"(?i){reason}"):
        validate_study_frame(frame)


def test_phase6d_rejects_duplicate_recorded_case() -> None:
    frame = pd.concat([_valid(), _valid()], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_study_frame(frame)


def test_phase6d_accepts_numeric_failed_coordinates_without_prices() -> None:
    frame = _valid()
    frame.loc[0, "status"] = "failed"
    for field in ("european_put", "american_put", "european_call",
                  "early_exercise_premium", "parity_residual"):
        frame.loc[0, field] = float("nan")
    frame.loc[0, "failure"] = "model declined this lattice"
    assert validate_study_frame(frame).loc[0, "status"] == "failed"
