"""Dividend-grid robustness experiments and related chart views.

Consolidated without dropping individual regression cases.
"""

from __future__ import annotations

from types import SimpleNamespace
import pytest
from experiments import dividend_grid_robustness as study
from src.dividend_timing import dividend_grid_step
import pandas as pd
from frontend.research_grid_views import REQUIRED, validate_study_frame


# test_dividend_grid_robustness.py (tests preserved)
def _fake_price(request):
    """Deterministic pricing spy: no dependence on expensive lattice builds."""
    date = request.dividends[0].time
    forward_term = study.parity_target(date)
    put = 6.0 + 2.0 / request.steps + (date - .5) * 8.0
    price = put
    if request.option_type == "Call":
        price = put + forward_term
    elif request.exercise == "American":
        price = put + 0.4
    return SimpleNamespace(price=price)


def test_two_independent_axes_do_not_confuse_shift_with_grid() -> None:
    rows = study.run_experiment(steps=(25, 40), shifts=(-1, 0, 1),
                                date_steps=40, pricer=_fake_price,
                                models=(study.MODELS[0],))
    assert len(rows) == 5
    fixed = [r for r in rows if r["panel"] == "grid"]
    shifted = [r for r in rows if r["panel"] == "date"]
    assert {r["dividend_time"] for r in fixed} == {0.5}
    assert {r["steps"] for r in shifted} == {40}
    assert sorted(r["date_shift_days"] for r in shifted) == [-1, 0, 1]
    assert all(r["status"] == "success" for r in rows)


def test_parity_premium_and_finest_sample_gap_are_defined() -> None:
    rows = study.run_experiment(steps=(25, 40), shifts=(0,), date_steps=40,
                                pricer=_fake_price, models=(study.MODELS[0],))
    grid = {r["steps"]: r for r in rows if r["panel"] == "grid"}
    assert grid[25]["gap_to_finest_sample"] == pytest.approx(2/25 - 2/40)
    assert grid[40]["gap_to_finest_sample"] == pytest.approx(0.0)
    assert all(r["finest_reference_steps"] == 40 for r in grid.values())
    assert all(r["early_exercise_premium"] == pytest.approx(0.4) for r in rows)
    assert all(r["parity_residual"] == pytest.approx(0.0, abs=1e-12) for r in rows)
    assert rows[-1]["date_shift_put"] == pytest.approx(0.0)


def test_per_model_reference_does_not_mix_lattices() -> None:
    rows = study.run_experiment(steps=(25, 40), shifts=(0,), date_steps=40,
                                pricer=_fake_price)
    assert len(rows) == 6
    assert {r["model"] for r in rows} == set(study.MODELS)
    assert all(r["finest_reference_steps"] == 40
               for r in rows if r["panel"] == "grid")


def test_failed_triplet_is_explicit_and_never_imputed() -> None:
    def failing(request):
        if request.exercise == "American":
            raise RuntimeError("Injected pricer outage")
        return _fake_price(request)
    rows = study.run_experiment(steps=(25,), shifts=(0,), date_steps=25,
                                pricer=failing, models=(study.MODELS[0],))
    assert len(rows) == 2
    assert all(r["status"] == "failed" for r in rows)
    assert all(r["european_put"] is None for r in rows)
    assert all("Injected pricer outage" in r["failure"] for r in rows)
    assert all(r["gap_to_finest_sample"] is None for r in rows)


def test_finite_prices_required() -> None:
    def invalid(request):
        return SimpleNamespace(price=float("nan"))
    row = study.measure_case(study.MODELS[0], 25, 0.5, pricer=invalid)
    assert row["status"] == "failed"
    assert "Non-finite" in row["failure"]


@pytest.mark.parametrize("steps,days,date_steps", [
    ((), (0,), 100), ((25, 25), (0,), 100),
    ((0,), (0,), 100), ((25,), (1,), 100),
    ((25,), (0, 0), 100), ((25,), (0,), 0),
])
def test_invalid_designs_fail_loudly(steps, days, date_steps) -> None:
    with pytest.raises(ValueError):
        study.validate_design(steps, days, date_steps)


def test_exdate_boundary_convention_and_dividend_pv() -> None:
    assert dividend_grid_step(.5, 1.0, 4) == 2
    assert dividend_grid_step(.5 + 1e-8, 1.0, 4) == 3
    assert study.parity_target(.51) != study.parity_target(.5)
    assert study.parity_target(.5) == pytest.approx(
        100 - 3 * __import__("math").exp(-.02 * .5)
        - 100 * __import__("math").exp(-.02)
    )


def test_grid_and_date_panels_share_one_valuation_when_identical() -> None:
    calls = []
    def recording(request):
        calls.append(request)
        return _fake_price(request)
    study.run_experiment(steps=(40,), shifts=(0,), date_steps=40,
                         pricer=recording, models=(study.MODELS[0],))
    assert len(calls) == 3  # Not 6: identical contract triplet memoized.


def test_summary_names_reference_and_contract_change_explicitly() -> None:
    rows = study.run_experiment(steps=(25,), shifts=(0,), date_steps=25,
                                pricer=_fake_price, models=(study.MODELS[0],))
    summary = study.summarize(rows)
    assert "NOT exact error" in summary
    assert "DIFFERENT dividend contract" in summary
    assert "valid: 2; failed: 0" in summary


def test_quick_and_full_outputs_are_disjoint() -> None:
    assert set(study.output_paths(True)).isdisjoint(study.output_paths(False))
    assert "phase6c_quick" in str(study.output_paths(True)[0])


def test_csv_fieldnames_are_unique() -> None:
    assert len(study.FIELDS) == len(set(study.FIELDS))


# test_dividend_grid_views.py (tests preserved)
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
