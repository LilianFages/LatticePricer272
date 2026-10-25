"""v4.2 regression: missingness provenance and preserved research data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from frontend.research_lab_service import (
    CATALOG,
    DatasetError,
    as_csv,
    load_dataset,
    missingness_by_column,
    missingness_diagnostics,
)


COMPARISON_FIELDS = (
    "branch_price", "absolute_gap", "relative_gap", "branch_nodes",
    "node_savings_pct", "branch_runtime_seconds", "branch_mass_error",
)


def _sample_csv() -> pd.DataFrame:
    passed = dict(status="passed", branch_error=None, allocation_error=None,
                  error=None, **{key: 0.0 for key in COMPARISON_FIELDS})
    allocation_only = dict(status="allocation_only",
                           branch_error="RuntimeError: baseline failure",
                           allocation_error=None, error="No pricing comparison",
                           **{key: None for key in COMPARISON_FIELDS})
    return pd.DataFrame([passed, passed, allocation_only])


# Each missing cell must be assigned to exactly one cause.
def test_missingness_categories_are_disjoint():
    frame = _sample_csv()
    result = missingness_diagnostics("Robustness", frame)
    assert result == {
        "total": 14, "empty_error_fields": 7,
        "unavailable_comparisons": 7, "unclassified": 0,
    }


# The 108-case reference contains 96 comparisons and 12 allocation-only runs.
def test_missingness_known_108_case_profile():
    baseline = dict(status="passed", branch_error=None, allocation_error=None,
                    error=None, **{key: 1.0 for key in COMPARISON_FIELDS})
    allocation = dict(status="allocation_only",
                      branch_error="RuntimeError", allocation_error=None,
                      error="No comparison", **{key: None for key in COMPARISON_FIELDS})
    rows = pd.DataFrame([baseline] * 96 + [allocation] * 12)
    result = missingness_diagnostics("Robustness", rows)
    assert result["total"] == 384
    assert result["empty_error_fields"] == 300
    assert result["unavailable_comparisons"] == 84
    assert result["unclassified"] == 0


# Unexpected null fields must remain visible as unclassified data loss.
def test_unknown_missingness_is_not_explained_away():
    frame = _sample_csv()
    frame["new_metric"] = [None, None, 1.0]
    report = missingness_diagnostics("Robustness", frame)
    assert report["unclassified"] == 2
    assert report["total"] == 16


# A missing value in another experiment has no automatic branch-failure cause.
def test_other_catalogue_blanks_not_assumed_to_be_baseline_failures():
    data = pd.DataFrame({"price": [1.0, None]})
    assert missingness_diagnostics("Accuracy/cost", data) == {
        "total": 1, "empty_error_fields": 0,
        "unavailable_comparisons": 0, "unclassified": 1,
    }


# Column diagnostics must not mutate the original measurements.
def test_missingness_by_column_counts_only_nonzero_and_preserves_source():
    frame = _sample_csv()
    original = frame.copy(deep=True)
    detail = missingness_by_column(frame)
    assert int(detail["missing_cells"].sum()) == 14
    assert detail.loc[detail["column"].eq("branch_price"), "missing_cells"].iloc[0] == 1
    assert detail.loc[detail["column"].eq("branch_price"), "share_pct"].iloc[0] == pytest.approx(33.3)
    pd.testing.assert_frame_equal(frame, original)


# Missing status fields prevent valid robustness provenance analysis.
def test_missingness_requires_status_for_robustness():
    with pytest.raises(DatasetError, match="status"):
        missingness_diagnostics("Robustness", pd.DataFrame({"gap": [None]}))


# Diagnostics must leave the original CSV values unchanged.
def test_csv_remains_unimputed_after_provenance_check():
    frame = _sample_csv()
    missingness_diagnostics("Robustness", frame)
    exported = as_csv(frame).decode("utf-8-sig")
    assert "allocation_only" in exported
    assert "RuntimeError" in exported
    assert not frame.loc[2, "absolute_gap"] == 0.0
    assert pd.isna(frame.loc[2, "absolute_gap"])




# Catalogue registration must preserve dataset names and locations.
def test_catalogue_schema_and_source_paths_unchanged():
    assert CATALOG["Robustness"].path == "results/data/allocation_pruning_robustness.csv"
    assert CATALOG["Temporal profiles"].path == "results/data/option_by_dates_profiles.csv"
    assert CATALOG["Accuracy/cost"].path == "results/data/accuracy_cost_multistrike_aggregate.csv"


# Check empirical exports without treating optional datasets as required.
def test_real_datasets_when_available():
    # This runs on the user's checkout, and skips only in standalone archive QA.
    from frontend.research_lab_service import ROOT
    if not (ROOT / CATALOG["Robustness"].path).is_file():
        pytest.skip("Full experimental CSV not present outside checkout")
    frame = load_dataset("Robustness")
    report = missingness_diagnostics("Robustness", frame)
    assert report["total"] == 384
    assert report["empty_error_fields"] == 300
    assert report["unavailable_comparisons"] == 84
    assert report["unclassified"] == 0
