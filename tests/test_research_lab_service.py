"""Unit tests: Research Lab loads real schemas, never invents comparisons."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from frontend.research_lab_service import (
    CATALOG, DatasetError, as_csv, comparable_cases, coverage_summary,
    frontier_thresholds, load_dataset, mass_consistency, observed_pareto,
    robustness_by_step,
)


# This mixed fixture distinguishes comparable runs from allocation-only outcomes.
def _sample_robustness() -> pd.DataFrame:
    return pd.DataFrame([
        {"scenario": "none", "nb_steps": 100, "kind": "call", "status": "passed",
         "allocation_status": "success", "branch_status": "success", "node_savings_pct": 34.4,
         "absolute_gap": 0.0},
        {"scenario": "two", "nb_steps": 200, "kind": "put", "status": "allocation_only",
         "allocation_status": "success", "branch_status": "error", "node_savings_pct": float("nan"),
         "absolute_gap": float("nan")},
        {"scenario": "none", "nb_steps": 100, "kind": "put", "status": "passed",
         "allocation_status": "success", "branch_status": "success", "node_savings_pct": 36.0,
         "absolute_gap": 0.0},
    ])


# Only successful two-sided runs constitute direct accuracy comparisons.
def test_coverage_separates_comparisons_from_allocation_only():
    assert coverage_summary(_sample_robustness()) == {
        "total": 3, "compared": 2, "allocation_only": 1, "failed": 0}


# Exclude rows with an unavailable branch-side reference from price-gap statistics.
def test_comparable_cases_excludes_unavailable_reference():
    comp = comparable_cases(_sample_robustness())
    assert len(comp) == 2
    assert comp["absolute_gap"].notna().all()


# Average node savings must use only rows with comparable node counts.
def test_savings_only_uses_comparable_cases():
    result = robustness_by_step(_sample_robustness())
    assert len(result) == 1
    assert result.iloc[0]["mean_savings_pct"] == pytest.approx(35.2)


# Status metadata is mandatory to interpret missing numerical fields.
def test_missing_status_fields_fails():
    with pytest.raises(DatasetError):
        coverage_summary(pd.DataFrame({"status": ["passed"]}))


# Pareto membership requires joint non-domination in error and elapsed time.
def test_pareto_filters_dominated_and_keeps_tradeoff():
    data = pd.DataFrame({"model": ["A", "B", "C", "D"],
                         "max_median_runtime_seconds": [.010, .020, .030, .040],
                         "worst_absolute_error": [.02, .01, .015, .005]})
    result = observed_pareto(data)
    assert set(result["model"]) == {"A", "B", "D"}


# Equal observations and exact-zero errors remain legitimate frontier data.
def test_pareto_retains_equivalent_points_and_zero_error():
    data = pd.DataFrame({"model": ["A", "B", "C"],
                         "max_median_runtime_seconds": [.01, .01, .02],
                         "worst_absolute_error": [0, 0, 0]})
    assert set(observed_pareto(data)["model"]) == {"A", "B"}


# Nonpositive timing or invalid errors cannot define a feasible frontier.
def test_pareto_rejects_bad_values():
    frame = pd.DataFrame({"max_median_runtime_seconds": [0.0, -.01, .01],
                          "worst_absolute_error": [.1, .01, -.01]})
    assert observed_pareto(frame).empty


# Target selection uses observed feasible timing, not an extrapolated optimum.
def test_threshold_best_observed_fastest():
    frame = pd.DataFrame({"model": ["A", "B", "C"], "nb_steps": [20, 50, 100],
                          "worst_absolute_error": [.012, .004, .001],
                          "max_median_runtime_seconds": [.001, .015, .040]})
    result = frontier_thresholds(frame, (.01, .005, .0005))
    assert result.iloc[0]["model"] == "B"
    assert result.iloc[0]["runtime_ms"] == pytest.approx(15)
    assert result.iloc[1]["model"] == "B"
    assert result.iloc[2]["status"] == "not reached"


# Probability-mass diagnostics must preserve the observed numerical residual.
def test_mass_consistency_small_errors():
    assert mass_consistency(pd.DataFrame({"unconditional_probability_mass":
                                          [1.0, 1.0 + 1e-12, 1.0 - 1e-12]})) == pytest.approx(1e-12)


# Do not fabricate missing probability conservation measurements.
def test_mass_missing_column_rejected():
    with pytest.raises(DatasetError):
        mass_consistency(pd.DataFrame({"probability": [1.0]}))


# A known experiment must load with its expected scientific schema.
def test_catalogue_can_load_known_schema(tmp_path: Path):
    spec = CATALOG["Runtime scaling"]
    p = tmp_path / spec.path
    p.parent.mkdir(parents=True)
    p.write_text("nb_steps,recursive_seconds,backward_seconds\n100,0.1,0.2\n", encoding="utf-8")
    result = load_dataset("Runtime scaling", root=tmp_path)
    assert result.iloc[0]["nb_steps"] == 100


# Schema drift must fail visibly rather than mislabel numerical series.
def test_catalogue_rejects_schema_drift(tmp_path: Path):
    spec = CATALOG["Robustness"]
    p = tmp_path / spec.path
    p.parent.mkdir(parents=True)
    p.write_text("scenario,nb_steps\nnone,100\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="Unexpected columns"):
        load_dataset("Robustness", root=tmp_path)


# Dataset selection is confined to the declared experiment catalogue.
def test_catalogue_rejects_unknown_file_name():
    with pytest.raises(DatasetError, match="Unknown research dataset"):
        load_dataset("../../src/trinomial_tree.py")


# CSV export must preserve Unicode and spreadsheet-readable encoding.
def test_csv_export_utf8_with_excel_bom():
    value = as_csv(pd.DataFrame({"label": ["européen"], "n": [4]}))
    assert value.startswith(b"\xef\xbb\xbf")
    assert b"europ" in value



# Missing files must raise an explicit data-availability error.
def test_dataset_missing_is_explicit(tmp_path: Path):
    with pytest.raises(DatasetError, match="Missing experiment export"):
        load_dataset("Pruning benchmark", root=tmp_path)


# Integration assertions apply to actual exports when present.
def test_repository_exports_when_available():
    """Schema smoke test when run inside the real Git checkout."""
    from frontend.research_lab_service import ROOT
    for name in ("Robustness", "Accuracy/cost", "Temporal profiles", "Pruning benchmark"):
        if not (ROOT / CATALOG[name].path).exists():
            pytest.skip("Real repository exports not present in standalone test copy")
    robust = load_dataset("Robustness")
    cost = load_dataset("Accuracy/cost")
    temporal = load_dataset("Temporal profiles")
    assert len(robust) > 0 and len(cost) > 0 and len(temporal) > 0
    assert coverage_summary(robust)["compared"] > 0
    assert mass_consistency(temporal) < 1e-8


# Lot 5B.5 — archived and live workflows share the same observed-data policy.
@pytest.mark.parametrize("bad_time,bad_gap", [
    (None, 0.02), (-1.0, 0.01), (0.0, 0.0), ("not-a-number", 0.1),
    (0.01, None), (0.01, -0.01), (0.01, "invalid"),
])
def test_cost_error_filters_agree_between_pareto_and_targets(bad_time, bad_gap):
    source = pd.DataFrame({
        "model": ["Valid", "Invalid"], "nb_steps": [20, 40],
        "max_median_runtime_seconds": [0.01, bad_time],
        "worst_absolute_error": [0.001, bad_gap],
    })
    original = source.copy(deep=True)
    frontier = observed_pareto(source)
    target = frontier_thresholds(source, (0.002,))
    assert frontier["model"].tolist() == ["Valid"]
    assert target.loc[0, "model"] == "Valid"
    pd.testing.assert_frame_equal(source, original)


def test_cost_error_exact_zero_gap_is_accepted_by_both_summaries():
    source = pd.DataFrame({
        "model": ["Zero", "Other"], "nb_steps": [20, 40],
        "max_median_runtime_seconds": [0.02, 0.03],
        "worst_absolute_error": [0.0, 0.002],
    })
    assert observed_pareto(source)["model"].tolist() == ["Zero"]
    best = frontier_thresholds(source, (0.0,))
    assert best.loc[0, "model"] == "Zero"
    assert best.loc[0, "error_eur"] == 0.0


def test_research_archives_still_use_single_dispatcher():
    import frontend.research_lab as lab
    assert not hasattr(lab, "render_research_lab_page")
    assert len(lab.RESEARCH_ARCHIVE_HANDLERS) == 6
    assert all(callable(handler) for handler in lab.RESEARCH_ARCHIVE_HANDLERS.values())
