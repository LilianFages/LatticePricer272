"""Research-lab contracts, stopping rules and runtime validation.

Consolidated without dropping individual regression cases.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest
from frontend.research_lab_service import (
    CATALOG, DatasetError, as_csv, comparable_cases, coverage_summary,
    frontier_thresholds, load_dataset, mass_consistency, observed_pareto,
    robustness_by_step,
)
from frontend import analytics_views, research_efficiency_views
from frontend.pricing_service import PricingInputError
from frontend.research_contract import (
    build_research_request,
    render_research_contract_fields,
)
from dataclasses import replace
from types import SimpleNamespace
from frontend.pricing_service import DividendInput, PricingInputError, PricingRequest
from frontend.research_stopping_service import compare_dividend_exercise


# test_research_lab_service.py (tests preserved)
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
def test_coverage_separates_comparisons_from_allocation_only() -> None:
    assert coverage_summary(_sample_robustness()) == {
        "total": 3, "compared": 2, "allocation_only": 1, "failed": 0}


# Exclude rows with an unavailable branch-side reference from price-gap statistics.
def test_comparable_cases_excludes_unavailable_reference() -> None:
    comp = comparable_cases(_sample_robustness())
    assert len(comp) == 2
    assert comp["absolute_gap"].notna().all()


# Average node savings must use only rows with comparable node counts.
def test_savings_only_uses_comparable_cases() -> None:
    result = robustness_by_step(_sample_robustness())
    assert len(result) == 1
    assert result.iloc[0]["mean_savings_pct"] == pytest.approx(35.2)


# Status metadata is mandatory to interpret missing numerical fields.
def test_missing_status_fields_fails() -> None:
    with pytest.raises(DatasetError):
        coverage_summary(pd.DataFrame({"status": ["passed"]}))


# Pareto membership requires joint non-domination in error and elapsed time.
def test_pareto_filters_dominated_and_keeps_tradeoff() -> None:
    data = pd.DataFrame({"model": ["A", "B", "C", "D"],
                         "max_median_runtime_seconds": [.010, .020, .030, .040],
                         "worst_absolute_error": [.02, .01, .015, .005]})
    result = observed_pareto(data)
    assert set(result["model"]) == {"A", "B", "D"}


# Equal observations and exact-zero errors remain legitimate frontier data.
def test_pareto_retains_equivalent_points_and_zero_error() -> None:
    data = pd.DataFrame({"model": ["A", "B", "C"],
                         "max_median_runtime_seconds": [.01, .01, .02],
                         "worst_absolute_error": [0, 0, 0]})
    assert set(observed_pareto(data)["model"]) == {"A", "B"}


# Nonpositive timing or invalid errors cannot define a feasible frontier.
def test_pareto_rejects_bad_values() -> None:
    frame = pd.DataFrame({"max_median_runtime_seconds": [0.0, -.01, .01],
                          "worst_absolute_error": [.1, .01, -.01]})
    assert observed_pareto(frame).empty


# Target selection uses observed feasible timing, not an extrapolated optimum.
def test_threshold_best_observed_fastest() -> None:
    frame = pd.DataFrame({"model": ["A", "B", "C"], "nb_steps": [20, 50, 100],
                          "worst_absolute_error": [.012, .004, .001],
                          "max_median_runtime_seconds": [.001, .015, .040]})
    result = frontier_thresholds(frame, (.01, .005, .0005))
    assert result.iloc[0]["model"] == "B"
    assert result.iloc[0]["runtime_ms"] == pytest.approx(15)
    assert result.iloc[1]["model"] == "B"
    assert result.iloc[2]["status"] == "not reached"


# Probability-mass diagnostics must preserve the observed numerical residual.
def test_mass_consistency_small_errors() -> None:
    assert mass_consistency(pd.DataFrame({"unconditional_probability_mass":
                                          [1.0, 1.0 + 1e-12, 1.0 - 1e-12]})) == pytest.approx(1e-12)


# Do not fabricate missing probability conservation measurements.
def test_mass_missing_column_rejected() -> None:
    with pytest.raises(DatasetError):
        mass_consistency(pd.DataFrame({"probability": [1.0]}))


# A known experiment must load with its expected scientific schema.
def test_catalogue_can_load_known_schema(tmp_path: Path) -> None:
    spec = CATALOG["Runtime scaling"]
    p = tmp_path / spec.path
    p.parent.mkdir(parents=True)
    p.write_text("nb_steps,recursive_seconds,backward_seconds\n100,0.1,0.2\n", encoding="utf-8")
    result = load_dataset("Runtime scaling", root=tmp_path)
    assert result.iloc[0]["nb_steps"] == 100


# Schema drift must _fail visibly rather than mislabel numerical series.
def test_catalogue_rejects_schema_drift(tmp_path: Path) -> None:
    spec = CATALOG["Robustness"]
    p = tmp_path / spec.path
    p.parent.mkdir(parents=True)
    p.write_text("scenario,nb_steps\nnone,100\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="Unexpected columns"):
        load_dataset("Robustness", root=tmp_path)


# Dataset selection is confined to the declared experiment catalogue.
def test_catalogue_rejects_unknown_file_name() -> None:
    with pytest.raises(DatasetError, match="Unknown research dataset"):
        load_dataset("../../src/trinomial_tree.py")


# CSV export must preserve Unicode and spreadsheet-readable encoding.
def test_csv_export_utf8_with_excel_bom() -> None:
    value = as_csv(pd.DataFrame({"label": ["européen"], "n": [4]}))
    assert value.startswith(b"\xef\xbb\xbf")
    assert b"europ" in value



# Missing files must raise an explicit data-availability error.
def test_dataset_missing_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="Missing experiment export"):
        load_dataset("Pruning benchmark", root=tmp_path)


# Integration assertions apply to actual exports when present.
def test_repository_exports_when_available() -> None:
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
def test_cost_error_filters_agree_between_pareto_and_targets(bad_time: object, bad_gap: object) -> None:
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


def test_cost_error_exact_zero_gap_is_accepted_by_both_summaries() -> None:
    # Preserve observed data, including missing and failed outcomes.
    source = pd.DataFrame({
        "model": ["Zero", "Other"], "nb_steps": [20, 40],
        "max_median_runtime_seconds": [0.02, 0.03],
        "worst_absolute_error": [0.0, 0.002],
    })
    assert observed_pareto(source)["model"].tolist() == ["Zero"]
    best = frontier_thresholds(source, (0.0,))
    assert best.loc[0, "model"] == "Zero"
    assert best.loc[0, "error_eur"] == 0.0


def test_research_archives_still_use_single_dispatcher() -> None:
    import frontend.research_lab as lab
    assert not hasattr(lab, "render_research_lab_page")
    assert len(lab.RESEARCH_ARCHIVE_HANDLERS) == 6
    assert all(callable(handler) for handler in lab.RESEARCH_ARCHIVE_HANDLERS.values())


# test_research_contract_shared.py (tests preserved)
def _fields(dividends: object=None):
    return {
        "spot": 120.0, "strike": 110.0, "volatility": 0.32,
        "rate": -0.01, "maturity": 1.5,
        "option_type": "Put", "exercise": "American",
        "dividends_frame": pd.DataFrame(dividends or [], columns=["time", "amount"]),
    }


def test_both_workspaces_reuse_identical_contract_functions() -> None:
    assert analytics_views._contract_fields is render_research_contract_fields
    assert research_efficiency_views._contract_fields is render_research_contract_fields
    assert analytics_views._get_request is build_research_request
    assert research_efficiency_views._get_request is build_research_request


def test_pricing_request_preserves_units_market_and_lattice_settings() -> None:
    request = build_research_request(
        _fields(), "Trinomial · Allocation pruning", 125, 1e-12,
    )
    # Verify the expected numerical or workflow invariant on this case.
    assert (request.spot, request.strike, request.volatility) == (120.0, 110.0, 0.32)
    assert (request.rate, request.maturity) == (-0.01, 1.5)
    assert (request.option_type, request.exercise) == ("Put", "American")
    assert (request.model, request.steps, request.pruning_threshold) == (
        "Trinomial · Allocation pruning", 125, 1e-12,
    )
    assert request.dividends == ()


def test_discrete_cash_dividends_are_sorted_and_zero_ignored() -> None:
    request = build_research_request(
        _fields([{"time": 1.0, "amount": 2.0},
                 {"time": 0.5, "amount": 1.5},
                 {"time": 0.8, "amount": 0.0}]),
        "Trinomial · Allocation pruning", 100, 1e-18,
    )
    assert [(p.time, p.amount) for p in request.dividends] == [(0.5, 1.5), (1.0, 2.0)]


# Preserve observed data, including missing and failed outcomes.
def test_incomplete_dividend_is_not_silently_dropped() -> None:
    fields = _fields([{"time": 0.5, "amount": None}])
    with pytest.raises(PricingInputError, match="Dividend row 1"):
        build_research_request(fields, "Trinomial · Full", 80, 0.0)


def test_research_contract_does_not_depend_on_analytics_view() -> None:
    from inspect import getsource
    import frontend.research_contract as shared
    code = getsource(shared)
    assert "from frontend.analytics_views" not in code
    assert "from frontend.model_comparison" not in code
    assert "from frontend.dividend_input import parse_dividends" in code


# test_research_stopping_service.py (tests preserved)
def _request(**kwargs: object):
    base = PricingRequest(
        model="Trinomial · Allocation pruning", exercise="American",
        option_type="Put", steps=20,
        dividends=(DividendInput(.5, 3.),),
    )
    return replace(base, **kwargs)


# Comparison must share a contract and discretization without mutation.
def test_same_market_and_grid_without_mutating_original() -> None:
    observed = []

    def _fake_analyzer(req: PricingRequest):
        observed.append(req)
        return SimpleNamespace(request=req, premium=8.0 if req.dividends else 7.0,
                               early_exercise_probability=.25 if req.dividends else .30)

    original = _request()
    result = compare_dividend_exercise(original, analyzer=_fake_analyzer)
    assert observed[0] is original
    assert observed[1] is not original
    assert observed[1] == replace(original, dividends=())
    assert original.dividends == (DividendInput(.5, 3.),)
    assert result.premium_difference == pytest.approx(1.0)
    assert result.node_exercise_difference == pytest.approx(-.05)


# Invalid comparisons _fail before a lattice is built or priced.
def test_no_dividends_rejected_without_pricing() -> None:
    calls = []
    with pytest.raises(PricingInputError, match="Add at least one"):
        compare_dividend_exercise(_request(dividends=()), analyzer=calls.append)
    assert calls == []


def test_binomial_cash_dividend_is_not_silently_substituted() -> None:
    with pytest.raises(PricingInputError, match="Binomial"):
        compare_dividend_exercise(_request(model="Binomial"))


def test_two_dividends_in_same_lattice_step_rejected() -> None:
    req = _request(dividends=(DividendInput(.451, 2.), DividendInput(.49, 1.)))
    with pytest.raises(PricingInputError, match="same lattice"):
        compare_dividend_exercise(req)


def test_interactive_depth_guard_precedes_analysis() -> None:
    # Preserve observed data, including missing and failed outcomes.
    calls = []
    with pytest.raises(PricingInputError, match="at most N=200"):
        compare_dividend_exercise(_request(steps=201), analyzer=calls.append)
    assert not calls


def test_analyzer_failure_is_not_silenced() -> None:
    def _fail(_: object):
        raise RuntimeError("engine failure")
    with pytest.raises(RuntimeError, match="engine failure"):
        compare_dividend_exercise(_request(), analyzer=_fail)


# Valid dividend events remain explicit in both comparison scenarios.
def test_two_cashflow_events_preserved_in_dividend_case() -> None:
    req = _request(steps=20, dividends=(DividendInput(.25, 2.), DividendInput(.75, 1.)))
    seen = []

    def _analyzer(r: object):
        seen.append(r)
        return SimpleNamespace(request=r, premium=3.0,
                               early_exercise_probability=0.0)

    compare_dividend_exercise(req, analyzer=_analyzer)
    assert seen[0].dividends == req.dividends
    assert seen[1].dividends == ()
    assert seen[1].steps == seen[0].steps


def test_real_engine_comparison_uses_two_valid_studies() -> None:
    result = compare_dividend_exercise(_request(steps=20))
    assert result.with_dividends.request.dividends
    assert not result.without_dividends.request.dividends
    # Verify the expected numerical or workflow invariant on this case.
    assert result.with_dividends.request.spot == result.without_dividends.request.spot
    assert len(result.with_dividends.dates) == 21
    assert len(result.without_dividends.dates) == 21
    assert result.with_dividends.worst_probability_mass_error < 2e-10
    assert result.without_dividends.worst_probability_mass_error < 2e-10

# LOT_5C4_GUARDED_REFACTOR
