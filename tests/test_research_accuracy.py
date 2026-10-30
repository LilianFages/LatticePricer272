"""Lot 3A: observed price-gap/runtime frontier is exact and non-mutating."""
from __future__ import annotations

from dataclasses import replace

import pytest

from frontend.analytics_service import ConvergenceResult, PricingObservation
from frontend.research_accuracy_service import (
    accuracy_cost_points, best_observed_under_target,
)
import pandas as pd
from frontend.analytics_views import _accuracy_cost_chart, _pareto_path_frame


def _observation(
    model: str, n: int, price: float | None,
    ms: float | None, status: str = "ok"
) -> PricingObservation:
    return PricingObservation(model, float(n), price, ms, status)


# Preserve observed data, including missing and failed outcomes.
def _report(
    *rows: PricingObservation, ref: float | None = 10.,
    ref_type: str = "analytic"
) -> ConvergenceResult:
    return ConvergenceResult(tuple(rows), "Reference", ref, ref_type, None)


def test_frontier_detects_dominance_across_models() -> None:
    report = _report(_observation("A", 20, 10.10, 4.),
                     _observation("A", 40, 10.04, 6.),
                     _observation("B", 20, 10.10, 5.),
                     _observation("B", 50, 10.03, 9.))
    points = accuracy_cost_points(report)
    assert [p.pareto for p in points] == [True, True, False, True]
    assert [p.absolute_gap for p in points] == pytest.approx([.10, .04, .10, .03])


def test_exact_equal_configurations_are_both_nondominated() -> None:
    rows = accuracy_cost_points(_report(
        _observation("A", 40, 10., 5.), _observation("B", 40, 10., 5.)))
    assert len(rows) == 2 and all(p.pareto for p in rows)


# Preserve observed data, including missing and failed outcomes.
def test_failed_and_invalid_observations_never_count_as_zero() -> None:
    rows = accuracy_cost_points(_report(
        _observation("A", 40, None, None, "failed"),
        _observation("B", 20, 10., float("nan")),
        _observation("C", 20, 10.03, 3.)))
    assert [(p.model, p.steps) for p in rows] == [("C", 20)]


def test_missing_reference_disables_frontier() -> None:
    report = _report(_observation("A", 20, 10., 2.), ref=None)
    assert accuracy_cost_points(report) == ()


def test_numerical_reference_gap_is_not_discarded() -> None:
    report = _report(_observation("A", 20, 9.95, 2.),
                     ref_type="numerical")
    assert accuracy_cost_points(report)[0].absolute_gap == pytest.approx(.05)


# Preserve observed data, including missing and failed outcomes.
def test_fastest_observed_meeting_target() -> None:
    rows = accuracy_cost_points(_report(
        _observation("Slow", 40, 10.001, 10.),
        _observation("Fast", 20, 10.004, 2.),
        _observation("Inexact", 5, 10.5, 1.)))
    assert best_observed_under_target(rows, .005).model == "Fast"
    assert best_observed_under_target(rows, .0001) is None


@pytest.mark.parametrize("target", [-1., float("nan"), float("inf")])
def test_invalid_target_rejected(target: float) -> None:
    with pytest.raises(ValueError):
        best_observed_under_target((), target)


def test_result_is_not_mutated() -> None:
    original = _report(_observation("A", 15, 11., 4.))
    copy = replace(original)
    accuracy_cost_points(original)
    # Verify the expected numerical or workflow invariant on this case.
    assert original == copy


# Lot 5B.7B — Pareto visualization tests
def _pareto_data() -> pd.DataFrame:
    return pd.DataFrame([
        {"Model":"Binomial", "N":20, "Premium (€)":10.1,
         "Runtime (ms)":1., "Absolute gap (€)":.1, "Pareto candidate":True},
        {"Model":"Trinomial", "N":40, "Premium (€)":10.05,
         "Runtime (ms)":3., "Absolute gap (€)":.05, "Pareto candidate":True},
        {"Model":"Binomial", "N":40, "Premium (€)":10.09,
         "Runtime (ms)":4., "Absolute gap (€)":.09, "Pareto candidate":False},
        {"Model":"Trinomial", "N":80, "Premium (€)":10.01,
         "Runtime (ms)":7., "Absolute gap (€)":.01, "Pareto candidate":True},
    ])


def test_pareto_vertices_global_and_sorted() -> None:
    p = _pareto_path_frame(_pareto_data().iloc[::-1])
    assert p["Runtime (ms)"].tolist() == [1., 3., 7.]
    assert p["Absolute gap (€)"].tolist() == [.1, .05, .01]
    assert p["Model"].tolist() == ["Binomial", "Trinomial", "Trinomial"]


# Preserve observed data, including missing and failed outcomes.
def test_identical_frontier_points_do_not_duplicate_line_vertices() -> None:
    src = _pareto_data()
    extra = src.iloc[[0]].copy()
    extra.loc[extra.index[0], "Model"] = "Other model"
    path = _pareto_path_frame(pd.concat((src, extra), ignore_index=True))
    assert len(path) == 3


def _layer_data(spec: dict[str, object], index: int) -> object:
    source = spec["layer"][index]["data"]
    return (source["values"] if "values" in source
            else spec["datasets"][source["name"]])


def test_chart_contains_envelope_model_points_and_rings() -> None:
    spec = _accuracy_cost_chart(_pareto_data(), pareto_only=False,
                                gap_label="Absolute Black–Scholes error (€)").to_dict()
    assert len(spec["layer"]) == 3
    line, points, rings = spec["layer"]
    # Verify the expected numerical or workflow invariant on this case.
    assert line["mark"]["type"] == "line"
    assert line["mark"]["strokeDash"]
    assert points["mark"]["type"] == "circle"
    assert points["encoding"]["color"]["field"] == "Model"
    assert points["encoding"]["color"]["legend"]["orient"] == "bottom"
    assert rings["mark"]["type"] == "point"
    assert len(_layer_data(spec, 0)) == 3
    assert len(_layer_data(spec, 1)) == 4


def test_pareto_only_is_view_filter_without_changing_frontier() -> None:
    spec = _accuracy_cost_chart(_pareto_data(), pareto_only=True,
                                gap_label="Absolute price gap (€)").to_dict()
    assert len(_layer_data(spec, 0)) == 3
    assert len(_layer_data(spec, 1)) == 3
