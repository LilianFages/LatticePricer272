"""Lot 3A: observed price-gap/runtime frontier is exact and non-mutating."""
from __future__ import annotations

from dataclasses import replace

import pytest

from frontend.analytics_service import ConvergenceResult, PricingObservation
from frontend.research_accuracy_service import (
    accuracy_cost_points, best_observed_under_target,
)


def _observation(model, n, price, ms, status="ok"):
    return PricingObservation(model, float(n), price, ms, status)


def _report(*rows, ref=10., ref_type="analytic"):
    return ConvergenceResult(tuple(rows), "Reference", ref, ref_type, None)


def test_frontier_detects_dominance_across_models():
    report = _report(_observation("A", 20, 10.10, 4.),
                     _observation("A", 40, 10.04, 6.),
                     _observation("B", 20, 10.10, 5.),
                     _observation("B", 50, 10.03, 9.))
    points = accuracy_cost_points(report)
    assert [p.pareto for p in points] == [True, True, False, True]
    assert [p.absolute_gap for p in points] == pytest.approx([.10, .04, .10, .03])


def test_exact_equal_configurations_are_both_nondominated():
    rows = accuracy_cost_points(_report(
        _observation("A", 40, 10., 5.), _observation("B", 40, 10., 5.)))
    assert len(rows) == 2 and all(p.pareto for p in rows)


def test_failed_and_invalid_observations_never_count_as_zero():
    rows = accuracy_cost_points(_report(
        _observation("A", 40, None, None, "failed"),
        _observation("B", 20, 10., float("nan")),
        _observation("C", 20, 10.03, 3.)))
    assert [(p.model, p.steps) for p in rows] == [("C", 20)]


def test_missing_reference_disables_frontier():
    report = _report(_observation("A", 20, 10., 2.), ref=None)
    assert accuracy_cost_points(report) == ()


def test_numerical_reference_gap_is_not_discarded():
    report = _report(_observation("A", 20, 9.95, 2.),
                     ref_type="numerical")
    assert accuracy_cost_points(report)[0].absolute_gap == pytest.approx(.05)


def test_fastest_observed_meeting_target():
    rows = accuracy_cost_points(_report(
        _observation("Slow", 40, 10.001, 10.),
        _observation("Fast", 20, 10.004, 2.),
        _observation("Inexact", 5, 10.5, 1.)))
    assert best_observed_under_target(rows, .005).model == "Fast"
    assert best_observed_under_target(rows, .0001) is None


@pytest.mark.parametrize("target", [-1., float("nan"), float("inf")])
def test_invalid_target_rejected(target):
    with pytest.raises(ValueError):
        best_observed_under_target((), target)


def test_result_is_not_mutated():
    original = _report(_observation("A", 15, 11., 4.))
    copy = replace(original)
    accuracy_cost_points(original)
    assert original == copy
