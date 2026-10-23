"""Lot 3B: real allocations, exact same-N comparisons and no silent failures."""
from __future__ import annotations

from dataclasses import replace

import pytest

from frontend.pricing_service import PricingRequest
from frontend.research_efficiency_service import (
    ALLOCATION, BRANCH, FULL, EfficiencyPoint, _relative_to_full,
    count_allocated_nodes, efficiency_study,
)


def test_real_allocated_full_and_pruned_n8():
    request = PricingRequest(steps=8)
    full = count_allocated_nodes(replace(request, model=FULL, pruning_threshold=0.0))
    branch = count_allocated_nodes(replace(request, model=BRANCH))
    alloc = count_allocated_nodes(replace(request, model=ALLOCATION))
    assert full == 81
    assert 1 <= branch <= full
    assert 1 <= alloc <= full


def test_same_n_full_reference_only():
    rows = [
        EfficiencyPoint(FULL, 20, "ok", 9., 2., 400, 400, None),
        EfficiencyPoint(ALLOCATION, 20, "ok", 9.002, 1.5, 220, 210, None),
        EfficiencyPoint(ALLOCATION, 40, "ok", 9.01, 3., 300, 290, None),
    ]
    compared = _relative_to_full(rows)
    assert compared[1].absolute_gap_to_full == pytest.approx(.002)
    assert compared[1].allocation_savings_pct == pytest.approx(45.)
    assert compared[2].absolute_gap_to_full is None
    assert compared[2].allocation_savings_pct is None


@pytest.mark.parametrize("grid", [(), (10,), (10, 10), (40, 20), (1, 20), (20, 200)])
def test_invalid_grids_are_rejected(grid):
    with pytest.raises(ValueError):
        efficiency_study(PricingRequest(), steps_grid=grid)


def test_no_memory_runs_when_disabled():
    seen = []
    def timed(req):
        from frontend.pricing_service import PricingResult
        seen.append(req.model)
        return PricingResult(req, price=10., elapsed_ms=2., reachable_nodes=20,
                             allocated_nodes=None, pruned_transitions=None,
                             black_scholes=None)
    def never(_req):
        raise AssertionError("Memory profiler must be opt-in")
    report = efficiency_study(PricingRequest(steps=20), (10, 20), 1, False,
                              pricer=timed, allocation_counter=lambda _: 15,
                              memory_probe=never)
    assert len(report.points) == 6
    assert len(seen) == 6
    assert all(p.peak_python_mib is None for p in report.points)
    assert all(p.status == "ok" for p in report.points)


def test_failed_configuration_is_not_replaced_by_zero():
    from frontend.pricing_service import PricingResult
    def sometimes_fails(req):
        if req.model == BRANCH and req.steps == 20:
            raise RuntimeError("test failure")
        return PricingResult(req, price=10., elapsed_ms=2., reachable_nodes=20,
                             allocated_nodes=None, pruned_transitions=None,
                             black_scholes=None)
    result = efficiency_study(PricingRequest(steps=20), (10, 20), 1, False,
                              pricer=sometimes_fails, allocation_counter=lambda _: 12)
    missing = next(p for p in result.points if p.model == BRANCH and p.steps == 20)
    assert missing.status != "ok" and missing.price is None
    assert next(p for p in result.points if p.model == ALLOCATION and p.steps == 20).status == "ok"


def test_optional_memory_probe_kept_out_of_pricer_timing():
    from frontend.pricing_service import PricingResult
    n = {"timed": 0, "profiled": 0}
    def timed(req):
        n["timed"] += 1
        return PricingResult(req, price=10., elapsed_ms=2., reachable_nodes=10,
                             allocated_nodes=None, pruned_transitions=None,
                             black_scholes=None)
    def memory(_req):
        n["profiled"] += 1
        return (10., 0.3)
    report = efficiency_study(PricingRequest(steps=20), (10, 20), 3, True,
                              pricer=timed, allocation_counter=lambda _: 12,
                              memory_probe=memory)
    assert n == {"timed": 18, "profiled": 6}
    assert all(p.peak_python_mib == pytest.approx(.3) for p in report.points)
    assert all(p.median_ms == pytest.approx(2.) for p in report.points)


def test_real_study_prices_are_financially_consistent():
    request = PricingRequest(steps=10, pruning_threshold=1e-18)
    result = efficiency_study(request, (5, 10), repetitions=1)
    assert len(result.points) == 6
    assert all(point.status == "ok" for point in result.points)
    for step in (5, 10):
        full = next(p for p in result.points if p.model == FULL and p.steps == step)
        optimized = next(p for p in result.points if p.model == ALLOCATION and p.steps == step)
        assert optimized.absolute_gap_to_full == pytest.approx(
            abs(optimized.price - full.price), abs=1e-12,
        )
