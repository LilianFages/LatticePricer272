"""Research efficiency, memory measurement and chart presentation.

Consolidated without dropping individual regression cases.
"""

from __future__ import annotations

from dataclasses import replace
import pytest
from frontend.pricing_service import PricingRequest, PricingResult
from frontend.research_efficiency_service import (
    ALLOCATION, BRANCH, FULL, EfficiencyPoint, _relative_to_full,
    count_allocated_nodes, efficiency_study,
)
import math
import pandas as pd
from frontend.research_efficiency_service import ALLOCATION, BRANCH, FULL
from frontend.research_efficiency_views import (
    EFFICIENCY_CHART_HEIGHT,
    _allocation_nodes_chart,
    _chart,
    _overlapping_allocation_ns,
    _price_gap_label,
    _speedup_baseline_chart,
    _speedup_frame,
)


# test_research_efficiency_service.py (tests preserved)
def test_real_allocated_full_and_pruned_n8() -> None:
    request = PricingRequest(steps=8)
    full = count_allocated_nodes(replace(request, model=FULL, pruning_threshold=0.0))
    branch = count_allocated_nodes(replace(request, model=BRANCH))
    alloc = count_allocated_nodes(replace(request, model=ALLOCATION))
    assert full == 81
    assert 1 <= branch <= full
    assert 1 <= alloc <= full


def test_same_n_full_reference_only() -> None:
    rows = [
        EfficiencyPoint(FULL, 20, "ok", 9., 2., 400, 400, None),
        EfficiencyPoint(ALLOCATION, 20, "ok", 9.002, 1.5, 220, 210, None),
        EfficiencyPoint(ALLOCATION, 40, "ok", 9.01, 3., 300, 290, None),
    ]
    compared = _relative_to_full(rows)
    assert compared[1].absolute_gap_to_full == pytest.approx(.002)
    assert compared[1].allocation_savings_pct == pytest.approx(45.)
    # Verify the expected numerical or workflow invariant on this case.
    assert compared[2].absolute_gap_to_full is None
    assert compared[2].allocation_savings_pct is None


@pytest.mark.parametrize("grid", [(), (10,), (10, 10), (40, 20), (1, 20), (20, 200)])
def test_invalid_grids_are_rejected(grid: object) -> None:
    with pytest.raises(ValueError):
        efficiency_study(PricingRequest(), steps_grid=grid)


def test_no_memory_runs_when_disabled() -> None:
    seen = []
    def timed(req: PricingRequest) -> PricingResult:
        from frontend.pricing_service import PricingResult
        seen.append(req.model)
        return PricingResult(req, price=10., elapsed_ms=2., reachable_nodes=20,
                             allocated_nodes=None, pruned_transitions=None,
                             black_scholes=None)
    def never(_req: PricingRequest) -> None:
        # Preserve observed data, including missing and failed outcomes.
        raise AssertionError("Memory profiler must be opt-in")
    report = efficiency_study(PricingRequest(steps=20), (10, 20), 1, False,
                              pricer=timed, allocation_counter=lambda _: 15,
                              memory_probe=never)
    assert len(report.points) == 6
    assert len(seen) == 6
    assert all(p.peak_python_mib is None for p in report.points)
    assert all(p.status == "ok" for p in report.points)


def test_failed_configuration_is_not_replaced_by_zero() -> None:
    from frontend.pricing_service import PricingResult
    def sometimes_fails(req: PricingRequest) -> PricingResult:
        if req.model == BRANCH and req.steps == 20:
            raise RuntimeError("test failure")
        return PricingResult(req, price=10., elapsed_ms=2., reachable_nodes=20,
                             allocated_nodes=None, pruned_transitions=None,
                             black_scholes=None)
    result = efficiency_study(PricingRequest(steps=20), (10, 20), 1, False,
                              pricer=sometimes_fails, allocation_counter=lambda _: 12)
    # Preserve observed data, including missing and failed outcomes.
    missing = next(p for p in result.points if p.model == BRANCH and p.steps == 20)
    assert missing.status != "ok" and missing.price is None
    assert next(p for p in result.points if p.model == ALLOCATION and p.steps == 20).status == "ok"


def test_optional_memory_probe_kept_out_of_pricer_timing() -> None:
    from frontend.pricing_service import PricingResult
    n = {"timed": 0, "profiled": 0}
    def timed(req: PricingRequest) -> PricingResult:
        n["timed"] += 1
        return PricingResult(req, price=10., elapsed_ms=2., reachable_nodes=10,
                             allocated_nodes=None, pruned_transitions=None,
                             black_scholes=None)
    def memory(_req: PricingRequest) -> tuple[float, float]:
        n["profiled"] += 1
        return (10., 0.3)
    report = efficiency_study(PricingRequest(steps=20), (10, 20), 3, True,
                              pricer=timed, allocation_counter=lambda _: 12,
                              memory_probe=memory)
    # Verify the expected numerical or workflow invariant on this case.
    assert n == {"timed": 18, "profiled": 6}
    assert all(p.peak_python_mib == pytest.approx(.3) for p in report.points)
    assert all(p.median_ms == pytest.approx(2.) for p in report.points)


def test_real_study_prices_are_financially_consistent() -> None:
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


# test_research_efficiency_views.py (tests preserved)
def _spec() -> dict:
    sample = pd.DataFrame({
        "Model": ["Trinomial · Full", "Trinomial · Allocation pruning"],
        "N": [20, 40],
        "Allocated nodes": [441, 800],
    })
    return _chart(sample, "N", "Allocated nodes").to_dict()



# Chart geometry and legends are part of the stable presentation contract.
def test_efficiency_chart_is_taller_than_previous_layout() -> None:
    assert EFFICIENCY_CHART_HEIGHT == 305
    assert _spec()["height"] == EFFICIENCY_CHART_HEIGHT



def test_legend_uses_compact_bottom_layout() -> None:
    legend = _spec()["encoding"]["color"]["legend"]
    assert legend["orient"] == "bottom"
    assert legend["direction"] == "horizontal"
    assert legend["columns"] == 2
    assert legend["labelLimit"] >= 200
    assert legend.get("title") is None



def test_price_scale_keeps_nonzero_origin_and_tooltips() -> None:
    spec = _spec()
    assert spec["encoding"]["y"]["scale"]["zero"] is False
    # Verify the expected numerical or workflow invariant on this case.
    assert spec["encoding"]["x"]["field"] == "N"
    assert spec["encoding"]["color"]["field"] == "Model"
    assert spec.get("params"), "Zoom and pan must remain available."



# Explicit chart heights must override the default without rewriting the plot.
def test_explicit_chart_height_is_still_supported() -> None:
    sample = pd.DataFrame({
        "Model": ["Trinomial · Full"], "N": [20], "Allocated nodes": [441],
    })
    assert _chart(sample, "N", "Allocated nodes", height=270).to_dict()["height"] == 270



def _points() -> pd.DataFrame:
    return pd.DataFrame({
        "Model": [FULL, BRANCH, ALLOCATION] * 2,
        "N": [20, 20, 20, 40, 40, 40],
        "Allocated nodes": [441, 441, 220, 1681, 1681, 790],
    })



# Full and Branch allocation counts are only comparable at identical N.
def test_overlap_is_exact_and_at_equal_n() -> None:
    assert _overlapping_allocation_ns(_points()) == (20, 40)
    sample = _points()
    sample.loc[(sample["Model"] == BRANCH) & (sample["N"] == 40), "Allocated nodes"] = 900
    assert _overlapping_allocation_ns(sample) == (20,)



def test_no_silent_pairing_of_ambiguous_or_missing_levels() -> None:
    sample = _points()
    assert _overlapping_allocation_ns(pd.concat([sample, sample.iloc[:1]])) == ()
    assert _overlapping_allocation_ns(sample[sample["Model"] == ALLOCATION]) == ()



def test_full_is_wider_under_dashed_branch_with_untruncated_legend() -> None:
    spec = _allocation_nodes_chart(_points()).to_dict()
    layers = spec["layer"]
    assert len(layers) == 3
    # Verify the expected numerical or workflow invariant on this case.
    assert layers[0]["mark"]["strokeWidth"] > layers[1]["mark"]["strokeWidth"]
    assert layers[1]["mark"]["strokeDash"] == [6, 5]
    assert layers[0]["mark"]["point"]["size"] > layers[1]["mark"]["point"]["size"]
    assert layers[0]["encoding"]["color"]["legend"]["columns"] == 1
    assert layers[0]["encoding"]["color"]["legend"]["labelLimit"] >= 300



# The full-tree baseline is measured, not an inferred asymptotic ratio.
def test_speedup_baseline_is_one_and_line_is_still_measured() -> None:
    sample = pd.DataFrame({
        "Model": [BRANCH, ALLOCATION], "N": [20, 20],
        "Speed-up (×)": [1.15, 1.35],
        "Median latency (ms)": [1.8, 1.5],
        "Full latency (ms)": [2.0, 2.0],
    })
    spec = _speedup_baseline_chart(sample).to_dict()
    assert spec["layer"][0]["mark"]["type"] == "rule"
    assert spec["layer"][0]["mark"]["strokeDash"] == [6, 4]
    assert spec["layer"][1]["mark"]["type"] == "line"
    assert spec["layer"][2]["mark"]["type"] == "text"
    assert spec["layer"][1]["encoding"]["color"]["legend"]["columns"] == 1
    assert "Full baseline" in spec["layer"][0]["encoding"]["y"]["field"]



def _latencies() -> pd.DataFrame:
    return pd.DataFrame([
        {"Model": FULL, "N": 20, "Median latency (ms)": 10.},
        {"Model": BRANCH, "N": 20, "Median latency (ms)": 5.},
        {"Model": ALLOCATION, "N": 20, "Median latency (ms)": 4.},
        {"Model": FULL, "N": 40, "Median latency (ms)": 40.},
        {"Model": BRANCH, "N": 40, "Median latency (ms)": 50.},
        {"Model": ALLOCATION, "N": 40, "Median latency (ms)": 20.},
    ])



# Speed-up is a positive latency ratio at identical lattice depth.
def test_speedup_uses_only_same_n_full_medians() -> None:
    rows = _speedup_frame(_latencies())
    observed = {(r["Model"], r["N"]): r["Speed-up (×)"]
                for _, r in rows.iterrows()}
    assert observed == pytest.approx({
        (BRANCH, 20): 2., (ALLOCATION, 20): 2.5,
        (BRANCH, 40): 0.8, (ALLOCATION, 40): 2.,
    })
    assert FULL not in rows["Model"].values



def test_speedup_excludes_missing_zero_and_duplicate_full_baselines() -> None:
    frame = _latencies()
    frame.loc[(frame["Model"] == FULL) & (frame["N"] == 40),
              "Median latency (ms)"] = 0.
    assert set(_speedup_frame(frame)["N"]) == {20}
    doubled = pd.concat((frame, frame.iloc[[0]]), ignore_index=True)
    assert _speedup_frame(doubled).empty



# Zero and nonfinite latencies cannot produce meaningful performance ratios.
def test_speedup_rejects_zero_or_nonfinite_pruned_latency() -> None:
    frame = _latencies()
    frame.loc[(frame["Model"] == BRANCH) & (frame["N"] == 20),
              "Median latency (ms)"] = 0.
    frame.loc[(frame["Model"] == ALLOCATION) & (frame["N"] == 40),
              "Median latency (ms)"] = math.inf
    combos = {(r["Model"], r["N"]) for _, r in _speedup_frame(frame).iterrows()}
    assert combos == {(ALLOCATION, 20), (BRANCH, 40)}



@pytest.mark.parametrize("value,expected", [
    (0., "0 €"), (None, "N/A"), (float("nan"), "N/A"),
    (1e-9, "1.00e-09 €"), (0.0001, "0.000100 €"),
])
def test_gap_display_preserves_nonzero_precision(value: float | None, expected: str) -> None:
    assert _price_gap_label(value) == expected

# LOT_5C3_GUARDED_REFACTOR
