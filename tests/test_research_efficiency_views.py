"""Computational Efficiency: measured speed-up, allocation charts and UI regressions.

Consolidated from the historical layout, clarity and chart cleanup suites.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

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


def _spec():
    sample = pd.DataFrame({
        "Model": ["Trinomial · Full", "Trinomial · Allocation pruning"],
        "N": [20, 40],
        "Allocated nodes": [441, 800],
    })
    return _chart(sample, "N", "Allocated nodes").to_dict()



def test_efficiency_chart_is_taller_than_previous_layout():
    assert EFFICIENCY_CHART_HEIGHT == 305
    assert _spec()["height"] == EFFICIENCY_CHART_HEIGHT



def test_legend_uses_compact_bottom_layout():
    legend = _spec()["encoding"]["color"]["legend"]
    assert legend["orient"] == "bottom"
    assert legend["direction"] == "horizontal"
    assert legend["columns"] == 2
    assert legend["labelLimit"] >= 200
    assert legend.get("title") is None



def test_price_scale_keeps_nonzero_origin_and_tooltips():
    spec = _spec()
    assert spec["encoding"]["y"]["scale"]["zero"] is False
    assert spec["encoding"]["x"]["field"] == "N"
    assert spec["encoding"]["color"]["field"] == "Model"
    assert spec.get("params"), "Zoom and pan must remain available."



def test_explicit_chart_height_is_still_supported():
    sample = pd.DataFrame({
        "Model": ["Trinomial · Full"], "N": [20], "Allocated nodes": [441],
    })
    assert _chart(sample, "N", "Allocated nodes", height=270).to_dict()["height"] == 270



def _points():
    return pd.DataFrame({
        "Model": [FULL, BRANCH, ALLOCATION] * 2,
        "N": [20, 20, 20, 40, 40, 40],
        "Allocated nodes": [441, 441, 220, 1681, 1681, 790],
    })



def test_overlap_is_exact_and_at_equal_n():
    assert _overlapping_allocation_ns(_points()) == (20, 40)
    sample = _points()
    sample.loc[(sample["Model"] == BRANCH) & (sample["N"] == 40), "Allocated nodes"] = 900
    assert _overlapping_allocation_ns(sample) == (20,)



def test_no_silent_pairing_of_ambiguous_or_missing_levels():
    sample = _points()
    assert _overlapping_allocation_ns(pd.concat([sample, sample.iloc[:1]])) == ()
    assert _overlapping_allocation_ns(sample[sample["Model"] == ALLOCATION]) == ()



def test_full_is_wider_under_dashed_branch_with_untruncated_legend():
    spec = _allocation_nodes_chart(_points()).to_dict()
    layers = spec["layer"]
    assert len(layers) == 3
    assert layers[0]["mark"]["strokeWidth"] > layers[1]["mark"]["strokeWidth"]
    assert layers[1]["mark"]["strokeDash"] == [6, 5]
    assert layers[0]["mark"]["point"]["size"] > layers[1]["mark"]["point"]["size"]
    assert layers[0]["encoding"]["color"]["legend"]["columns"] == 1
    assert layers[0]["encoding"]["color"]["legend"]["labelLimit"] >= 300



def test_speedup_baseline_is_one_and_line_is_still_measured():
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



def _latencies():
    return pd.DataFrame([
        {"Model": FULL, "N": 20, "Median latency (ms)": 10.},
        {"Model": BRANCH, "N": 20, "Median latency (ms)": 5.},
        {"Model": ALLOCATION, "N": 20, "Median latency (ms)": 4.},
        {"Model": FULL, "N": 40, "Median latency (ms)": 40.},
        {"Model": BRANCH, "N": 40, "Median latency (ms)": 50.},
        {"Model": ALLOCATION, "N": 40, "Median latency (ms)": 20.},
    ])



def test_speedup_uses_only_same_n_full_medians():
    rows = _speedup_frame(_latencies())
    observed = {(r["Model"], r["N"]): r["Speed-up (×)"]
                for _, r in rows.iterrows()}
    assert observed == pytest.approx({
        (BRANCH, 20): 2., (ALLOCATION, 20): 2.5,
        (BRANCH, 40): 0.8, (ALLOCATION, 40): 2.,
    })
    assert FULL not in rows["Model"].values



def test_speedup_excludes_missing_zero_and_duplicate_full_baselines():
    frame = _latencies()
    frame.loc[(frame["Model"] == FULL) & (frame["N"] == 40),
              "Median latency (ms)"] = 0.
    assert set(_speedup_frame(frame)["N"]) == {20}
    doubled = pd.concat((frame, frame.iloc[[0]]), ignore_index=True)
    assert _speedup_frame(doubled).empty



def test_speedup_rejects_zero_or_nonfinite_pruned_latency():
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
def test_gap_display_preserves_nonzero_precision(value, expected):
    assert _price_gap_label(value) == expected
