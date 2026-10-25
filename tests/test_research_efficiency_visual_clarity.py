"""Lot 4B.2 — semantics and layout of the measured efficiency figures."""
import pandas as pd

from frontend.research_efficiency_service import ALLOCATION, BRANCH, FULL
from frontend.research_efficiency_views import (
    _allocation_nodes_chart, _overlapping_allocation_ns, _speedup_baseline_chart,
)


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


def test_patch_keeps_generic_chart_untouched():
    from frontend.research_efficiency_views import _chart, EFFICIENCY_CHART_HEIGHT
    spec = _chart(_points(), "N", "Allocated nodes").to_dict()
    assert spec["height"] == EFFICIENCY_CHART_HEIGHT
    assert spec["encoding"]["color"]["legend"]["columns"] == 2
