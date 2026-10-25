"""Lot 4B.1 -- empirical Pareto frontier and same-N runtime speed-up."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from frontend.analytics_views import _accuracy_cost_chart, _pareto_path_frame
from frontend.research_efficiency_service import ALLOCATION, BRANCH, FULL
from frontend.research_efficiency_views import _price_gap_label, _speedup_frame


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


def _pareto_data():
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


def test_pareto_vertices_global_and_sorted():
    p = _pareto_path_frame(_pareto_data().iloc[::-1])
    assert p["Runtime (ms)"].tolist() == [1., 3., 7.]
    assert p["Absolute gap (€)"].tolist() == [.1, .05, .01]
    assert p["Model"].tolist() == ["Binomial", "Trinomial", "Trinomial"]


def test_identical_frontier_points_do_not_duplicate_line_vertices():
    src = _pareto_data()
    extra = src.iloc[[0]].copy()
    extra.loc[extra.index[0], "Model"] = "Other model"
    path = _pareto_path_frame(pd.concat((src, extra), ignore_index=True))
    assert len(path) == 3


def _layer_data(spec, index):
    source = spec["layer"][index]["data"]
    return (source["values"] if "values" in source
            else spec["datasets"][source["name"]])


def test_chart_contains_envelope_model_points_and_rings():
    spec = _accuracy_cost_chart(_pareto_data(), pareto_only=False,
                                gap_label="Absolute Black–Scholes error (€)").to_dict()
    assert len(spec["layer"]) == 3
    line, points, rings = spec["layer"]
    assert line["mark"]["type"] == "line"
    assert line["mark"]["strokeDash"]
    assert points["mark"]["type"] == "circle"
    assert points["encoding"]["color"]["field"] == "Model"
    assert points["encoding"]["color"]["legend"]["orient"] == "bottom"
    assert rings["mark"]["type"] == "point"
    assert len(_layer_data(spec, 0)) == 3
    assert len(_layer_data(spec, 1)) == 4


def test_pareto_only_is_view_filter_without_changing_frontier():
    spec = _accuracy_cost_chart(_pareto_data(), pareto_only=True,
                                gap_label="Absolute price gap (€)").to_dict()
    assert len(_layer_data(spec, 0)) == 3
    assert len(_layer_data(spec, 1)) == 3
