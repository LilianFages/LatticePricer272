"""Lot 3B.1 — Chart sizing and bottom-legend regressions."""
from __future__ import annotations

import pandas as pd

from frontend.research_efficiency_views import EFFICIENCY_CHART_HEIGHT, _chart


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
