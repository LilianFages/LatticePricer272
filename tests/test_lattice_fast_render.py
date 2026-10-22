"""Lot 2B.4: large-N interactive chart uses representative view, full local detail."""
from __future__ import annotations

import pandas as pd
import pytest

from frontend.lattice_plot_sampling import select_chart_rows
from frontend.lattice_explorer import _chart, _plot_domain, _lattice_chart_data
from frontend.lattice_explorer_service import snapshot_priced_tree
from frontend.pricing_service import PricingRequest


def test_overview_has_bounded_number_of_marks_and_keeps_state_coverage():
    rows = pd.DataFrame([
        {"step": step, "stock": i, "state": "Early exercise" if i < 20 else "Continuation"}
        for step in range(101) for i in range(101)
    ])
    selected = select_chart_rows(rows, overview=True, per_step=10)
    assert len(selected) <= 1010
    assert selected.groupby("step").size().min() >= 2
    assert (selected.groupby("step")["state"].nunique() == 2).all()
    assert len(rows) == 10201


def test_detail_window_keeps_every_original_row_without_mutation():
    original = pd.DataFrame([{"step": t, "stock": i, "state": "Continuation"}
                             for t in range(101) for i in range(201)])
    detail = select_chart_rows(original, step_window=(48, 54))
    assert len(detail) == 7 * 201
    assert detail["step"].min() == 48 and detail["step"].max() == 54
    assert len(original) == 101 * 201


def test_invalid_window_is_rejected():
    with pytest.raises(ValueError):
        select_chart_rows(pd.DataFrame([{"step": 2}]), step_window=(3, 1))


def test_n100_chart_overview_and_local_window_build_without_full_node_payload():
    req = PricingRequest(model="Trinomial · Allocation pruning", steps=100)
    snap = snapshot_priced_tree(req)
    assert snap.displayed_nodes > 5000
    domain = _plot_domain([snap], False)
    overview = _chart(snap, domain, all_allocated=False, height=370,
                      show_links=False, overview=True)
    # The raw service still exposes every node. Only chart presentation is sampled.
    full_nodes, _ = _lattice_chart_data(snap, include_links=False)
    assert len(full_nodes) == snap.displayed_nodes
    assert len(select_chart_rows(full_nodes, overview=True)) <= 1010
    assert "layer" in overview.to_dict()
    detail = _chart(snap, _plot_domain([snap], False, step_window=(48, 54)),
                    all_allocated=False, height=370, show_links=False,
                    step_window=(48, 54))
    assert "layer" in detail.to_dict()
    assert len(select_chart_rows(full_nodes, step_window=(48, 54))) <= 7 * 201
