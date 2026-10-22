"""Lot 2B.5 — optimized overview, exact local geometry and ex-date visibility."""
from __future__ import annotations

from collections import Counter

from frontend.lattice_plot_sampling import overview_step_indices, select_chart_rows
from frontend.lattice_explorer import _chart, _lattice_chart_data, _plot_domain
from frontend.lattice_explorer_service import snapshot_priced_tree
from frontend.pricing_service import DividendInput, PricingRequest


def test_n100_overview_timeline_is_sparse_and_keeps_edges():
    steps = overview_step_indices(100)
    assert 0 in steps and 100 in steps
    assert len(steps) <= 40
    assert len(steps) >= 30


def test_exdividend_neighbour_layers_never_lost():
    steps = overview_step_indices(100, event_times=(0.505, 0.75), maturity=1.0)
    assert {0, 50, 51, 75, 100} <= steps


def test_small_steps_have_no_sampling_of_time():
    assert overview_step_indices(19) == frozenset(range(20))
    assert overview_step_indices(20) == frozenset(range(21))


def test_overview_preserves_exercise_states_with_small_number_of_marks():
    import pandas as pd
    rows = pd.DataFrame([
        {"step": step, "stock": i, "state": ("Early exercise" if i < 30 else
         "Continuation" if i < 60 else "Simplified" if i < 90 else "Maturity")}
        for step in range(101) for i in range(101)
    ])
    steps = overview_step_indices(100)
    sample = select_chart_rows(
        rows.loc[rows["step"].isin(steps)], overview=True, per_step=8,
    )
    assert len(sample) <= 40 * 8
    assert Counter(sample["step"]).keys() == steps
    assert (sample.groupby("step")["state"].nunique() == 4).all()
    assert (sample.groupby("step")["stock"].min() == 0).all()
    assert (sample.groupby("step")["stock"].max() == 100).all()
    assert len(rows) == 10201


def test_real_engine_still_keeps_all_nodes_and_full_local_window():
    request = PricingRequest(model="Trinomial · Allocation pruning", steps=100)
    snap = snapshot_priced_tree(request)
    full, _ = _lattice_chart_data(snap, include_links=False)
    selected = overview_step_indices(request.steps)
    sparse = select_chart_rows(
        full.loc[full["step"].isin(selected)], overview=True, per_step=8,
    )
    assert len(full) == snap.displayed_nodes
    assert len(sparse) <= 40 * 8
    local = select_chart_rows(full, step_window=(48, 54))
    assert len(local) == sum(n.step in range(48, 55) for n in snap.nodes)
    figure = _chart(snap, _plot_domain([snap], False),
                    all_allocated=False, height=370, show_links=False, overview=True)
    assert "layer" in figure.to_dict()
