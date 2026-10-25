"""Regression: bounded lattice chart sampling, exact local views and real links.

Consolidates former fast-render / v26 / v27 test modules. Their behavioral
invariants are preserved; one overlapping N=100 scenario is merged into the
rendering test. Snapshot fixture is read-only and shared within this module.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd
import pytest

from frontend.lattice_plot_sampling import (
    overview_step_indices, representative_transitions, select_chart_rows,
)
from frontend.lattice_explorer import _chart, _lattice_chart_data, _plot_domain
from frontend.lattice_explorer_service import snapshot_priced_tree
from frontend.pricing_service import PricingRequest

# Lot 5B.7A — consolidated lattice sampling/links regressions.


@pytest.fixture(scope="module")
def n100_snapshot():
    """One real priced N=100 tree for read-only chart assertions."""
    return snapshot_priced_tree(PricingRequest(
        model="Trinomial · Allocation pruning", steps=100,
    ))


# --- Temporal and state sampling ----------------------------------


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


# --- Chart rendering and exact focus windows ----------------------------------


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


def test_n100_chart_overview_and_local_window_build_without_full_node_payload(n100_snapshot):
    req = n100_snapshot.request
    snap = n100_snapshot
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
    # Sparse overview never erases the exact selected temporal layers.
    steps = overview_step_indices(req.steps)
    sparse = select_chart_rows(
        full_nodes.loc[full_nodes["step"].isin(steps)],
        overview=True, per_step=8,
    )
    assert len(sparse) <= 40 * 8
    local = select_chart_rows(full_nodes, step_window=(48, 54))
    assert len(local) == sum(48 <= node.step <= 54 for node in snap.nodes)


# --- Bounded and authentic transition links ----------------------------------


@dataclass(frozen=True)
class Edge:
    step: int
    from_price: float
    to_price: float
    from_id: str
    to_id: str
    kind: str = "Up"
    simplified: bool = False


def test_bounded_links_are_real_and_spread_across_layers():
    original = tuple(
        Edge(step, float(i), float(i + 1), f"{step}-{i}", f"{step + 1}-{i}",
             simplified=(i == 79))
        for step in range(47, 54) for i in range(80)
    )
    sampled = representative_transitions(
        original, step_window=(47, 54), max_links=140,
    )
    assert len(sampled) == 140
    assert set(sampled) <= set(original)
    assert {edge.step for edge in sampled} == set(range(47, 54))
    assert all(any(edge.simplified for edge in sampled if edge.step == step)
               for step in range(47, 54))
    assert all(min(edge.from_price for edge in sampled if edge.step == step) == 0
               for step in range(47, 54))
    assert all(max(edge.from_price for edge in sampled if edge.step == step) == 79
               for step in range(47, 54))
    assert sampled == representative_transitions(
        original, step_window=(47, 54), max_links=140,
    )
    assert len(original) == 7 * 80


def test_small_link_counts_are_complete_not_resampled():
    edges = (Edge(0, 100, 110, "a", "b"), Edge(0, 100, 90, "a", "c"))
    assert representative_transitions(edges, max_links=240) == edges
    with pytest.raises(ValueError):
        representative_transitions(edges, max_links=1)


def test_real_n100_detail_chart_caps_edges_without_changing_snapshots(n100_snapshot):
    snap = n100_snapshot
    original_count = len(snap.edges)
    window = (47, 54)
    real_links = [edge for edge in snap.edges if window[0] <= edge.step < window[1]]
    assert len(real_links) > 240
    sampled = representative_transitions(
        snap.edges, step_window=window, max_links=240,
    )
    nodes, records = _lattice_chart_data(
        snap, include_links=True, step_window=window, max_links=240,
    )
    assert len(records) == 2 * len(sampled)
    assert len(records) <= 480
    assert len(nodes) == snap.displayed_nodes
    figure = _chart(
        snap, _plot_domain([snap], False, window),
        all_allocated=False, height=360, show_links=True,
        step_window=window, max_links=240,
    )
    assert "layer" in figure.to_dict()
    assert len(snap.edges) == original_count
    assert set(sampled).issubset(set(snap.edges))
