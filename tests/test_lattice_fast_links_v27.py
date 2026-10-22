"""Lot 2B.6: bounded visual links, exact underlying geometry intact."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from frontend.lattice_plot_sampling import representative_transitions
from frontend.lattice_explorer import _chart, _lattice_chart_data, _plot_domain
from frontend.lattice_explorer_service import snapshot_priced_tree
from frontend.pricing_service import PricingRequest


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


def test_real_n100_detail_chart_caps_edges_without_changing_snapshots():
    snap = snapshot_priced_tree(PricingRequest(
        steps=100, model="Trinomial · Allocation pruning",
    ))
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
