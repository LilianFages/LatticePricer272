"""Lot 2B.7: exact local links, no disconnected/fictional sampled edges."""
from __future__ import annotations

from dataclasses import replace

import pytest

from frontend.lattice_explorer import focus_lattice_snapshot, _chart
from frontend.lattice_explorer_service import snapshot_priced_tree
from frontend.pricing_service import PricingRequest


def test_n100_local_spot_corridor_is_exact_and_lightweight():
    base = snapshot_priced_tree(PricingRequest(
        steps=100, model="Trinomial · Allocation pruning",
    ))
    window = (45, 55)
    band = (85.0, 115.0)
    focused = focus_lattice_snapshot(base, window, band)
    originals = {n.node_id for n in base.nodes}
    ids = {n.node_id for n in focused.nodes}
    assert ids <= originals
    assert ids == {n.node_id for n in base.nodes
                   if window[0] <= n.step <= window[1] and band[0] <= n.price <= band[1]}
    assert focused.edges == tuple(e for e in base.edges
                                  if window[0] <= e.step < window[1]
                                  and e.from_id in ids and e.to_id in ids)
    assert focused.edges
    assert all(e in base.edges for e in focused.edges)
    assert all(e.from_id in ids and e.to_id in ids for e in focused.edges)
    assert len(focused.nodes) < 450
    assert len(focused.edges) < 1200
    assert len(base.nodes) > len(focused.nodes)
    assert base.model_price == focused.model_price
    assert base.request == focused.request
    # Chart serializes only the focused induced subgraph, including every edge.
    fig = _chart(focused, band, all_allocated=False, height=370,
                 show_links=True, step_window=window, max_links=None)
    assert "layer" in fig.to_dict()


def test_full_tree_remains_unchanged_and_view_is_deterministic():
    base = snapshot_priced_tree(PricingRequest(steps=20, model="Binomial"))
    old_nodes, old_edges = base.nodes, base.edges
    a = focus_lattice_snapshot(base, (7, 11), (88.0, 116.0))
    b = focus_lattice_snapshot(base, (7, 11), (88.0, 116.0))
    assert a.nodes == b.nodes
    assert a.edges == b.edges
    assert base.nodes is old_nodes
    assert base.edges is old_edges
    assert base.displayed_nodes > a.displayed_nodes


def test_branch_pruning_ghosts_are_real_and_local_only():
    base = snapshot_priced_tree(PricingRequest(
        steps=10, model="Trinomial · Branch pruning", pruning_threshold=1e-5,
    ))
    assert base.unreached_allocations > 0
    focus = focus_lattice_snapshot(base, (0, 10), (0.0, 10000.0))
    assert focus.nodes == base.nodes
    assert focus.edges == base.edges
    assert focus.allocated_geometry == base.allocated_geometry


def test_invalid_focus_bounds_fail_explicitly():
    base = snapshot_priced_tree(PricingRequest(steps=5, model="Binomial"))
    for window, band in [((-1, 3), (80., 120.)), ((4, 3), (80., 120.)),
                         ((0, 6), (80., 120.)), ((0, 4), (120., 80.)),
                         ((0, 4), (-10., 120.))]:
        with pytest.raises(ValueError):
            focus_lattice_snapshot(base, window, band)
