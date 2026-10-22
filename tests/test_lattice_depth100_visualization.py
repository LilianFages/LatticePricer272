"""Lot 2B.3 — adaptive full-node lattice at N=100 and honest ghost controls."""
from __future__ import annotations

import altair as alt
import pytest

from frontend.lattice_explorer import (
    _chart, _has_ghost_allocations, _lattice_chart_data, _plot_domain,
)
from frontend.lattice_explorer_service import MAX_VISUAL_STEPS, snapshot_priced_tree
from frontend.contract_workflow import assess_transfer
from frontend.pricing_service import PricingRequest


def test_visual_cap_is_one_hundred_and_transfer_is_exact():
    assert MAX_VISUAL_STEPS == 100
    assert assess_transfer(PricingRequest(steps=100), "lattice_explorer").exact
    check = assess_transfer(PricingRequest(steps=101), "lattice_explorer")
    assert not check.exact
    assert any("N=101" in issue for issue in check.issues)


def test_allocation_filter_is_noop_for_fully_connected_tree():
    snap = snapshot_priced_tree(PricingRequest(steps=20, model="Binomial"))
    assert snap.unreached_allocations == 0
    assert not _has_ghost_allocations([snap])
    assert _plot_domain([snap], True) == _plot_domain([snap], False)


def test_allocation_filter_reveals_real_branch_pruning_ghosts():
    snap = snapshot_priced_tree(PricingRequest(
        steps=10, model="Trinomial · Branch pruning", pruning_threshold=1e-5,
    ))
    assert snap.unreached_allocations > 0
    assert _has_ghost_allocations([snap])
    assert sum(not node.connected for node in snap.allocated_geometry) == snap.unreached_allocations
    assert snap.measured_allocations > snap.displayed_nodes
    domain = _plot_domain([snap], True)
    with alt.data_transformers.enable("default", max_rows=None):
        all_nodes = _chart(snap, domain, all_allocated=True,
                           height=350, show_links=True).to_dict()
        connected = _chart(snap, domain, all_allocated=False,
                           height=350, show_links=True).to_dict()
    assert len(all_nodes["layer"]) == len(connected["layer"]) + 1


def test_hundred_step_trinomial_nodes_can_render_without_heavy_edges():
    snap = snapshot_priced_tree(PricingRequest(
        steps=100, model="Trinomial · Allocation pruning",
    ))
    assert 5000 < snap.displayed_nodes <= 10201
    assert 0 < len(snap.edges) <= 30000
    nodes, links = _lattice_chart_data(snap, include_links=False)
    assert len(nodes) == snap.displayed_nodes
    assert links == []
    # Altair ordinarily rejects >5000 DataFrame rows: the UI's scoped
    # transformer override must support all actual priced nodes.
    with alt.data_transformers.enable("default", max_rows=None):
        chart = _chart(snap, _plot_domain([snap], False),
                       all_allocated=False, height=450, show_links=False)
        spec = chart.to_dict()
    assert "layer" in spec
    assert len(spec["layer"]) >= 1
