"""Regression checks for actual allocation geometry and simplified links."""

from dataclasses import replace

import pytest

from src.node import Node
from frontend.lattice_allocation import _column_nodes, capture_allocated_columns
from frontend.lattice_explorer_service import snapshot_priced_tree
from frontend.pricing_service import PricingRequest


def test_column_reader_keeps_distinct_nodes_and_detects_cycle() -> None:
    a, b = Node(110), Node(90)
    a.lower_neighbor = b
    assert _column_nodes(a) == (a, b)
    b.lower_neighbor = a
    with pytest.raises(RuntimeError, match="Cycle"):
        _column_nodes(a)


@pytest.mark.parametrize("model,expected", [
    ("Binomial", 21),
    ("Hybrid", 21),
    ("Trinomial · Full", 36),
    ("Trinomial · Branch pruning", 36),
    ("Trinomial · Allocation pruning", 36),
])
def test_actual_allocation_trace_on_five_step_engines(
    model: str, expected: int
) -> None:
    snap = snapshot_priced_tree(PricingRequest(model=model, steps=5))
    assert snap.measured_allocations == expected
    assert snap.unreached_allocations == 0
    assert snap.displayed_nodes == expected
    assert len({(n.step, n.node_id) for n in snap.allocated_geometry}) == expected
    assert all(n.connected for n in snap.allocated_geometry)
    if snap.engine_allocated_nodes is not None:
        assert snap.engine_allocated_nodes == expected


def test_branch_and_allocation_are_geometrically_distinct_at_n10() -> None:
    req = PricingRequest(
        model="Trinomial · Branch pruning", steps=10,
        option_type="Put", exercise="American", pruning_threshold=1e-5,
    )
    branch = snapshot_priced_tree(req)
    allocation = snapshot_priced_tree(replace(
        req, model="Trinomial · Allocation pruning"
    ))
    assert branch.measured_allocations == 121
    assert branch.displayed_nodes == 113
    assert allocation.measured_allocations == 113
    assert branch.unreached_allocations == (
        branch.measured_allocations - branch.displayed_nodes
    )
    assert allocation.measured_allocations < branch.measured_allocations
    assert allocation.engine_allocated_nodes == allocation.measured_allocations
    assert branch.model_price == pytest.approx(allocation.model_price, abs=1e-8)
    assert any(edge.simplified for edge in branch.edges)
    assert any(node.simplified for node in branch.nodes)
    assert any(not item.connected for item in branch.allocated_geometry)


def test_wrapper_is_restored_after_success_and_failure() -> None:
    from src.trinomial_tree import TrinomialTree

    tree = TrinomialTree(100.0, 0.02, 0.2, 1.0, 3)
    assert "_build_column" not in vars(tree)
    with capture_allocated_columns(tree) as columns:
        tree.build()
    assert len(columns) == 4
    assert "_build_column" not in vars(tree)

    other = TrinomialTree(100.0, 0.02, 0.2, 1.0, 3)
    with pytest.raises(ZeroDivisionError):
        with capture_allocated_columns(other):
            raise ZeroDivisionError("forced test failure")
    assert "_build_column" not in vars(other)


def test_no_synthetic_allocations_present() -> None:
    for model in ("Trinomial · Branch pruning", "Trinomial · Allocation pruning"):
        snap = snapshot_priced_tree(PricingRequest(
            model=model, steps=9, pruning_threshold=1e-3,
        ))
        assert snap.measured_allocations >= snap.displayed_nodes
        assert snap.unreached_allocations == (
            snap.measured_allocations - snap.displayed_nodes
        )
        assert all(n.step <= 9 for n in snap.allocated_geometry)
        assert all(n.option_value is not None for n in snap.allocated_geometry if n.connected)
