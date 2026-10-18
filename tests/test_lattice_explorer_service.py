"""Lattice Explorer integration and node-geometry correctness checks."""

import math
from dataclasses import replace

import pytest

from src.node import Node
from frontend.lattice_explorer_service import (
    MAX_VISUAL_STEPS, _exercise_state, _successors, snapshot_priced_tree,
)
from frontend.pricing_service import PricingInputError, PricingRequest


def test_group_recombined_branches() -> None:
    root = Node(100)
    next_node = Node(102)
    root.next_up = next_node
    root.next_mid = next_node
    root.next_down = next_node
    root.up_probability = 0.0
    root.mid_probability = 1.0
    root.down_probability = 0.0
    edges = _successors(root)
    assert len(edges) == 1
    assert edges[0][0] == "Mid"
    assert edges[0][2] == pytest.approx(1.0)


def test_positive_parallel_branches_sum_probabilities() -> None:
    root = Node(100)
    next_node = Node(102)
    root.next_up = root.next_down = next_node
    root.up_probability = root.down_probability = 0.5
    edges = _successors(root)
    assert len(edges) == 1
    assert edges[0][2] == pytest.approx(1.0)


def test_probability_mass_rejection() -> None:
    root = Node(100)
    root.next_up = Node(105)
    root.up_probability = 0.7
    with pytest.raises(RuntimeError, match="mass"):
        _successors(root)


def test_early_exercise_decision_from_priced_successors() -> None:
    n = Node(60)
    a, b = Node(65), Node(55)
    a.option_value = 1.0
    b.option_value = 1.0
    n.next_up, n.next_down = a, b
    n.up_probability = n.down_probability = 0.5
    request = PricingRequest(
        steps=2, option_type="Put", exercise="American", strike=100
    )
    assert _exercise_state(n, request, 1) == "Early exercise"
    assert _exercise_state(n, replace(request, exercise="European"), 1) == "Continuation"
    assert _exercise_state(n, request, 2) == "Maturity"


@pytest.mark.parametrize("model", [
    "Binomial", "Trinomial · Full", "Trinomial · Branch pruning",
    "Trinomial · Allocation pruning", "Hybrid",
])
def test_real_tree_has_all_steps_and_unit_mass(model: str) -> None:
    request = PricingRequest(model=model, steps=4, exercise="European")
    snapshot = snapshot_priced_tree(request)
    assert len(snapshot.column_masses) == 5
    assert all(math.isclose(v, 1.0, abs_tol=1e-9) for v in snapshot.column_masses)
    assert set(n.step for n in snapshot.nodes) == set(range(5))
    assert snapshot.model_price >= 0
    assert snapshot.nodes[0].option_value == pytest.approx(snapshot.model_price)
    assert all(n.state != "Early exercise" for n in snapshot.nodes)
    assert all(0 <= n.reach_probability <= 1 + 1e-12 for n in snapshot.nodes)


def test_am_put_exercise_labels_present() -> None:
    snap = snapshot_priced_tree(PricingRequest(
        model="Binomial", option_type="Put", exercise="American",
        rate=0.08, volatility=0.10, steps=8, strike=115.0,
    ))
    assert any(n.state == "Early exercise" for n in snap.nodes)


def test_allocation_count_distinct_from_connected_nodes() -> None:
    snap = snapshot_priced_tree(PricingRequest(
        model="Trinomial · Allocation pruning", steps=8,
    ))
    assert snap.engine_allocated_nodes is not None
    assert snap.engine_allocated_nodes >= snap.displayed_nodes


def test_max_steps_guard_before_tree_construction() -> None:
    with pytest.raises(PricingInputError, match="at most"):
        snapshot_priced_tree(PricingRequest(steps=MAX_VISUAL_STEPS + 1))


def test_cash_dividend_binomial_failure_is_explicit() -> None:
    from frontend.pricing_service import DividendInput
    with pytest.raises(PricingInputError, match="does not support"):
        snapshot_priced_tree(PricingRequest(
            model="Binomial", steps=5,
            dividends=(DividendInput(0.5, 3.0),),
        ))
