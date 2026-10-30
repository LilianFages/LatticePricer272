"""Regression: event-aware exercise labels are not fictitious parent exercises."""
from __future__ import annotations

import math
from dataclasses import replace

import pytest

from frontend.exercise_boundary_service import _exercise
from frontend.exercise_policy import dividend_events_for_request
from frontend.lattice_explorer_service import _exercise_state
from frontend.pricing_service import DividendInput, PricingRequest, _build_tree
from src.backward_pricer import BackwardPricer
from src.option import CallOption
from src.recursive_pricer import RecursivePricer


@pytest.mark.parametrize("model", ("Hybrid", "Trinomial · Allocation pruning"))
@pytest.mark.parametrize("ex_date,vol,amount,step", (
    (.50, 0.0, 20.0, 4),
    (.55, 0.0, 20.0, 5),
    (.50, .20, 8.0, 4),
    (.55, .20, 8.0, 5),
))
def test_cash_dividend_event_does_not_create_false_parent_exercise(
    model: str, ex_date: float, vol: float, amount: float, step: int,
) -> None:
    request = PricingRequest(
        spot=120.0, strike=100.0, rate=.05,
        volatility=vol, maturity=1.0, steps=10, model=model,
        option_type="Call", exercise="American",
        dividends=(DividendInput(ex_date, amount),),
    )
    tree = _build_tree(request)
    price = RecursivePricer().price(tree, CallOption(100., is_american=True))
    assert math.isfinite(price)
    layers = BackwardPricer._reachable_levels(tree.root, request.steps)
    event_steps = dividend_events_for_request(request)
    assert step + 1 in event_steps
    dt = request.maturity / request.steps
    old_false_positives = 0
    for node in layers[step]:
        intrinsic = max(node.price - request.strike, 0.)
        raw_hold = math.exp(-request.rate * dt) * math.fsum(
            p * float(child.option_value)
            for child, p in (
                (node.next_up, node.up_probability),
                (node.next_mid, node.mid_probability),
                (node.next_down, node.down_probability),
            ) if child is not None and p is not None
        )
        if intrinsic > raw_hold + 1e-10 * max(1., intrinsic, abs(raw_hold)):
            if float(node.option_value) > intrinsic + 1e-8:
                old_false_positives += 1
        # If the engine values continuing strictly more, an immediate
        # node exercise label is mathematically inconsistent.
        if float(node.option_value) > intrinsic + 1e-8:
            assert not _exercise(node, request, step, dt)
            assert _exercise_state(node, request, step) == "Continuation"
        assert (_exercise(node, request, step, dt)
                == (_exercise_state(node, request, step) == "Early exercise"))
    assert old_false_positives > 0


def test_contract_mapping_matches_engine_event_steps() -> None:
    from src.dividend_exercise import dividend_exercise_steps
    request = PricingRequest(
        model="Hybrid", steps=20, option_type="Put", exercise="American",
        dividends=(DividendInput(.25, 2.), DividendInput(.775, 3.)),
    )
    actual_tree = _build_tree(request)
    assert dividend_events_for_request(request) == dividend_exercise_steps(actual_tree)
    assert dividend_events_for_request(replace(request, exercise="European")) == {}
