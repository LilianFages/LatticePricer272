"""Shared node-exercise diagnostics following the pricing engine's dividend rules.

Event-time exercise can occur between lattice nodes. This module identifies
only immediate exercise at the observed node; it does not invent event-time
stopping probabilities or assign event exercise to a parent node.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any

from frontend.pricing_service import PricingRequest
from src.contracts.dividend import CashDividend
from src.contracts.dividend_exercise import (
    DividendStepEvent, before_dividend_value, dividend_exercise_steps,
    offgrid_exercise_bound,
)
from src.contracts.option import CallOption, PutOption


def dividend_events_for_request(request: PricingRequest) -> dict[int, DividendStepEvent]:
    """Apply the exact mapping used by the engine, without constructing a new tree."""
    if request.exercise != "American" or not request.dividends:
        return {}
    terms = SimpleNamespace(
        nb_steps=request.steps, maturity=request.maturity,
        dt=request.maturity / request.steps,
        rate=request.rate, volatility=request.volatility,
        dividends=tuple(CashDividend(d.time, d.amount) for d in request.dividends),
    )
    return dividend_exercise_steps(terms)


def immediate_node_exercise(
    node: Any,
    request: PricingRequest,
    step: int,
    dt: float,
    successors: Iterable[tuple[Any, float]],
    event_steps: Mapping[int, DividendStepEvent] | None = None,
) -> bool:
    """Identify immediate node exercise against event-aware continuation.

    This classification is distinct from exercising immediately before/after
    a future ex-date, which the engine can incorporate into continuation.
    """
    option = (CallOption if request.option_type == "Call" else PutOption)(
        request.strike, is_american=request.exercise == "American"
    )
    intrinsic = option.payoff(float(node.price))
    if intrinsic <= 0.0:
        return False
    if step == request.steps:
        return True
    if request.exercise != "American":
        return False

    events = dividend_events_for_request(request) if event_steps is None else event_steps
    event = events.get(step + 1)
    expected = math.fsum(
        probability * before_dividend_value(
            child, float(child.option_value), option, event
        )
        for child, probability in successors
    )
    hold = math.exp(-request.rate * dt) * expected
    hold = offgrid_exercise_bound(node, hold, option, event)
    scale = max(1.0, abs(intrinsic), abs(hold))
    return intrinsic > hold + 1e-10 * scale
