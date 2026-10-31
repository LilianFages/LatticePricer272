"""On-demand pruning/complexity diagnostics on real pricing engines.

Full and pruned trees are evaluated at equal N and with identical contracts.
Allocation counts are measured on actual Node objects from independent builds.
Timing medians are based on separate normal pricing runs (no tracemalloc).
Optional tracemalloc peak Python memory uses a separate pricing run; it is
neither RSS nor a measure of total process memory.
"""
from __future__ import annotations

import math
import statistics
import tracemalloc
from dataclasses import dataclass, replace
from typing import Callable, Sequence

from frontend.lattice_allocation import capture_allocated_columns
from frontend.pricing_service import (
    PricingFunction, PricingInputError, PricingRequest, PricingResult, _build_tree,
    price_contract, validate_request,
)

FULL = "Trinomial · Full"
BRANCH = "Trinomial · Branch pruning"
ALLOCATION = "Trinomial · Allocation pruning"
EFFICIENCY_MODELS = (FULL, BRANCH, ALLOCATION)
MAX_RESEARCH_N = 150  # Interactive workload bound, not an engine restriction.


@dataclass(frozen=True, slots=True)
class EfficiencyPoint:
    model: str
    steps: int
    status: str
    price: float | None
    median_ms: float | None
    allocated_nodes: int | None
    linked_nodes: int | None
    peak_python_mib: float | None
    absolute_gap_to_full: float | None = None
    allocation_savings_pct: float | None = None


@dataclass(frozen=True, slots=True)
class EfficiencyReport:
    request: PricingRequest
    points: tuple[EfficiencyPoint, ...]
    repetitions: int
    measured_peak_memory: bool


def count_allocated_nodes(request: PricingRequest) -> int:
    """Count actual objects created in every column; no model-price timing."""
    validate_request(request)
    tree = _build_tree(request)
    with capture_allocated_columns(tree) as columns:
        tree.build()
    if len(columns) != request.steps + 1:
        raise RuntimeError("Incomplete measured allocation trace.")
    count = sum(len(column) for column in columns)
    if count < 1:
        raise RuntimeError("No allocated lattice nodes found.")
    reported = getattr(tree, "allocated_node_count", None)
    if reported is not None and count != reported:
        raise RuntimeError("Measured allocations disagree with engine counter.")
    return count


def peak_python_memory_mib(
    request: PricingRequest, *, pricer: PricingFunction = price_contract,
) -> tuple[float, float]:
    """Price once under tracemalloc, independently of latency measurements."""
    if tracemalloc.is_tracing():
        raise RuntimeError("An external tracemalloc session is already active.")
    tracemalloc.start()
    try:
        result = pricer(request)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return float(result.price), float(peak) / (1024.0 ** 2)


def _validate_grid(steps_grid: Sequence[int], repetitions: int) -> tuple[int, ...]:
    ns = tuple(steps_grid)
    if not 2 <= len(ns) <= 6:
        raise PricingInputError("Choose 2–6 time-step depths.")
    if (any(type(n) is not int or not 2 <= n <= MAX_RESEARCH_N for n in ns)
            or tuple(sorted(set(ns))) != ns):
        raise PricingInputError("Depths must be unique, increasing integers in [2, 150].")
    if type(repetitions) is not int or repetitions not in (1, 3, 5):
        raise PricingInputError("Timing repetitions must be 1, 3 or 5.")
    return ns


def _run_one(
    request: PricingRequest,
    repetitions: int,
    measure_memory: bool,
    pricer: PricingFunction,
    allocation_counter: Callable[[PricingRequest], int],
    memory_probe: Callable[[PricingRequest], tuple[float, float]],
) -> EfficiencyPoint:
    try:
        # Timing runs are independent of the optional tracemalloc valuation.
        timed_results: list[PricingResult] = [pricer(request) for _ in range(repetitions)]
        prices = [float(result.price) for result in timed_results]
        if any(not math.isclose(price, prices[0], rel_tol=1e-10, abs_tol=1e-10)
               for price in prices[1:]):
            raise RuntimeError("Independent timing runs returned inconsistent premiums.")
        elapsed = [float(result.elapsed_ms) for result in timed_results]
        if not all(math.isfinite(x) and x >= 0 for x in elapsed):
            raise RuntimeError("Non-finite measured latency.")
        # Count actual allocations, not merely reachable priced nodes.
        allocated = allocation_counter(request)
        peak: float | None = None
        if measure_memory:
            memory_price, peak = memory_probe(request)
            if not math.isclose(memory_price, prices[0], rel_tol=1e-10, abs_tol=1e-10):
                raise RuntimeError("Memory-run premium differs from timed run.")
        return EfficiencyPoint(
            model=request.model, steps=request.steps, status="ok",
            price=prices[0], median_ms=statistics.median(elapsed),
            allocated_nodes=int(allocated),
            linked_nodes=timed_results[-1].reachable_nodes,
            peak_python_mib=peak,
        )
    except (PricingInputError, ValueError, RuntimeError, ArithmeticError,
            OverflowError, ZeroDivisionError, RecursionError) as exc:
        return EfficiencyPoint(request.model, request.steps,
                               f"{type(exc).__name__}: {exc}",
                               None, None, None, None, None)


def _relative_to_full(points: Sequence[EfficiencyPoint]) -> tuple[EfficiencyPoint, ...]:
    """Attach gaps/savings only to a successful FULL model at identical N."""
    # A reference price must use the same N and a successful full engine.
    originals = {point.steps: point for point in points
                 if point.model == FULL and point.status == "ok"}
    decorated: list[EfficiencyPoint] = []
    for point in points:
        ref = originals.get(point.steps)
        if point.status != "ok" or ref is None:
            decorated.append(point)
            continue
        gap = abs(point.price - ref.price)
        savings = (100.0 * (1.0 - point.allocated_nodes / ref.allocated_nodes)
                   if ref.allocated_nodes else None)
        decorated.append(replace(point, absolute_gap_to_full=gap,
                                 allocation_savings_pct=savings))
    return tuple(decorated)


def efficiency_study(
    request: PricingRequest,
    steps_grid: Sequence[int] = (20, 40, 80, 100),
    repetitions: int = 3,
    measure_memory: bool = False,
    *,
    pricer: PricingFunction = price_contract,
    allocation_counter: Callable[[PricingRequest], int] = count_allocated_nodes,
    memory_probe: Callable[[PricingRequest], tuple[float, float]] = peak_python_memory_mib,
) -> EfficiencyReport:
    """Reprice three trinomial architectures with fixed contract and pruning.

    The unpruned full tree is a same-N comparator, not a true-price oracle.
    Failed model/depth combinations remain explicit missing observations.
    """
    ns = _validate_grid(steps_grid, repetitions)
    # Preserve cash dividends and all contract details; validate with a model
    # supported by the study independently of the request's original model.
    validate_request(replace(request, model=FULL, steps=ns[-1], pruning_threshold=0.0))
    points: list[EfficiencyPoint] = []
    for n in ns:
        for model in EFFICIENCY_MODELS:
            candidate = replace(request, model=model, steps=n,
                                pruning_threshold=(0.0 if model == FULL
                                                   else request.pruning_threshold))
            points.append(_run_one(candidate, repetitions, measure_memory,
                                   pricer, allocation_counter, memory_probe))
    return EfficiencyReport(request, _relative_to_full(points), repetitions,
                            measure_memory)
