"""Observed runtime/price-gap tradeoffs in a *live* convergence study.

The finite-N numerical comparator is not an exact price. Runtime observations
come from one engine valuation per configuration, not repeated benchmarks.
Neither a Pareto label nor a user-selected target is an accuracy guarantee.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from frontend.analytics_service import ConvergenceResult
from src.accuracy_metrics import nondominated_indices


@dataclass(frozen=True, slots=True)
class AccuracyCostPoint:
    model: str
    steps: int
    premium: float
    runtime_ms: float
    absolute_gap: float
    pareto: bool


def accuracy_cost_points(outcome: ConvergenceResult) -> tuple[AccuracyCostPoint, ...]:
    """Compute observed nondominance; preserve the existing failure filters."""
    reference = outcome.reference_price
    if reference is None or not math.isfinite(reference):
        return ()
    observations = []
    for point in outcome.points:
        if point.status != "ok" or point.price is None or point.runtime_ms is None:
            continue
        price, elapsed, n = point.price, point.runtime_ms, point.coordinate
        if (not all(math.isfinite(v) for v in (price, elapsed, n))
                or price < 0 or elapsed < 0 or not float(n).is_integer() or n < 1):
            continue
        observations.append((point.model, int(n), float(price),
                             float(elapsed), abs(float(price) - reference)))
    pairs = [(elapsed, gap) for _, _, _, elapsed, gap in observations]
    frontier = set(nondominated_indices(pairs))
    return tuple(
        AccuracyCostPoint(model, n, price, elapsed, gap, index in frontier)
        for index, (model, n, price, elapsed, gap) in enumerate(observations)
    )

def best_observed_under_target(
    points: Sequence[AccuracyCostPoint], target_eur: float,
) -> AccuracyCostPoint | None:
    """Cheapest *observed* qualifying configuration; no out-of-sample promise."""
    if not math.isfinite(target_eur) or target_eur < 0:
        raise ValueError("The price-gap target must be finite and nonnegative.")
    candidates = (p for p in points if p.absolute_gap <= target_eur)
    return min(candidates,
               key=lambda p: (p.runtime_ms, p.absolute_gap, p.steps, p.model),
               default=None)
