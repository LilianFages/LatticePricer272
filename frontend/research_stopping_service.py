"""On-demand, apples-to-apples cash-dividend comparisons for exercise analytics.

The two scenarios share spot, strike, rate, volatility, maturity, grid, engine,
option kind, exercise style and pruning threshold. Only cash dividends differ.

The reported first-exercise probabilities describe *lattice-node* stopping.
Dividend-event exercise between nodes is not included in that distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from frontend.exercise_boundary_service import (
    MAX_BOUNDARY_STEPS, ExerciseStudy, run_exercise_study,
)
from frontend.pricing_service import (
    PricingInputError, PricingRequest, validate_request,
)


@dataclass(frozen=True, slots=True)
class DividendExerciseComparison:
    with_dividends: ExerciseStudy
    without_dividends: ExerciseStudy

    @property
    def premium_difference(self) -> float:
        """Difference with minus without dividends, at identical starting spot."""
        return self.with_dividends.premium - self.without_dividends.premium

    @property
    def node_exercise_difference(self) -> float:
        """Diagnostic only; NOT the change in complete event-time stopping mass."""
        return (self.with_dividends.early_exercise_probability
                - self.without_dividends.early_exercise_probability)


def compare_dividend_exercise(
    request: PricingRequest,
    *,
    analyzer: Callable[[PricingRequest], ExerciseStudy] = run_exercise_study,
) -> DividendExerciseComparison:
    """Evaluate two deliberately comparable lattice contracts, only on demand.

    Preserve any validation failure; never fabricate a zero-dividend premium or
    silently adjust the spot/forward to neutralize dividends.
    """
    if not request.dividends:
        raise PricingInputError(
            "Add at least one cash dividend to compare with a no-dividend contract."
        )
    validate_request(request)
    if request.steps > MAX_BOUNDARY_STEPS:
        raise PricingInputError(
            f"Optimal Stopping comparison allows at most N={MAX_BOUNDARY_STEPS}."
        )
    baseline = replace(request, dividends=())
    return DividendExerciseComparison(
        with_dividends=analyzer(request),
        without_dividends=analyzer(baseline),
    )
