"""Independent event-time exercise audit for American cash-dividend options.

No lattice pricing equations are duplicated here. The volatility-zero
benchmark is an exact continuous-exercise reference for deterministic cash
jumps and serves to detect missed pre-exercise decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.dividend_exercise import aligned_event_step
from src.risk.contracts import RiskRequest


@dataclass(frozen=True, slots=True)
class DividendExerciseAudit:
    warning: str | None
    deterministic_reference: float | None = None
    model_minus_reference: float | None = None
    optimal_exercise_event: str | None = None


def _intrinsic(spot: float, strike: float, kind: str) -> float:
    """Return the intrinsic payoff for a deterministic scenario."""
    if kind == "Call":
        return max(spot - strike, 0.0)
    if kind == "Put":
        return max(strike - spot, 0.0)
    raise ValueError(f"Unknown option type: {kind}.")


def _event_exercise_candidates(
    request: RiskRequest,
    stock_before: float,
    event_time: float,
    cash_amount: float,
) -> tuple[float, float]:
    """Discount deterministic intrinsic payoffs on both sides of a cash jump."""
    discount = math.exp(-request.rate * event_time)
    before = discount * _intrinsic(
        stock_before, request.strike, request.option_type,
    )
    after = discount * _intrinsic(
        stock_before - cash_amount, request.strike, request.option_type,
    )
    return before, after


def _exercise_at_cash_jump(
    request: RiskRequest,
    spot_before: float,
    dividend_time: float,
    amount: float,
    current_best: float,
    current_event: str,
) -> tuple[float, str, float]:
    """Compare event-side payoffs and preserve the selected event label."""
    before, after = _event_exercise_candidates(
        request, spot_before, dividend_time, amount,
    )
    if before > current_best:
        current_best, current_event = before, "pre-dividend"
    stock_after = spot_before - amount
    if after > current_best:
        current_best, current_event = after, "post-dividend"
    return current_best, current_event, stock_after


def deterministic_american_reference(request: RiskRequest) -> tuple[float, str]:
    """Exact American value at sigma=0: exercise at event sides or endpoints.

    Discounted intrinsic value is monotone between consecutive ex-dates while
    in the money; the maximum is therefore among those event boundaries.
    """
    if request.volatility != 0.0:
        raise ValueError("The deterministic reference requires zero volatility.")
    if request.exercise != "American":
        raise ValueError("American exercise is required for this benchmark.")

    # Exercise at valuation is the initial admissible candidate.
    best = _intrinsic(request.spot, request.strike, request.option_type)
    best_event = "valuation"
    current_spot = request.spot
    previous_date = 0.0
    for dividend in sorted(request.dividends, key=lambda event: event.time):
        current_spot *= math.exp(request.rate * (dividend.time - previous_date))
        best, best_event, current_spot = _exercise_at_cash_jump(
            request, current_spot, dividend.time, dividend.amount,
            best, best_event,
        )
        previous_date = dividend.time

    # Expiry completes the finite set of exercise-date candidates.
    current_spot *= math.exp(request.rate * (request.maturity - previous_date))
    terminal = math.exp(-request.rate * request.maturity) * _intrinsic(
        current_spot, request.strike, request.option_type,
    )
    if terminal > best:
        best, best_event = terminal, "expiry"
    return best, best_event


def assess_dividend_exercise(
    request: RiskRequest,
    model_price: float,
) -> DividendExerciseAudit:
    """Disclose off-grid exercise resolution, and independent sigma=0 gaps."""
    if request.exercise != "American" or not request.dividends:
        return DividendExerciseAudit(warning=None)

    # An off-grid event needs a qualified exercise-policy disclosure.
    off_grid = sum(
        aligned_event_step(dividend.time, request.maturity, request.steps) is None
        for dividend in request.dividends
    )
    warning = None
    if off_grid:
        warning = (
            f"American dividend exercise: {off_grid} off-grid ex-date(s). "
            "The engine includes a nonanticipative precommitted exercise "
            "strategy at the actual ex-date. For zero volatility this is "
            "exact; for stochastic prices it is a conservative lower bound, "
            "NOT the full optimal event-time exercise policy. "
            "Check convergence with more time steps."
        )

    if request.volatility != 0.0:
        return DividendExerciseAudit(warning=warning)

    # Sigma-zero admits an independent exact reference valuation.
    reference, event = deterministic_american_reference(request)
    gap = model_price - reference
    if not math.isclose(model_price, reference, rel_tol=1e-10, abs_tol=1e-10):
        supplement = (
            f"Independent deterministic American reference: "
            f"{reference:.6f} EUR (optimal {event}); "
            f"model minus reference {gap:+.6f} EUR."
        )
        warning = (warning + " " if warning else "") + supplement
    return DividendExerciseAudit(
        warning=warning,
        deterministic_reference=reference,
        model_minus_reference=gap,
        optimal_exercise_event=event,
    )
