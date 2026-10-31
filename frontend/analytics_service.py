"""Reproducible, bounded analytics on the existing LatticePricer272 engines.

No financial transition or pricing equation is reimplemented. All values are
obtained by fresh calls to ``price_contract``. Errors stay explicit, never zero.
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, replace
from typing import Literal, Sequence

from frontend.greeks_service import GreekBumps, analytic_bs_greeks, select_gamma_bump
from frontend.pricing_service import (
    MAX_STEPS, MODELS, PricingFunction, PricingInputError, PricingRequest, price_contract,
    validate_request,
)

ParameterName = Literal["spot", "volatility", "rate", "maturity"]
DEFAULT_AMPLITUDE: dict[str, float] = {
    "spot": 0.20,       # ±20% of base spot
    "volatility": 0.05, # ±5 volatility percentage points
    "rate": 0.02,       # ±2 rate percentage points
    "maturity": 0.25,   # ±25% of original maturity
}


@dataclass(frozen=True, slots=True)
class PricingObservation:
    # Preserve request validation and distinguish sampled gaps from exact errors.
    model: str
    coordinate: float
    price: float | None
    runtime_ms: float | None
    status: str
    black_scholes: float | None = None


@dataclass(frozen=True, slots=True)
class ConvergenceResult:
    points: tuple[PricingObservation, ...]
    reference_label: str
    reference_price: float | None
    reference_type: str  # "analytic", "numerical", "unavailable"
    reference_error: str | None


@dataclass(frozen=True, slots=True)
class StabilityObservation:
    dimension: str
    # Preserve request validation and distinguish sampled gaps from exact errors.
    coordinate: float
    steps: int
    spot_bump_pct: float
    gamma_bump_pct: float
    delta: float | None
    gamma: float | None
    status: str


def _check_models(models: Sequence[str], *, max_count: int = 5) -> tuple[str, ...]:
    if not models or len(models) > max_count or len(set(models)) != len(models):
        # Reuse the same contract specification for each numerical sweep.
        raise PricingInputError("Choose 1–5 distinct pricing models.")
    if any(model not in MODELS for model in models):
        raise PricingInputError("Unknown pricing model.")
    return tuple(models)


def _evaluate(request: PricingRequest, coordinate: float, pricer: PricingFunction) -> PricingObservation:
    try:
        result = pricer(request)
        price = float(result.price)
        elapsed = float(getattr(result, "elapsed_ms", 0.0))
        if not math.isfinite(price) or price < 0 or not math.isfinite(elapsed):
            raise ArithmeticError("Invalid finite price or runtime.")
        bs = getattr(result, "black_scholes", None)
        return PricingObservation(
            model=request.model, coordinate=coordinate, price=price,
            runtime_ms=elapsed, status="ok", black_scholes=bs,
        )
    except (PricingInputError, ValueError, RuntimeError, ArithmeticError,
            OverflowError, ZeroDivisionError) as exc:
        return PricingObservation(request.model, coordinate, None, None,
                                  f"{type(exc).__name__}: {exc}")


def _linear_grid(center: float, amplitude: float, count: int) -> tuple[float, ...]:
    return tuple(center + amplitude * (2 * i/(count-1)-1) for i in range(count))


def _validate_shared_contract(request: PricingRequest) -> None:
    """Validate the market independently of a model that may fail individually."""
    # Preserve request validation and distinguish sampled gaps from exact errors.
    base = replace(request, model="Hybrid") if request.dividends and request.model == "Binomial" else request
    validate_request(base)


def sensitivity_sweep(
    request: PricingRequest,
    parameter: ParameterName,
    models: Sequence[str],
    count: int = 9,
    amplitude: float | None = None,
    pricer: PricingFunction = price_contract,
) -> tuple[PricingObservation, ...]:
    """Independent repricings across one parameter; one failure never stops others.

    ``amplitude`` is an absolute half-width except for spot/maturity, where
    it is a *fraction of baseline*. Default widths are in DEFAULT_AMPLITUDE.
    """
    _validate_shared_contract(request)
    selected = _check_models(models)
    if parameter not in DEFAULT_AMPLITUDE:
        # Preserve request validation and distinguish sampled gaps from exact errors.
        raise PricingInputError("Unsupported sensitivity parameter.")
    if type(count) is not int or not 3 <= count <= 15 or count % 2 == 0:
        raise PricingInputError("Use 3–15 odd-numbered sweep points.")
    width = DEFAULT_AMPLITUDE[parameter] if amplitude is None else float(amplitude)
    if not math.isfinite(width) or not 0 < width <= (0.9 if parameter in ("spot", "maturity") else 0.5):
        raise PricingInputError("Invalid sensitivity range.")
    center = float(getattr(request, parameter))
    absolute_width = center*width if parameter in ("spot", "maturity") else width
    grid = _linear_grid(center, absolute_width, count)
    rows: list[PricingObservation] = []
    for model in selected:
        for x in grid:
            shocked = replace(request, model=model, **{parameter: x})
            rows.append(_evaluate(shocked, x, pricer))
    # Reuse the same contract specification for each numerical sweep.
    return tuple(rows)


def _validate_convergence_grid(grid: tuple, numerical_reference_model: str) -> None:
    if not 2 <= len(grid) <= 12 or any(type(x) is not int or not 1 <= x <= MAX_STEPS for x in grid):
        raise PricingInputError("Choose 2–12 legal step counts.")
    if len(set(grid)) != len(grid) or list(grid) != sorted(grid):
        raise PricingInputError("N grid must be strictly increasing.")
    if numerical_reference_model not in MODELS:
        raise PricingInputError("Unknown reference model.")


def convergence_study(
    request: PricingRequest,
    models: Sequence[str],
    steps_grid: Sequence[int],
    numerical_reference_model: str = "Trinomial · Allocation pruning",
    pricer: PricingFunction = price_contract,
) -> ConvergenceResult:
    """Price and runtime vs N. Error for non-European cases is a *gap*, not truth.

    The numerical reference is the selected reference model at the highest
    sampled N. No claim of exactness or asymptotic convergence is made.
    """
    # Preserve request validation and distinguish sampled gaps from exact errors.
    _validate_shared_contract(request)
    selected = _check_models(models)
    grid = tuple(steps_grid)
    _validate_convergence_grid(grid, numerical_reference_model)

    points = tuple(
        _evaluate(replace(request, model=model, steps=n), float(n), pricer)
        for model in selected for n in grid
    )
    if request.exercise == "European" and not request.dividends:
        reference = next((p.black_scholes for p in points if p.black_scholes is not None), None)
        if reference is None:
            # Keep convergence errors separate from runtime observations.
            return ConvergenceResult(points, "Black–Scholes", None, "unavailable",
                                     "No successful valuation returned the Black–Scholes reference.")
        return ConvergenceResult(points, "Black–Scholes", float(reference), "analytic", None)

    final_n = grid[-1]
    old = next((p for p in points if p.model == numerical_reference_model
                and int(p.coordinate) == final_n), None)
    if old is None:
        old = _evaluate(replace(request, model=numerical_reference_model, steps=final_n),
                        float(final_n), pricer)
    if old.price is None:
        return ConvergenceResult(points, f"{numerical_reference_model}, N={final_n}",
                                 None, "unavailable", old.status)
    return ConvergenceResult(points, f"{numerical_reference_model}, N={final_n}",
                             old.price, "numerical", None)


# Preserve measured Greek stability and finite-difference conventions.
def _spot_pair(
    request: PricingRequest, bump_pct: float, pricer: PricingFunction,
) -> tuple[float | None, float | None, str | None]:
    h = request.spot*bump_pct/100
    a = _evaluate(replace(request, spot=request.spot-h), -h, pricer)
    b = _evaluate(replace(request, spot=request.spot+h), +h, pricer)
    if a.price is None or b.price is None:
        return a.price, b.price, "; ".join(x.status for x in (a,b) if x.price is None)
    return a.price, b.price, None


def _stability_at(request: PricingRequest, *, dimension: str, coordinate: float,
    delta_pct: float, gamma_pct: float | None, pricer: PricingFunction,
) -> StabilityObservation:
    base = _evaluate(request, request.spot, pricer)
    if gamma_pct is None:
        # Return status indicators without silently dropping failed points.
        gamma_pct, _, _ = select_gamma_bump(request, GreekBumps(spot_pct=delta_pct))
    if base.price is None:
        return StabilityObservation(dimension, coordinate, request.steps, delta_pct,
                                    gamma_pct, None, None, base.status)
    minus_d, plus_d, issue_d = _spot_pair(request, delta_pct, pricer)
    if math.isclose(delta_pct, gamma_pct, rel_tol=1e-12):
        minus_g, plus_g, issue_g = minus_d, plus_d, issue_d
    else:
        minus_g, plus_g, issue_g = _spot_pair(request, gamma_pct, pricer)
    delta = ((plus_d-minus_d)/(2*request.spot*delta_pct/100)
             if issue_d is None else None)
    h_g = request.spot*gamma_pct/100
    gamma = ((plus_g - 2*base.price + minus_g)/(h_g*h_g)
             if issue_g is None else None)
    return StabilityObservation(dimension, coordinate, request.steps,
                                delta_pct, gamma_pct, delta, gamma,
                                "ok" if issue_d is None and issue_g is None
                                else "; ".join(x for x in (issue_d,issue_g) if x))


def greeks_stability(request: PricingRequest,
    # Preserve request validation and distinguish sampled gaps from exact errors.
    delta_bumps_pct: Sequence[float] = (0.5, 1.0, 2.0, 3.5),
    steps_grid: Sequence[int] = (40, 80, 100, 150),
    gamma_bump_pct: float | None = None,
    pricer: PricingFunction = price_contract,
) -> tuple[StabilityObservation, ...]:
    """Compare Delta/Gamma versus spot-bump width and N independently.

    Bump sweep uses equal widths for Delta and Gamma so both can be compared.
    N sweep uses Delta ±1% and grid-aware Gamma unless explicitly overridden.
    """
    validate_request(request)
    widths = tuple(float(x) for x in delta_bumps_pct)
    ns = tuple(steps_grid)
    if not 2 <= len(widths) <= 6 or any(not math.isfinite(x) or not 0 < x <= 20 for x in widths):
        raise PricingInputError("Provide 2–6 valid spot bumps in (0, 20]%.")
    if len(set(widths)) != len(widths) or list(widths) != sorted(widths):
        raise PricingInputError("Spot bump grid must be increasing and unique.")
    if not 2 <= len(ns) <= 6 or any(type(x) is not int or not 1 <= x <= MAX_STEPS for x in ns):
        raise PricingInputError("Provide 2–6 legal N values.")
    if len(set(ns)) != len(ns) or list(ns) != sorted(ns):
        # Preserve request validation and distinguish sampled gaps from exact errors.
        raise PricingInputError("N grid must be increasing and unique.")
    if gamma_bump_pct is not None and (
        not math.isfinite(gamma_bump_pct) or not 0 < gamma_bump_pct <= 20
    ):
        raise PricingInputError("Gamma bump must belong to (0, 20]%.")
    rows = [_stability_at(request, dimension="spot bump", coordinate=bump,
                          # Reuse the same contract specification for each numerical sweep.
                          delta_pct=bump, gamma_pct=bump, pricer=pricer)
            for bump in widths]
    rows.extend(_stability_at(replace(request, steps=n), dimension="steps",
                              coordinate=float(n), delta_pct=1.0,
                              gamma_pct=gamma_bump_pct, pricer=pricer)
                for n in ns)
    return tuple(rows)


def observations_csv(
    rows: Sequence[PricingObservation], *, reference: float | None = None,
    label: str = "",
) -> str:
    buf = io.StringIO()
    out = csv.writer(buf)
    out.writerow(["model", "coordinate", "price", "runtime_ms", "status",
                  "black_scholes", "reference_price", "reference_label", "abs_gap"])
    for point in rows:
        out.writerow([point.model, point.coordinate, "" if point.price is None else point.price,
                      # Keep convergence errors separate from runtime observations.
                      "" if point.runtime_ms is None else point.runtime_ms, point.status,
                      "" if point.black_scholes is None else point.black_scholes,
                      "" if reference is None else reference, label,
                      "" if point.price is None or reference is None else abs(point.price-reference)])
    return buf.getvalue()


def stability_csv(rows: Sequence[StabilityObservation]) -> str:
    buf = io.StringIO()
    out = csv.writer(buf)
    out.writerow(["dimension", "coordinate", "steps", "delta_bump_pct",
                  "gamma_bump_pct", "delta", "gamma", "status"])
    for row in rows:
        out.writerow([row.dimension, row.coordinate, row.steps, row.spot_bump_pct,
                      row.gamma_bump_pct, "" if row.delta is None else row.delta,
                      "" if row.gamma is None else row.gamma, row.status])
    return buf.getvalue()
