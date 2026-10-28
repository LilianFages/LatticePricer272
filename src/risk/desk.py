"""Seven desk Greeks and curves, calculated with injected pricing callbacks.

All pricing calls delegate to the existing lattice engines. In particular,
no new pricing equation or dividend convention is introduced here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from collections.abc import Callable

from src.pricing_errors import PricingInputError
from src.risk.contracts import (
    PricingFunction,
    RiskRequest as PricingRequest,
    RiskValidator,
)
from src.risk.greeks import GreekBumps, GreeksResult
from src.risk.numerics import (
    black_scholes_vanna_vomma,
    vanna_from_samples,
    vomma_from_samples,
)

GreekName = Literal["Delta", "Gamma", "Vega", "Vanna", "Vomma", "Theta", "Rho"]
GREEK_NAMES: tuple[GreekName, ...] = (
    "Delta", "Gamma", "Vega", "Vanna", "Vomma", "Theta", "Rho",
)
GREEK_UNITS = {
    "Delta": "EUR premium / EUR spot",
    "Gamma": "EUR premium / EUR spot²",
    "Vega": "EUR premium / +1 volatility pp",
    "Vanna": "Delta / +1 volatility pp",
    "Vomma": "EUR premium / (1 volatility pp)²",
    "Theta": "EUR premium / +1 calendar day",
    "Rho": "EUR premium / +1 bp of rate",
}
GREEK_KEYS = {
    "Delta": "Delta",
    "Gamma": "Gamma",
    "Vega": "Vega / 1pp",
    "Theta": "Theta / 1d",
    "Rho": "Rho / 1bp",
}
EXPECTED_PRICING_ERRORS = (
    PricingInputError, ValueError, RuntimeError, OverflowError, ZeroDivisionError,
    ArithmeticError,
)


@dataclass(frozen=True, slots=True)
class DeskRiskSnapshot:
    """Five existing Greeks plus independently bumped cross/vol convexities."""

    base: GreeksResult
    vanna_1pp: float | None
    vomma_1pp2: float | None
    bs_vanna_1pp: float | None
    bs_vomma_1pp2: float | None
    extra_evaluations: int
    extra_issues: tuple[str, ...]

    def value(self, name: GreekName) -> float | None:
        if name not in GREEK_NAMES:
            raise ValueError(f"Unsupported Greek: {name}.")
        if name == "Vanna":
            return self.vanna_1pp
        if name == "Vomma":
            return self.vomma_1pp2
        return self.base.measured()[GREEK_KEYS[name]]

    def benchmark(self, name: GreekName) -> float | None:
        if name not in GREEK_NAMES:
            raise ValueError(f"Unsupported Greek: {name}.")
        if name == "Vanna":
            return self.bs_vanna_1pp
        if name == "Vomma":
            return self.bs_vomma_1pp2
        return (self.base.analytic or {}).get(GREEK_KEYS[name])


@dataclass(frozen=True, slots=True)
class GreekCurvePoint:
    spot: float
    value: float | None
    error: str | None = None


def _bs_second_order(request: PricingRequest) -> tuple[float | None, float | None]:
    """Analytic BS Vanna/Vomma only when standard BS assumptions apply."""
    if (request.exercise != "European" or request.dividends
            or request.volatility <= 0.0 or request.strike <= 0.0):
        return None, None
    return black_scholes_vanna_vomma(
        spot=request.spot, strike=request.strike, rate=request.rate,
        volatility=request.volatility, maturity=request.maturity,
    )


def compute_desk_risk(
    request: PricingRequest,
    bumps: GreekBumps | None = None,
    *,
    pricer: PricingFunction,
    validate_request: RiskValidator,
    calculate_greeks: Callable[..., GreeksResult],
) -> DeskRiskSnapshot:
    """Compute seven desk Greeks; extras never invalidate valid base Greeks.

    Vanna is a mixed spot/volatility finite difference, normalized for +1
    volatility percentage point. Vomma is volatility curvature normalized for
    a one-percentage-point squared shock. Zero/low vol uses a forward stencil.
    """
    validate_request(request)
    selected = bumps or GreekBumps()
    selected.validate()
    base = calculate_greeks(request, selected, pricer=pricer)
    issues: list[str] = []
    extra_calls = 0
    h_s = request.spot * selected.spot_pct / 100.0
    h_vol = selected.vol_pp / 100.0
    prices = base.scenario_prices

    def bump_price(spot: float, volatility: float) -> float | None:
        nonlocal extra_calls
        extra_calls += 1
        try:
            value = float(pricer(replace(request, spot=spot, volatility=volatility)).price)
            if not math.isfinite(value):
                raise ArithmeticError("Non-finite shocked premium")
            return value
        except EXPECTED_PRICING_ERRORS as exc:
            issues.append(f"Vanna shock unavailable: {type(exc).__name__}: {exc}")
            return None

    plus_hi = bump_price(request.spot + h_s, request.volatility + h_vol)
    minus_hi = bump_price(request.spot - h_s, request.volatility + h_vol)
    if request.volatility >= h_vol:
        plus_lo = bump_price(request.spot + h_s, request.volatility - h_vol)
        minus_lo = bump_price(request.spot - h_s, request.volatility - h_vol)
        vanna = vanna_from_samples(
            plus_high=plus_hi, minus_high=minus_hi,
            plus_low=plus_lo, minus_low=minus_lo,
            spot_bump=h_s, vol_bump=h_vol, centered=True,
        )
    else:
        # Forward volatility difference; reuse already computed spot shocks.
        spot_plus = prices.get("spot+")
        spot_minus = prices.get("spot-")
        vanna = vanna_from_samples(
            plus_high=plus_hi, minus_high=minus_hi,
            plus_low=spot_plus, minus_low=spot_minus,
            spot_bump=h_s, vol_bump=h_vol, centered=False,
        )
        issues.append("Vanna uses a forward volatility difference near zero volatility.")
    vol_up = prices.get("volatility+")
    if request.volatility >= h_vol:
        vol_down = prices.get("volatility-")
        vomma = vomma_from_samples(
            base=base.price, high=vol_up, low=vol_down,
            vol_bump=h_vol, centered=True,
        )
    else:
        vol_up2 = prices.get("volatility++")
        vomma = vomma_from_samples(
            base=base.price, high=vol_up, low=vol_up2,
            vol_bump=h_vol, centered=False,
        )
        issues.append("Vomma uses a forward second difference near zero volatility.")
    analytic_vanna, analytic_vomma = _bs_second_order(request)
    return DeskRiskSnapshot(
        base=base, vanna_1pp=vanna, vomma_1pp2=vomma,
        bs_vanna_1pp=analytic_vanna, bs_vomma_1pp2=analytic_vomma,
        extra_evaluations=extra_calls, extra_issues=tuple(issues),
    )


def greek_table(snapshot: DeskRiskSnapshot) -> tuple[dict[str, object], ...]:
    """Render-neutral table with seven sensitivities and optional BS references."""
    rows: list[dict[str, object]] = []
    for name in GREEK_NAMES:
        model_value = snapshot.value(name)
        bs_value = snapshot.benchmark(name)
        rows.append({
            "Greek": name,
            "Model": model_value,
            "Black-Scholes": bs_value,
            "Model - BS": (model_value - bs_value
                           if model_value is not None and bs_value is not None else None),
            "Unit": GREEK_UNITS[name],
        })
    return tuple(rows)


def compute_greek_curve(
    request: PricingRequest,
    name: GreekName,
    *,
    count: int = 5,
    span_pct: float = 10.0,
    bumps: GreekBumps | None = None,
    pricer: PricingFunction,
    validate_request: RiskValidator,
    calculate_greeks: Callable[..., GreeksResult],
    compute_desk_risk: Callable[..., DeskRiskSnapshot],
) -> tuple[GreekCurvePoint, ...]:
    """Explicitly requested spot curve. Individual failures are not zero-filled."""
    validate_request(request)
    if name not in GREEK_NAMES:
        raise ValueError("Unsupported Greek curve.")
    if type(count) is not int or not 3 <= count <= 7 or count % 2 == 0:
        raise ValueError("Curve requires 3, 5 or 7 spot points.")
    if not math.isfinite(span_pct) or not 0.0 < span_pct <= 30.0:
        raise ValueError("Curve half-width must lie in (0, 30]%.")
    selected = bumps or GreekBumps()
    selected.validate()
    points: list[GreekCurvePoint] = []
    for i in range(count):
        spot = request.spot * (1.0 + span_pct / 100.0 * (2 * i / (count - 1) - 1))
        shocked = replace(request, spot=spot)
        try:
            if name in ("Vanna", "Vomma"):
                value = compute_desk_risk(shocked, selected, pricer=pricer).value(name)
            else:
                greeks = calculate_greeks(shocked, selected, pricer=pricer)
                value = greeks.measured()[GREEK_KEYS[name]]
            if value is None or not math.isfinite(value):
                points.append(GreekCurvePoint(spot, None, "Greek unavailable for this spot."))
            else:
                points.append(GreekCurvePoint(spot, float(value)))
        except EXPECTED_PRICING_ERRORS as exc:
            points.append(GreekCurvePoint(spot, None, f"{type(exc).__name__}: {exc}"))
    return tuple(points)
