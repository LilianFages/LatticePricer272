"""Frontend compatibility facade for seven reusable risk Greeks and hedging."""

from __future__ import annotations

from frontend.greeks_service import GreekBumps, calculate_greeks
from frontend.pricing_service import (
    PricingFunction,
    PricingRequest,
    price_contract,
    validate_request,
)
from src.risk.desk import (
    EXPECTED_PRICING_ERRORS,
    GREEK_KEYS,
    GREEK_NAMES,
    GREEK_UNITS,
    DeskRiskSnapshot,
    GreekCurvePoint,
    GreekName,
    _bs_second_order,
    compute_desk_risk as _compute_desk_risk,
    compute_greek_curve as _compute_greek_curve,
    greek_table,
)
from src.risk.hedging import StockHedge, calculate_stock_hedge


def compute_desk_risk(
    request: PricingRequest,
    bumps: GreekBumps | None = None,
    pricer: PricingFunction = price_contract,
) -> DeskRiskSnapshot:
    """Inject existing validators and pricing engine into the shared core."""
    # Keep the core engine injectable so test doubles and notebook adapters work.
    return _compute_desk_risk(
        request,
        bumps,
        pricer=pricer,
        validate_request=validate_request,
        calculate_greeks=calculate_greeks,
    )


def compute_greek_curve(
    request: PricingRequest,
    name: GreekName,
    *,
    count: int = 5,
    span_pct: float = 10.0,
    bumps: GreekBumps | None = None,
    pricer: PricingFunction = price_contract,
) -> tuple[GreekCurvePoint, ...]:
    """Retain lazy-curve semantics and monkeypatchable risk callbacks."""
    # Greeks are evaluated lazily through the existing shared implementation.
    return _compute_greek_curve(
        request,
        name,
        count=count,
        span_pct=span_pct,
        bumps=bumps,
        pricer=pricer,
        validate_request=validate_request,
        calculate_greeks=calculate_greeks,
        compute_desk_risk=compute_desk_risk,
    )
