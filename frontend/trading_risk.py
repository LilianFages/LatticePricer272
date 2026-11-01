"""Frontend compatibility facade for seven reusable risk Greeks and hedging."""

from __future__ import annotations

from frontend.greeks_service import GreekBumps, calculate_greeks
from frontend.pricing_service import (
    PricingFunction,
    PricingRequest,
    price_contract,
    validate_terminal_request,
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
    # Keep the injectable risk engine and its pricing callback unchanged.
    compute_desk_risk as _compute_desk_risk,
    compute_greek_curve as _compute_greek_curve,
    greek_table,
)
from src.risk.hedging import StockHedge, calculate_stock_hedge


# Shared annotations for readable, short function signatures.
RiskPricer = PricingFunction

# Local type aliases keep adapter signatures readable and fully annotated.
R = PricingRequest
N = GreekName
B = GreekBumps | None
PF = RiskPricer
pc = price_contract
Curve = tuple[GreekCurvePoint, ...]

def compute_desk_risk(request: R, bumps: B = None, pricer: PF = pc) -> DeskRiskSnapshot:
    """Inject existing validators and pricing engine into the shared core."""
    # Keep the core engine injectable so test doubles and notebook adapters work.
    return _compute_desk_risk(
        request,
        bumps,
        pricer=pricer,
        validate_request=validate_terminal_request,
        calculate_greeks=calculate_greeks,
    )


def compute_greek_curve(request: R, name: N, count: int = 5, span_pct: float = 10.0, bumps: B = None, pricer: PF = pc) -> Curve:
    """Retain lazy-curve semantics and monkeypatchable risk callbacks."""
    # Greeks are evaluated lazily through the existing shared implementation.
    return _compute_greek_curve(
        request,
        name,
        count=count,
        span_pct=span_pct,
        bumps=bumps,
        pricer=pricer,
        validate_request=validate_terminal_request,
        calculate_greeks=calculate_greeks,
        # Use the existing injectable desk-risk function for each curve point.
        compute_desk_risk=compute_desk_risk,
    )
