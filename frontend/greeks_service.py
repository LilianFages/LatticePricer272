"""Compatibility adapter exposing the reusable risk kernel to workspaces.

Keep existing import paths stable. All Greek formulae live in src/risk/.
"""

from __future__ import annotations

from frontend.pricing_service import (
    PricingFunction,
    PricingRequest,
    price_contract,
    validate_request,
)
from src.risk.greeks import (
    BP,
    DAY,
    VOL_POINT,
    GreekBumps,
    GreeksResult,
    analytic_bs_greeks,
    calculate_greeks as _calculate_greeks,
    greeks_csv,
    roll_calendar_day,
    select_gamma_bump,
)


def calculate_greeks(
    request: PricingRequest,
    bumps: GreekBumps | None = None,
    pricer: PricingFunction = price_contract,
) -> GreeksResult:
    """Preserve the historical API, including explicit pricer injection."""
    return _calculate_greeks(
        request,
        bumps,
        pricer=pricer,
        validate_request=validate_request,
    )
