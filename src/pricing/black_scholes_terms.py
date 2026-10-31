"""Shared Black--Scholes standardized terms for prices and risk.

The function is purely mathematical and does not depend on tree, option,
Streamlit, numerical engine, or application-level pricing contracts.
Caller guards for zero volatility, zero strike, or zero maturity still apply.
"""
from __future__ import annotations

import math


def standardized_terms(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> tuple[float, float]:
    """Return (d1, d2) for a regular Black--Scholes contract.

    Defined only for strictly positive spot, strike, volatility and maturity.
    Special cases are handled by the calling valuation/risk operation.
    """
    sigma_sqrt_t = volatility * math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate + 0.5 * volatility ** 2) * maturity
    ) / sigma_sqrt_t
    return d1, d1 - sigma_sqrt_t
