"""Pure numerical second-order Greeks, independent of all pricing engines.

The caller provides option premiums at explicitly shocked contract inputs.
Return ``None`` when a required premium is unavailable, never an invented zero.
Volatility derivatives are normalized for percentage points (1 pp = 0.01).
"""

from __future__ import annotations

import math

from src.black_scholes_terms import standardized_terms


def _check_bump(name: str, size: float) -> None:
    if not math.isfinite(size) or size <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")


def vanna_from_samples(
    *,
    plus_high: float | None,
    minus_high: float | None,
    plus_low: float | None,
    minus_low: float | None,
    spot_bump: float,
    vol_bump: float,
    centered: bool
) -> float | None:
    """Mixed d²V/dS dσ from crossed spot/volatility bumps, per vol pp.

    High/low refer to volatility; plus/minus refer to spot.
    For ``centered=False``, low samples are the base-volatility spot bumps.
    """
    _check_bump("Spot bump", spot_bump)
    _check_bump("Volatility bump", vol_bump)
    samples = (plus_high, minus_high, plus_low, minus_low)
    if any(value is None for value in samples):
        return None
    if any(not math.isfinite(value) for value in samples):
        return None
    numerator = (plus_high - minus_high) - (plus_low - minus_low)
    denominator = (4.0 if centered else 2.0) * spot_bump * vol_bump
    return 0.01 * numerator / denominator


def vomma_from_samples(
    *,
    base: float,
    high: float | None,
    low: float | None,
    vol_bump: float,
    centered: bool
) -> float | None:
    """Second volatility derivative per (1 volatility percentage point)².

    With ``centered=False`` low is the price at σ + 2h (not σ - h).
    """
    _check_bump("Volatility bump", vol_bump)
    if high is None or low is None:
        return None
    if not all(math.isfinite(v) for v in (base, high, low)):
        return None
    numerator = (high - 2.0 * base + low) if centered else (
        low - 2.0 * high + base
    )
    return 0.0001 * numerator / (vol_bump * vol_bump)


def black_scholes_vanna_vomma(
    *, spot: float, strike: float, rate: float,
    volatility: float, maturity: float
) -> tuple[float, float]:
    """Analytical European no-dividend BS second-order Greeks, pp-normalized.

    Eligibility (European, no cash dividends) is validated by the caller.
    """
    if not all(math.isfinite(x) for x in (spot, strike, rate, volatility, maturity)):
        raise ValueError("Black-Scholes inputs must be finite.")
    if min(spot, strike, volatility, maturity) <= 0.0:
        raise ValueError("Spot, strike, volatility and maturity must be positive.")
    root_t = math.sqrt(maturity)
    d1, d2 = standardized_terms(spot, strike, rate, volatility, maturity)
    density = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    vanna = -density * d2 / volatility * 0.01
    vomma = spot * density * root_t * d1 * d2 / volatility * 0.0001
    return vanna, vomma
