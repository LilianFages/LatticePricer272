"""Memory-bounded binomial pricing for deep vanilla lattices.

Matches BinomialTree's grid and transition probabilities without constructing
Node objects. The rolling value array uses O(N) memory and O(N^2) operations.
Discrete dividends are deliberately unsupported.
"""

from __future__ import annotations

import math

import numpy as np


def price_large_binomial(
    *,
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    steps: int,
    is_call: bool,
    is_american: bool,
) -> float:
    """Price a vanilla binomial option with rolling NumPy arrays."""
    if type(steps) is not int or steps < 1:
        raise ValueError("Steps must be a strictly positive integer.")
    inputs = (spot, strike, rate, volatility, maturity)
    if not all(math.isfinite(x) for x in inputs):
        raise ValueError("Pricing inputs must be finite.")
    if spot <= 0.0 or strike < 0.0 or volatility < 0.0 or maturity <= 0.0:
        raise ValueError("Invalid binomial pricing inputs.")

    dt = maturity / steps
    variance_factor = math.exp(volatility * volatility * dt)
    middle = (1.0 + variance_factor) / 2.0
    alpha = middle + math.sqrt((middle - 1.0) * (middle + 1.0))
    log_alpha = math.log(alpha)
    up_probability = 1.0 / (alpha + 1.0)
    down_probability = 1.0 - up_probability
    discount_factor = math.exp(-rate * dt)
    positions = np.arange(steps + 1, dtype=np.float64)

    def payoff(stock: np.ndarray) -> np.ndarray:
        return (np.maximum(stock - strike, 0.0)
                if is_call else np.maximum(strike - stock, 0.0))

    try:
        with np.errstate(over="raise", invalid="raise"):
            terminal_exponents = (
                rate * maturity + (steps - 2.0 * positions) * log_alpha
            )
            values = payoff(spot * np.exp(terminal_exponents))
            for step in range(steps - 1, -1, -1):
                values = discount_factor * (
                    up_probability * values[:-1]
                    + down_probability * values[1:]
                )
                if is_american:
                    exponents = (
                        rate * dt * step
                        + (step - 2.0 * positions[:step + 1]) * log_alpha
                    )
                    values = np.maximum(
                        values, payoff(spot * np.exp(exponents)),
                    )
    except (OverflowError, FloatingPointError) as exc:
        raise ValueError(
            "The requested depth and market inputs exceed numerical limits."
        ) from exc

    result = float(values[0])
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("The binomial premium is invalid.")
    return result
