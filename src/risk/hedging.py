"""Pure stock delta-hedge arithmetic; no pricing or UI dependency.

Signed shares: positive means BUY; negative means SELL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class StockHedge:
    """A stock trade that offsets the signed option delta exposure."""

    option_delta_shares: float
    existing_shares: float
    target_shares: float
    exact_trade: float
    rounded_trade: int
    residual_delta_shares: float
    instruction: Literal["BUY", "SELL", "HOLD"]


def calculate_stock_hedge(
    delta: float,
    side: Literal["Long", "Short"],
    contracts: int,
    multiplier: int = 100,
    existing_shares: float = 0.0,
) -> StockHedge:
    """Return signed stock trade to offset option delta and existing shares.

    Whole-share rounding uses half-away-from-zero, not banker's rounding.
    No trade is placed by this calculation.
    """
    if side not in ("Long", "Short"):
        raise ValueError("Position must be Long or Short.")
    if type(contracts) is not int or not 1 <= contracts <= 1_000_000:
        raise ValueError("Number of contracts must be a positive integer <= 1,000,000.")
    if type(multiplier) is not int or not 1 <= multiplier <= 1_000_000:
        raise ValueError("Contract multiplier must be a positive integer <= 1,000,000.")
    if not math.isfinite(delta) or not math.isfinite(existing_shares):
        raise ValueError("Delta and existing shares must be finite.")
    direction = 1 if side == "Long" else -1
    option_exposure = direction * delta * contracts * multiplier
    target_shares = -option_exposure
    exact_trade = target_shares - existing_shares
    rounded = (math.floor(exact_trade + 0.5) if exact_trade >= 0
               else math.ceil(exact_trade - 0.5))
    action: Literal["BUY", "SELL", "HOLD"] = (
        "BUY" if rounded > 0 else "SELL" if rounded < 0 else "HOLD"
    )
    residual = option_exposure + existing_shares + rounded
    return StockHedge(
        option_delta_shares=option_exposure,
        existing_shares=existing_shares,
        target_shares=target_shares,
        exact_trade=exact_trade,
        rounded_trade=rounded,
        residual_delta_shares=residual,
        instruction=action,
    )
