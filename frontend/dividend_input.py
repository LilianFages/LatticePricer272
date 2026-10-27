"""Pure adapter for cash-dividend DataFrame inputs used across UI workspaces.

The financial validation remains in pricing_service. Both historical UI error
wordings are preserved to avoid surprising users and tests.
"""
from __future__ import annotations

import math
from typing import Literal

import pandas as pd

from frontend.pricing_service import DividendInput, PricingInputError

InputContext = Literal["terminal", "analysis"]


def _validated_dividend_amount(
    raw_amount: object, raw_time: object, row_number: int, context: InputContext,
) -> float | None:
    """Validate a cash amount; fully blank rows return None."""
    if pd.isna(raw_amount):
        if not pd.isna(raw_time):
            instruction = (
                "enter an amount or delete the row." if context == "terminal"
                else "supply an amount or delete the row."
            )
            raise PricingInputError(f"Dividend row {row_number}: {instruction}")
        return None
    try:
        amount = float(raw_amount)
    except (ValueError, TypeError):
        raise PricingInputError("Dividend amounts must be numeric.") from None
    if not math.isfinite(amount):
        raise PricingInputError("Dividend amounts must be finite.")
    return amount


def _validated_dividend_time(
    raw_time: object, row_number: int, context: InputContext,
) -> float:
    """Validate payment-time entry, retaining workspace error wording."""
    if pd.isna(raw_time):
        instruction = (
            "enter the payment time in years." if context == "terminal"
            else "supply a payment time."
        )
        raise PricingInputError(f"Dividend row {row_number}: {instruction}")
    try:
        dividend_time = float(raw_time)
    except (ValueError, TypeError):
        raise PricingInputError("Dividend times must be numeric.") from None
    return dividend_time


def parse_cash_dividends(
    frame: pd.DataFrame, *, context: InputContext = "analysis",
) -> tuple[DividendInput, ...]:
    """Parse chronological cashflows, ignoring zero and fully blank rows.

    No date rounding, cashflow omission, pricing or contract mutation occurs.
    The pricing adapter independently validates the resulting schedule.
    """
    if context not in ("terminal", "analysis"):
        raise ValueError("Unsupported dividend input context.")

    dividends: list[DividendInput] = []
    for row_number, record in enumerate(frame.to_dict("records"), 1):
        raw_amount = record.get("amount")
        raw_time = record.get("time")
        amount = _validated_dividend_amount(
            raw_amount, raw_time, row_number, context,
        )
        if amount is None:
            continue
        if amount == 0.0:
            continue
        dividend_time = _validated_dividend_time(
            raw_time, row_number, context,
        )
        dividends.append(DividendInput(time=dividend_time, amount=amount))
    return tuple(sorted(dividends, key=lambda dividend: dividend.time))


def parse_dividends(frame: pd.DataFrame) -> tuple[DividendInput, ...]:
    """Backward-compatible public name for Model Comparison and analytics."""
    return parse_cash_dividends(frame, context="analysis")
