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
        if pd.isna(raw_amount):
            if not pd.isna(raw_time):
                instruction = (
                    "enter an amount or delete the row." if context == "terminal"
                    else "supply an amount or delete the row."
                )
                raise PricingInputError(f"Dividend row {row_number}: {instruction}")
            continue
        try:
            amount = float(raw_amount)
        except (ValueError, TypeError):
            raise PricingInputError("Dividend amounts must be numeric.") from None
        if not math.isfinite(amount):
            raise PricingInputError("Dividend amounts must be finite.")
        if amount == 0.0:
            continue
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
        dividends.append(DividendInput(time=dividend_time, amount=amount))
    return tuple(sorted(dividends, key=lambda dividend: dividend.time))


def parse_dividends(frame: pd.DataFrame) -> tuple[DividendInput, ...]:
    """Backward-compatible public name for Model Comparison and analytics."""
    return parse_cash_dividends(frame, context="analysis")
