"""Pure date-to-pricing-ticket adapter for the Pricing Terminal.

Keep calendar dates as UI metadata. Financial models still consume ACT/365F
fractions via the existing immutable PricingRequest API.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, datetime

import pandas as pd

from frontend.calendar_contract import (
    CalendarInputError, ContractCalendar, DatedCashDividend,
)
from frontend.pricing_service import (
    DividendInput, PricingInputError, PricingRequest, validate_request,
)


def _as_date(value: object, row_number: int) -> date:
    """Normalize Streamlit DateColumn values; reject arbitrary date strings."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise CalendarInputError(
        f"Dividend row {row_number}: enter a valid ex-dividend date."
    )


def _nonnegative_amount(raw_amount: object, row_number: int) -> float:
    """Validate a dated cashflow without accepting boolean amounts."""
    if isinstance(raw_amount, bool):
        raise CalendarInputError(
            f"Dividend row {row_number}: amount must be numeric."
        )
    try:
        amount = float(raw_amount)
    except (ValueError, TypeError, OverflowError) as exc:
        raise CalendarInputError(
            f"Dividend row {row_number}: amount must be numeric."
        ) from exc
    if not math.isfinite(amount) or amount < 0:
        raise CalendarInputError(
            f"Dividend row {row_number}: amount must be finite and nonnegative."
        )
    return amount


def parse_dated_dividends(frame: pd.DataFrame) -> tuple[DatedCashDividend, ...]:
    """Ignore optional zero rows; never discard a malformed positive cashflow."""
    events: list[DatedCashDividend] = []
    for row_number, row in enumerate(frame.to_dict("records"), start=1):
        amount_raw = row.get("amount")
        date_raw = row.get("ex_date")
        if pd.isna(amount_raw):
            if not pd.isna(date_raw):
                raise CalendarInputError(
                    f"Dividend row {row_number}: enter an amount or delete the row."
                )
            continue
        amount = _nonnegative_amount(amount_raw, row_number)
        if amount == 0.0:
            continue
        if pd.isna(date_raw):
            raise CalendarInputError(
                f"Dividend row {row_number}: positive amount requires an ex-date."
            )
        events.append(DatedCashDividend(_as_date(date_raw, row_number), amount))
    return tuple(sorted(events, key=lambda item: item.ex_date))


def dated_request(
    base: PricingRequest,
    valuation_date: date,
    expiry_date: date,
    dated_dividends: tuple[DatedCashDividend, ...],
) -> tuple[PricingRequest, ContractCalendar]:
    """Derive the exact model ticket, leaving the original immutable ticket intact."""
    calendar = ContractCalendar(valuation_date, expiry_date, dated_dividends)
    resolved = calendar.resolve()
    request = replace(
        base,
        maturity=resolved.maturity,
        dividends=tuple(DividendInput(item.time, item.amount)
                        for item in resolved.dividends),
    )
    try:
        validate_request(request)
    except PricingInputError as exc:
        raise CalendarInputError(str(exc)) from exc
    return request, calendar


def calendar_matches_request(calendar: ContractCalendar, request: PricingRequest) -> bool:
    """Check calendar metadata matches the saved, successful pricing inputs."""
    if not isinstance(calendar, ContractCalendar) or not isinstance(request, PricingRequest):
        return False
    resolved = calendar.resolve()
    if not math.isclose(resolved.maturity, request.maturity, rel_tol=0.0, abs_tol=1e-12):
        return False
    expected = tuple((d.time, d.amount) for d in resolved.dividends)
    actual = tuple((d.time, d.amount) for d in request.dividends)
    return expected == actual
