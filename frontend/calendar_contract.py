"""Calendar inputs for equity-option contracts.

The lattice engines use year fractions, not calendar dates. This module
converts dated inputs without changing their pricing conventions. The sole
supported day-count convention is ACT/365 Fixed, consistent with the existing
one-day Greek roll-down. No financial engine is imported here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime


DAYS_PER_YEAR = 365.0
DAY_COUNT = "ACT/365F"


class CalendarInputError(ValueError):
    """Invalid or inconsistent calendar data supplied by the user."""


@dataclass(frozen=True, slots=True)
class DatedCashDividend:
    """Cash dividend paid at the specified ex-dividend date."""

    ex_date: date
    amount: float


@dataclass(frozen=True, slots=True)
class RelativeCashDividend:
    """Dividend converted into the numerical engine's time convention."""

    time: float
    amount: float


@dataclass(frozen=True, slots=True)
class ResolvedCalendar:
    """Exact numerical schedule derived from the original dated inputs."""

    maturity: float
    dividends: tuple[RelativeCashDividend, ...]
    day_count: str = DAY_COUNT


@dataclass(frozen=True, slots=True)
class ContractCalendar:
    """User-facing dates retained independently of the numerical pricing ticket.

    Dates use ACT/365 Fixed. Dividend events are mapped to time since the
    valuation date; any event on the valuation date is excluded.
    """

    valuation_date: date
    expiry_date: date
    dividends: tuple[DatedCashDividend, ...] = ()

    def _resolved_dividends(self) -> tuple[RelativeCashDividend, ...]:
        """Validate and convert all dated cashflows in ACT/365F time."""
        dividends: list[RelativeCashDividend] = []
        seen_dates: set[date] = set()

        for dividend in self.dividends:
            if not isinstance(dividend, DatedCashDividend):
                raise CalendarInputError(
                    "Each dividend must have an ex-date and a cash amount."
                )
            _require_date(dividend.ex_date, "Dividend ex-date")
            if not (self.valuation_date < dividend.ex_date <= self.expiry_date):
                raise CalendarInputError(
                    "Dividend ex-date must fall after valuation and on or "
                    "before expiry."
                )
            if dividend.ex_date in seen_dates:
                raise CalendarInputError(
                    "Two cash dividends cannot share the same ex-date."
                )
            seen_dates.add(dividend.ex_date)
            amount = _positive_cash_amount(dividend.amount)
            days = (dividend.ex_date - self.valuation_date).days
            dividends.append(RelativeCashDividend(days / DAYS_PER_YEAR, amount))

        dividends.sort(key=lambda item: item.time)
        return tuple(dividends)

    def resolve(self) -> ResolvedCalendar:
        """Validate the calendar and return engine-compatible year fractions.

        This operation neither reprices a contract nor modifies any lattice.
        A subsequent PricingRequest validation must still check whether two
        dividends fall into the same time step for the selected N.
        """
        _require_date(self.valuation_date, "Valuation date")
        _require_date(self.expiry_date, "Expiry date")
        if self.expiry_date <= self.valuation_date:
            raise CalendarInputError(
                "Expiry date must be later than valuation date."
            )

        maturity_days = (self.expiry_date - self.valuation_date).days
        dividends = self._resolved_dividends()
        return ResolvedCalendar(
            maturity=maturity_days / DAYS_PER_YEAR,
            dividends=dividends,
        )


def _positive_cash_amount(raw_amount: object) -> float:
    """Reject booleans and nonpositive or nonfinite dividend cashflows."""
    try:
        amount = float(raw_amount)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CalendarInputError(
            "Cash dividend amount must be a finite positive number."
        ) from exc
    if isinstance(raw_amount, bool) or not math.isfinite(amount) or amount <= 0:
        raise CalendarInputError(
            "Cash dividend amount must be a finite positive number."
        )
    return amount


def _require_date(value: object, name: str) -> None:
    """Reject datetime values: a date-based ticket must not hide a time zone."""
    if not isinstance(value, date) or isinstance(value, datetime):
        raise CalendarInputError(f"{name} must be a calendar date.")
