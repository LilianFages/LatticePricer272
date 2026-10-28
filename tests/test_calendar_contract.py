"""Tests for ACT/365F calendar-to-engine conversion and date validation."""

from dataclasses import FrozenInstanceError
from datetime import date, datetime

import pytest

from frontend.calendar_contract import (
    CalendarInputError,
    ContractCalendar,
    DatedCashDividend,
)


START = date(2026, 9, 1)
EXPIRY = date(2027, 6, 28)


# Calendar maturity follows ACT/365F rather than an implicit rounded year.
def test_expiry_is_calendar_driven() -> None:
    resolved = ContractCalendar(START, EXPIRY).resolve()
    assert resolved.maturity == pytest.approx(300 / 365)
    assert resolved.day_count == "ACT/365F"
    assert resolved.dividends == ()


def test_single_cash_dividend_uses_ex_date() -> None:
    calendar = ContractCalendar(
        START, EXPIRY, (DatedCashDividend(date(2027, 4, 21), 4.0),)
    )
    resolved = calendar.resolve()
    assert resolved.dividends[0].time == pytest.approx(232 / 365)
    assert resolved.dividends[0].amount == 4.0


# Date sorting must not mutate the caller-provided cash schedule.
def test_multiple_dividends_are_sorted_without_changing_input() -> None:
    late = DatedCashDividend(date(2027, 4, 21), 4.0)
    early = DatedCashDividend(date(2026, 12, 1), 1.0)
    calendar = ContractCalendar(START, EXPIRY, (late, early))
    resolved = calendar.resolve()
    assert [item.amount for item in resolved.dividends] == [1.0, 4.0]
    assert calendar.dividends == (late, early)


def test_leap_year_uses_fixed_365_denominator() -> None:
    calendar = ContractCalendar(date(2024, 2, 28), date(2024, 3, 1))
    assert calendar.resolve().maturity == pytest.approx(2 / 365)


# A single calendar day is a valid strictly positive maturity.
def test_one_day_maturity() -> None:
    calendar = ContractCalendar(date(2026, 9, 1), date(2026, 9, 2))
    assert calendar.resolve().maturity == pytest.approx(1 / 365)


def test_dividend_at_expiry_is_preserved() -> None:
    calendar = ContractCalendar(
        START, EXPIRY, (DatedCashDividend(EXPIRY, 0.25),)
    )
    resolved = calendar.resolve()
    assert resolved.dividends[0].time == resolved.maturity


# Ex-date and expiry validation prevents retrospective event schedules.
@pytest.mark.parametrize("expiry", [START, date(2026, 8, 31)])
def test_nonfuture_expiry_is_rejected(expiry: date) -> None:
    with pytest.raises(CalendarInputError, match="later"):
        ContractCalendar(START, expiry).resolve()


@pytest.mark.parametrize("ex_date", [START, date(2026, 8, 31), date(2027, 6, 29)])
def test_dividend_outside_contract_is_rejected(ex_date: date) -> None:
    with pytest.raises(CalendarInputError, match="ex-date"):
        ContractCalendar(
            START, EXPIRY, (DatedCashDividend(ex_date, 1.0),)
        ).resolve()


@pytest.mark.parametrize("amount", [0, -2, float("nan"), float("inf"), "n/a", True])
def test_invalid_dividend_amount_is_rejected(amount: object) -> None:
    with pytest.raises(CalendarInputError, match="finite positive"):
        ContractCalendar(
            START, EXPIRY, (DatedCashDividend(date(2027, 4, 21), amount),)
        ).resolve()


# Duplicate ex-dates cannot be represented as separate cash events.
def test_duplicate_ex_dates_are_rejected() -> None:
    same = date(2027, 4, 21)
    with pytest.raises(CalendarInputError, match="same ex-date"):
        ContractCalendar(START, EXPIRY, (
            DatedCashDividend(same, 2), DatedCashDividend(same, 1)
        )).resolve()


@pytest.mark.parametrize("invalid", ["2026-09-01", datetime(2026, 9, 1)])
def test_invalid_valuation_date_type_is_rejected(invalid: object) -> None:
    with pytest.raises(CalendarInputError, match="Valuation date"):
        ContractCalendar(invalid, EXPIRY).resolve()


@pytest.mark.parametrize("invalid", ["2027-06-28", datetime(2027, 6, 28)])
def test_invalid_expiry_date_type_is_rejected(invalid: object) -> None:
    with pytest.raises(CalendarInputError, match="Expiry date"):
        ContractCalendar(START, invalid).resolve()


def test_invalid_dividend_date_type_is_rejected() -> None:
    with pytest.raises(CalendarInputError, match="Dividend ex-date"):
        ContractCalendar(
            START, EXPIRY, (DatedCashDividend("2027-04-21", 1.0),)
        ).resolve()


# Strict type validation protects the downstream lattice conversion.
def test_invalid_dividend_entry_type_is_rejected() -> None:
    with pytest.raises(CalendarInputError, match="Each dividend"):
        ContractCalendar(START, EXPIRY, ((date(2027, 4, 21), 2.0),)).resolve()


def test_calendar_is_frozen() -> None:
    calendar = ContractCalendar(START, EXPIRY)
    with pytest.raises(FrozenInstanceError):
        calendar.expiry_date = date(2027, 7, 1)

# LOT_5C3_GUARDED_REFACTOR
