"""Date-driven option contracts, calendar inputs and ticket workflows.

Consolidated without dropping individual regression cases.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, datetime
import pytest
from frontend.calendar_contract import (
    CalendarInputError,
    ContractCalendar,
    DatedCashDividend,
)
import pandas as pd
from frontend.calendar_contract import CalendarInputError, ContractCalendar, DatedCashDividend
from frontend.calendar_ticket import (
    calendar_matches_request, dated_request, parse_dated_dividends,
)
from frontend.contract_workflow import (
    REFERENCE_CALENDAR_KEY, clear_reference, get_reference_calendar,
    save_reference, save_reference_from_last_result,
)
from frontend.pricing_service import DividendInput, PricingRequest, PricingResult


# test_calendar_contract.py (tests preserved)
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


# Preserve the event-date and exercise-policy conventions in this case.
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


# test_calendar_ticket.py (tests preserved)
def _base(**overrides: object):
    return PricingRequest(**overrides)


def _price_result(request: PricingRequest):
    return PricingResult(request, 8.0, 1.0, 3, None, None, None)


# ACT/365F conversion retains exact contract settings and dates.
def test_resolve_calendar_and_preserve_model_settings() -> None:
    base = _base(spot=103, strike=110, steps=100, rate=-.01)
    request, calendar = dated_request(
        base, date(2026, 9, 1), date(2027, 9, 1),
        (DatedCashDividend(date(2027, 3, 1), 2.0),),
    )
    assert request.maturity == 1.0
    assert request.dividends == (DividendInput(181/365, 2.0),)
    assert request.spot == 103 and request.strike == 110
    assert request.model == base.model and request.steps == 100
    assert request.rate == -.01 and base.dividends == ()
    assert calendar_matches_request(calendar, request)


def test_actual_days_include_leap_day() -> None:
    request, _ = dated_request(_base(), date(2024, 1, 1), date(2025, 1, 1), ())
    assert request.maturity == pytest.approx(366/365)


# Preserve the event-date and exercise-policy conventions in this case.
def test_very_short_expiry_is_valid_for_engine_but_not_existing_legacy_ui() -> None:
    request, _ = dated_request(_base(), date(2026, 9, 1), date(2026, 9, 2), ())
    assert request.maturity == pytest.approx(1/365)


# The editor distinguishes placeholder rows from malformed dividends.
def test_dated_editor_skips_zero_row() -> None:
    frame = pd.DataFrame([{'ex_date': date(2027, 4, 21), 'amount': 0.0}])
    assert parse_dated_dividends(frame) == ()


def test_dated_editor_sorts_and_normalizes_datetime() -> None:
    frame = pd.DataFrame([
        {'ex_date': date(2027, 8, 1), 'amount': 3.0},
        {'ex_date': datetime(2027, 2, 1), 'amount': 2.0},
    ])
    events = parse_dated_dividends(frame)
    assert [(d.ex_date, d.amount) for d in events] == [
        (date(2027, 2, 1), 2.0), (date(2027, 8, 1), 3.0),
    ]


@pytest.mark.parametrize('amount', ['oops', -1.0, float('inf'), True])
def test_bad_cash_amount_is_rejected(amount: object) -> None:
    with pytest.raises(CalendarInputError):
        # Preserve the event-date and exercise-policy conventions in this case.
        parse_dated_dividends(pd.DataFrame([{'ex_date': date(2027, 4, 1), 'amount': amount}]))


def test_positive_dividend_needs_date() -> None:
    with pytest.raises(CalendarInputError, match='ex-date'):
        parse_dated_dividends(pd.DataFrame([{'ex_date': None, 'amount': 2.0}]))


def test_nonempty_date_without_amount_is_rejected() -> None:
    with pytest.raises(CalendarInputError, match='amount'):
        parse_dated_dividends(pd.DataFrame([{'ex_date': date(2027, 4, 1), 'amount': None}]))


def test_unparsed_string_ex_date_is_rejected() -> None:
    with pytest.raises(CalendarInputError, match='date'):
        parse_dated_dividends(pd.DataFrame([{'ex_date': '21/04/2027', 'amount': 4.0}]))


# Cash events must be compatible with the selected lattice time grid.
def test_colliding_events_at_same_N_are_rejected() -> None:
    with pytest.raises(CalendarInputError, match='same lattice time step'):
        dated_request(
            _base(steps=10), date(2026, 9, 1), date(2027, 9, 1),
            (DatedCashDividend(date(2027, 4, 20), 1.0),
             DatedCashDividend(date(2027, 4, 21), 1.0)),
        )


def test_binomial_with_cash_dividend_is_rejected() -> None:
    with pytest.raises(CalendarInputError, match='BinomialTree'):
        dated_request(
            _base(model='Binomial'), date(2026, 9, 1), date(2027, 9, 1),
            (DatedCashDividend(date(2027, 4, 21), 4.0),),
        )


# Calendar provenance follows successful saved results only.
def test_calendar_metadata_saved_only_with_latest_pricing_result() -> None:
    original = _base()
    req, cal = dated_request(_base(), date(2026, 9, 1), date(2027, 9, 1), ())
    state = {'last_pricing_result': _price_result(req), 'last_pricing_calendar': cal}
    assert save_reference_from_last_result(state) == req
    assert get_reference_calendar(state) == cal
    assert state[REFERENCE_CALENDAR_KEY] is cal
    state['last_pricing_result'] = _price_result(original)
    state['last_pricing_calendar'] = None
    assert get_reference_calendar(state) is cal
    assert save_reference_from_last_result(state) == original
    assert get_reference_calendar(state) is None


def test_stale_calendar_mismatch_refused_without_mutation() -> None:
    req = _base()
    cal = ContractCalendar(date(2026, 9, 1), date(2027, 9, 1))
    state = {'last_pricing_result': _price_result(req), 'last_pricing_calendar': cal}
    # Same T as default; mutate the dividend schedule to force incompatibility.
    state['last_pricing_result'] = _price_result(_base(dividends=(DividendInput(.5, 2.),)))
    old = state.copy()
    with pytest.raises(ValueError, match='calendar'):
        save_reference_from_last_result(state)
    assert state == old


def test_clearing_reference_clears_only_reference_calendar_metadata() -> None:
    req, cal = dated_request(_base(), date(2026, 9, 1), date(2027, 9, 1), ())
    state = {'last_pricing_result': _price_result(req), 'last_pricing_calendar': cal}
    save_reference_from_last_result(state)
    assert clear_reference(state)
    assert get_reference_calendar(state) is None
    assert state['last_pricing_calendar'] == cal
    assert state['last_pricing_result'].request == req


def test_explicit_numeric_reference_clears_previous_calendar() -> None:
    req, cal = dated_request(_base(), date(2026, 9, 1), date(2027, 9, 1), ())
    # Preserve the event-date and exercise-policy conventions in this case.
    state = {'last_pricing_result': _price_result(req), 'last_pricing_calendar': cal}
    save_reference_from_last_result(state)
    save_reference(state, _base(spot=105.))
    assert get_reference_calendar(state) is None

# LOT_5C4_GUARDED_REFACTOR
