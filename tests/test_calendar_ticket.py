"""Phase 1B-D2.1: calendar parsing, pricing conversion and reference metadata."""

from datetime import date, datetime

import pandas as pd
import pytest

from frontend.calendar_contract import CalendarInputError, ContractCalendar, DatedCashDividend
from frontend.calendar_ticket import (
    calendar_matches_request, dated_request, parse_dated_dividends,
)
from frontend.contract_workflow import (
    REFERENCE_CALENDAR_KEY, clear_reference, get_reference_calendar,
    save_reference, save_reference_from_last_result,
)
from frontend.pricing_service import DividendInput, PricingRequest, PricingResult


def _base(**overrides):
    return PricingRequest(**overrides)


def _price_result(request):
    return PricingResult(request, 8.0, 1.0, 3, None, None, None)


# ACT/365F conversion retains exact contract settings and dates.
def test_resolve_calendar_and_preserve_model_settings():
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


def test_actual_days_include_leap_day():
    request, _ = dated_request(_base(), date(2024, 1, 1), date(2025, 1, 1), ())
    assert request.maturity == pytest.approx(366/365)


def test_very_short_expiry_is_valid_for_engine_but_not_existing_legacy_ui():
    request, _ = dated_request(_base(), date(2026, 9, 1), date(2026, 9, 2), ())
    assert request.maturity == pytest.approx(1/365)


# The editor distinguishes placeholder rows from malformed dividends.
def test_dated_editor_skips_zero_row():
    frame = pd.DataFrame([{'ex_date': date(2027, 4, 21), 'amount': 0.0}])
    assert parse_dated_dividends(frame) == ()


def test_dated_editor_sorts_and_normalizes_datetime():
    frame = pd.DataFrame([
        {'ex_date': date(2027, 8, 1), 'amount': 3.0},
        {'ex_date': datetime(2027, 2, 1), 'amount': 2.0},
    ])
    events = parse_dated_dividends(frame)
    assert [(d.ex_date, d.amount) for d in events] == [
        (date(2027, 2, 1), 2.0), (date(2027, 8, 1), 3.0),
    ]


@pytest.mark.parametrize('amount', ['oops', -1.0, float('inf'), True])
def test_bad_cash_amount_is_rejected(amount):
    with pytest.raises(CalendarInputError):
        parse_dated_dividends(pd.DataFrame([{'ex_date': date(2027, 4, 1), 'amount': amount}]))


def test_positive_dividend_needs_date():
    with pytest.raises(CalendarInputError, match='ex-date'):
        parse_dated_dividends(pd.DataFrame([{'ex_date': None, 'amount': 2.0}]))


def test_nonempty_date_without_amount_is_rejected():
    with pytest.raises(CalendarInputError, match='amount'):
        parse_dated_dividends(pd.DataFrame([{'ex_date': date(2027, 4, 1), 'amount': None}]))


def test_unparsed_string_ex_date_is_rejected():
    with pytest.raises(CalendarInputError, match='date'):
        parse_dated_dividends(pd.DataFrame([{'ex_date': '21/04/2027', 'amount': 4.0}]))


# Cash events must be compatible with the selected lattice time grid.
def test_colliding_events_at_same_N_are_rejected():
    with pytest.raises(CalendarInputError, match='same lattice time step'):
        dated_request(
            _base(steps=10), date(2026, 9, 1), date(2027, 9, 1),
            (DatedCashDividend(date(2027, 4, 20), 1.0),
             DatedCashDividend(date(2027, 4, 21), 1.0)),
        )


def test_binomial_with_cash_dividend_is_rejected():
    with pytest.raises(CalendarInputError, match='BinomialTree'):
        dated_request(
            _base(model='Binomial'), date(2026, 9, 1), date(2027, 9, 1),
            (DatedCashDividend(date(2027, 4, 21), 4.0),),
        )


# Calendar provenance follows successful saved results only.
def test_calendar_metadata_saved_only_with_latest_pricing_result():
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


def test_stale_calendar_mismatch_refused_without_mutation():
    req = _base()
    cal = ContractCalendar(date(2026, 9, 1), date(2027, 9, 1))
    state = {'last_pricing_result': _price_result(req), 'last_pricing_calendar': cal}
    # Same T as default; mutate the dividend schedule to force incompatibility.
    state['last_pricing_result'] = _price_result(_base(dividends=(DividendInput(.5, 2.),)))
    old = state.copy()
    with pytest.raises(ValueError, match='calendar'):
        save_reference_from_last_result(state)
    assert state == old


def test_clearing_reference_clears_only_reference_calendar_metadata():
    req, cal = dated_request(_base(), date(2026, 9, 1), date(2027, 9, 1), ())
    state = {'last_pricing_result': _price_result(req), 'last_pricing_calendar': cal}
    save_reference_from_last_result(state)
    assert clear_reference(state)
    assert get_reference_calendar(state) is None
    assert state['last_pricing_calendar'] == cal
    assert state['last_pricing_result'].request == req


def test_explicit_numeric_reference_clears_previous_calendar():
    req, cal = dated_request(_base(), date(2026, 9, 1), date(2027, 9, 1), ())
    state = {'last_pricing_result': _price_result(req), 'last_pricing_calendar': cal}
    save_reference_from_last_result(state)
    save_reference(state, _base(spot=105.))
    assert get_reference_calendar(state) is None

# LOT_5C4_GUARDED_REFACTOR
