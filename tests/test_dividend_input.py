"""Lot 2C.1 — dividend-table parser invariants and historical UI behavior."""
from __future__ import annotations

import pandas as pd
import pytest

from frontend.dividend_input import parse_cash_dividends, parse_dividends
from frontend.pricing_service import DividendInput, PricingInputError


def test_zero_and_blank_rows_are_ignored():
    frame = pd.DataFrame([
        {"time": .5, "amount": 0.}, {"time": None, "amount": None},
    ])
    assert parse_cash_dividends(frame) == ()
    assert parse_dividends(frame) == ()


def test_cashflows_are_sorted_without_mutating_source_frame():
    frame = pd.DataFrame([
        {"time": .75, "amount": 1.2}, {"time": .25, "amount": 2.4},
    ])
    copy = frame.copy(deep=True)
    expected = (DividendInput(.25, 2.4), DividendInput(.75, 1.2))
    assert parse_cash_dividends(frame) == expected
    assert parse_cash_dividends(frame, context="terminal") == expected
    pd.testing.assert_frame_equal(frame, copy)


@pytest.mark.parametrize("context,fragment", [
    ("analysis", "supply an amount"),
    ("terminal", "enter an amount"),
])
def test_missing_amount_with_nonblank_date_is_rejected(context, fragment):
    with pytest.raises(PricingInputError, match=fragment):
        parse_cash_dividends(pd.DataFrame([{"time": .5, "amount": None}]),
                             context=context)


@pytest.mark.parametrize("context,fragment", [
    ("analysis", "supply a payment time"),
    ("terminal", "enter the payment time in years"),
])
def test_missing_time_with_positive_amount_is_rejected(context, fragment):
    with pytest.raises(PricingInputError, match=fragment):
        parse_cash_dividends(pd.DataFrame([{"time": None, "amount": 2.}]),
                             context=context)


@pytest.mark.parametrize("field,value,fragment", [
    ("amount", "bad", "amounts must be numeric"),
    ("amount", float("inf"), "amounts must be finite"),
    ("time", "bad", "times must be numeric"),
])
def test_invalid_values_fail_explicitly(field, value, fragment):
    row = {"time": .5, "amount": 1.}
    row[field] = value
    with pytest.raises(PricingInputError, match=fragment):
        parse_cash_dividends(pd.DataFrame([row]))


def test_no_dividend_is_invented_from_empty_frame():
    assert parse_dividends(pd.DataFrame(columns=["time", "amount"])) == ()


def test_unknown_context_rejected_without_parsing():
    with pytest.raises(ValueError, match="context"):
        parse_cash_dividends(pd.DataFrame(), context="unknown")
