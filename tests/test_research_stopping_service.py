"""Regression tests for live dividend-vs-no-dividend stopping comparisons."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from frontend.pricing_service import DividendInput, PricingInputError, PricingRequest
from frontend.research_stopping_service import compare_dividend_exercise


def _request(**kwargs):
    base = PricingRequest(
        model="Trinomial · Allocation pruning", exercise="American",
        option_type="Put", steps=20,
        dividends=(DividendInput(.5, 3.),),
    )
    return replace(base, **kwargs)


def test_same_market_and_grid_without_mutating_original():
    observed = []

    def fake_analyzer(req):
        observed.append(req)
        return SimpleNamespace(request=req, premium=8.0 if req.dividends else 7.0,
                               early_exercise_probability=.25 if req.dividends else .30)

    original = _request()
    result = compare_dividend_exercise(original, analyzer=fake_analyzer)
    assert observed[0] is original
    assert observed[1] is not original
    assert observed[1] == replace(original, dividends=())
    assert original.dividends == (DividendInput(.5, 3.),)
    assert result.premium_difference == pytest.approx(1.0)
    assert result.node_exercise_difference == pytest.approx(-.05)


def test_no_dividends_rejected_without_pricing():
    calls = []
    with pytest.raises(PricingInputError, match="Add at least one"):
        compare_dividend_exercise(_request(dividends=()), analyzer=calls.append)
    assert calls == []


def test_binomial_cash_dividend_is_not_silently_substituted():
    with pytest.raises(PricingInputError, match="Binomial"):
        compare_dividend_exercise(_request(model="Binomial"))


def test_two_dividends_in_same_lattice_step_rejected():
    req = _request(dividends=(DividendInput(.451, 2.), DividendInput(.49, 1.)))
    with pytest.raises(PricingInputError, match="same lattice"):
        compare_dividend_exercise(req)


def test_interactive_depth_guard_precedes_analysis():
    calls = []
    with pytest.raises(PricingInputError, match="at most N=200"):
        compare_dividend_exercise(_request(steps=201), analyzer=calls.append)
    assert not calls


def test_analyzer_failure_is_not_silenced():
    def fail(_):
        raise RuntimeError("engine failure")
    with pytest.raises(RuntimeError, match="engine failure"):
        compare_dividend_exercise(_request(), analyzer=fail)


def test_two_cashflow_events_preserved_in_dividend_case():
    req = _request(steps=20, dividends=(DividendInput(.25, 2.), DividendInput(.75, 1.)))
    seen = []

    def analyzer(r):
        seen.append(r)
        return SimpleNamespace(request=r, premium=3.0,
                               early_exercise_probability=0.0)

    compare_dividend_exercise(req, analyzer=analyzer)
    assert seen[0].dividends == req.dividends
    assert seen[1].dividends == ()
    assert seen[1].steps == seen[0].steps


def test_real_engine_comparison_uses_two_valid_studies():
    result = compare_dividend_exercise(_request(steps=20))
    assert result.with_dividends.request.dividends
    assert not result.without_dividends.request.dividends
    assert result.with_dividends.request.spot == result.without_dividends.request.spot
    assert len(result.with_dividends.dates) == 21
    assert len(result.without_dividends.dates) == 21
    assert result.with_dividends.worst_probability_mass_error < 2e-10
    assert result.without_dividends.worst_probability_mass_error < 2e-10
