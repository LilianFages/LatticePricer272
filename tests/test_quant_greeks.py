"""Unit tests for Quant Analytics finite differences and reporting conventions."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from frontend.greeks_service import (
    DAY, GreekBumps, analytic_bs_greeks, calculate_greeks,
    greeks_csv, roll_calendar_day,
)
from frontend.pricing_service import DividendInput, PricingInputError, PricingRequest


def _request(**changes):
    return replace(PricingRequest(), **changes)


def _polynomial_pricer(request):
    """Smooth toy pricing function, NOT a financial model or production fallback."""
    value = (
        0.7 * request.spot
        + 0.01 * request.spot ** 2
        + 3.5 * request.volatility
        + 14.0 * request.rate
        + 5.0 * request.maturity
        + sum(4.0 * d.time for d in request.dividends)
    )
    return SimpleNamespace(price=value, elapsed_ms=2.0)


def test_greek_stencils_and_reporting_units():
    result = calculate_greeks(_request(), GreekBumps(), pricer=_polynomial_pricer)
    assert result.delta == pytest.approx(2.7, abs=1e-10)
    assert result.gamma == pytest.approx(0.02, abs=1e-10)
    assert result.vega_1pp == pytest.approx(0.035, abs=1e-10)
    assert result.rho_1bp == pytest.approx(0.0014, abs=1e-10)
    assert result.theta_1day == pytest.approx(-5*DAY, abs=1e-10)
    assert result.evaluations == 10
    assert result.total_runtime_ms == pytest.approx(20.0)
    assert result.volatility_scheme == "central"


def test_near_zero_volatility_uses_forward_second_order():
    result = calculate_greeks(_request(volatility=0.001), pricer=_polynomial_pricer)
    assert result.vega_1pp == pytest.approx(0.035, abs=1e-10)
    assert result.volatility_scheme.startswith("forward")
    assert "volatility-" not in result.scenario_prices
    assert "volatility++" in result.scenario_prices


def test_zero_volatility_remains_valid_for_mocked_engine():
    result = calculate_greeks(_request(volatility=0.0), pricer=_polynomial_pricer)
    assert result.vega_1pp == pytest.approx(0.035, abs=1e-10)
    assert result.analytic is None


def test_calendar_roll_advances_dividend_and_maturity():
    request = _request(model="Hybrid", dividends=(DividendInput(0.5, 3.0),))
    rolled = roll_calendar_day(request)
    assert rolled is not None
    assert rolled.maturity == pytest.approx(request.maturity - DAY)
    assert rolled.dividends[0].time == pytest.approx(0.5-DAY)
    result = calculate_greeks(request, pricer=_polynomial_pricer)
    assert result.theta_1day == pytest.approx(-9.0*DAY, abs=1e-10)


def test_theta_unavailable_near_cash_ex_date_without_faking_zero():
    request = _request(
        model="Hybrid", dividends=(DividendInput(DAY/2, 0.25),),
    )
    result = calculate_greeks(request, pricer=_polynomial_pricer)
    assert result.theta_1day is None
    assert result.delta is not None
    assert result.evaluations == 9
    assert any("Theta" in issue for issue in result.issues)


def test_theta_unavailable_when_maturity_under_one_day():
    request = _request(maturity=DAY/2)
    result = calculate_greeks(request, pricer=_polynomial_pricer)
    assert result.theta_1day is None
    # At very short maturity the automatic grid spacing is below the Delta
    # bump, so the same two spot shocks are reused for Gamma.
    assert result.evaluations == 7


def test_one_failed_shock_does_not_zero_out_other_greeks():
    def partial(request):
        if request.rate > 0.02:
            raise RuntimeError("Synthetic shock failure")
        return _polynomial_pricer(request)

    result = calculate_greeks(_request(), pricer=partial)
    assert result.rho_1bp is None
    assert result.delta == pytest.approx(2.7)
    assert result.gamma == pytest.approx(0.02)
    assert len(result.issues) == 1
    assert "rate+" in result.issues[0]


def test_base_failure_is_not_silenced():
    def fail(_):
        raise RuntimeError("Synthetic base failure")

    with pytest.raises(PricingInputError, match="Base contract"):
        calculate_greeks(_request(), pricer=fail)


def test_perturbation_input_validation():
    with pytest.raises(PricingInputError):
        GreekBumps(spot_pct=0).validate()
    with pytest.raises(PricingInputError):
        GreekBumps(vol_pp=float("nan")).validate()
    with pytest.raises(PricingInputError):
        GreekBumps(rate_bp=101).validate()


def test_black_scholes_reference_restricted_to_applicable_contracts():
    assert analytic_bs_greeks(_request()) is not None
    assert analytic_bs_greeks(_request(exercise="American")) is None
    assert analytic_bs_greeks(_request(volatility=0.0)) is None
    assert analytic_bs_greeks(_request(strike=0.0)) is None
    assert analytic_bs_greeks(_request(model="Hybrid", dividends=(DividendInput(0.5, 2.0),))) is None


def test_black_scholes_put_call_greeks_obey_parity():
    call = analytic_bs_greeks(_request(option_type="Call"))
    put = analytic_bs_greeks(_request(option_type="Put"))
    assert call is not None and put is not None
    assert call["Delta"] - put["Delta"] == pytest.approx(1.0)
    assert call["Gamma"] == pytest.approx(put["Gamma"])
    assert call["Vega / 1pp"] == pytest.approx(put["Vega / 1pp"])
    assert call["Rho / 1bp"] - put["Rho / 1bp"] == pytest.approx(
        100.0 * 1.0 * __import__("math").exp(-0.02) * 0.0001
    )
    assert call["Theta / 1d"] - put["Theta / 1d"] == pytest.approx(
        -(0.02 * 100.0 * __import__("math").exp(-0.02))/365
    )


def test_export_contains_bumps_and_all_five_greeks():
    result = calculate_greeks(_request(), pricer=_polynomial_pricer)
    output = greeks_csv(result)
    assert "spot_bump_pct" in output
    assert "vol_bump_pp" in output
    assert "rate_bump_bp" in output
    assert "Theta / 1d" in output
    assert "analytic_BS" in output
    assert "contract,model" in output


def test_fd_greeks_against_independent_black_scholes_repricing():
    """Finance-scale check independent of lattice numerical convergence."""
    import math

    def n(x):
        return 0.5 * math.erfc(-x / math.sqrt(2))

    def bs_engine(request):
        s, k, sigma, t, r = (
            request.spot, request.strike, request.volatility,
            request.maturity, request.rate,
        )
        d1 = (math.log(s/k) + (r + 0.5*sigma*sigma)*t) / (sigma*math.sqrt(t))
        d2 = d1-sigma*math.sqrt(t)
        if request.option_type == "Call":
            value = s*n(d1)-k*math.exp(-r*t)*n(d2)
        else:
            value = k*math.exp(-r*t)*n(-d2)-s*n(-d1)
        return SimpleNamespace(price=value, elapsed_ms=1.0)

    for kind in ("Call", "Put"):
        request = _request(option_type=kind)
        actual = calculate_greeks(
            request, GreekBumps(spot_pct=0.1, vol_pp=0.1, rate_bp=1.0,
                                gamma_spot_pct=0.1),
            pricer=bs_engine,
        )
        ref = actual.analytic
        assert ref is not None
        assert actual.delta == pytest.approx(ref["Delta"], abs=3e-6)
        assert actual.gamma == pytest.approx(ref["Gamma"], abs=2e-6)
        assert actual.vega_1pp == pytest.approx(ref["Vega / 1pp"], abs=3e-6)
        assert actual.rho_1bp == pytest.approx(ref["Rho / 1bp"], abs=3e-8)
        assert actual.theta_1day == pytest.approx(ref["Theta / 1d"], abs=2e-5)
