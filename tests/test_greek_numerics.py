"""Finite-difference Greek estimates, sensitivity grids and numerical stability.

Consolidated without dropping individual regression cases.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import math
import pytest
from frontend.greeks_service import GreekBumps, calculate_greeks, select_gamma_bump
from frontend.pricing_service import PricingRequest
from frontend import analytics_views as views
from frontend.greeks_service import (
    DAY, GreekBumps, analytic_bs_greeks, calculate_greeks,
    greeks_csv, roll_calendar_day,
)
from frontend.pricing_service import DividendInput, PricingInputError, PricingRequest
from frontend.analytics_service import StabilityObservation
from frontend.research_risk_numerics import (
    GAMMA_N_MODES, n_sweep_gamma_bump, observations, summarize_stability,
)


# test_quant_gamma_grid.py (tests preserved)
def test_grid_spacing_trinomial_atm_100_steps() -> None:
    request = PricingRequest()
    pct, scheme, clipped = select_gamma_bump(request, GreekBumps())
    assert scheme == "grid-aware" and not clipped
    assert pct == pytest.approx(100*math.expm1(.2*math.sqrt(3/100)))
    assert 3.4 < pct < 3.6


def test_manual_gamma_width_is_respected() -> None:
    pct, scheme, clipped = select_gamma_bump(
        PricingRequest(), GreekBumps(gamma_spot_pct=5.0))
    assert (pct, scheme, clipped) == (5.0, "manual", False)


def test_default_gamma_stencil_spans_fixed_trinomial_payoff_kink() -> None:
    """Separate demonstrator grid, independent of production engine.

    It implements an ideal no-dividend recombining trinomial with probabilities
    from the stated course lattice, not production pricing-service internals.
    """
    # Check the numerical Greek or hedge rule independently of UI formatting.
    def price_on_grid(request: PricingRequest):
        s,k,r,sig,t,n=(request.spot,request.strike,request.rate,
                         request.volatility,request.maturity,request.steps)
        dt=t/n
        alpha=math.exp(sig*math.sqrt(3*dt))
        pd=math.expm1(sig**2*dt)/((1-alpha)*(1/alpha**2-1))
        pu=pd/alpha
        pm=1-pu-pd
        values={j:max(0,s*math.exp(r*t)*alpha**j-k) for j in range(-n,n+1)}
        for step in range(n-1,-1,-1):
            values={j:math.exp(-r*dt)*(pu*values[j+1]+pm*values[j]+pd*values[j-1])
                    for j in range(-step,step+1)}
        return SimpleNamespace(price=values[0],elapsed_ms=1.0)

    request=PricingRequest()
    narrow=calculate_greeks(request, GreekBumps(gamma_spot_pct=1),pricer=price_on_grid)
    fixed=calculate_greeks(request, GreekBumps(),pricer=price_on_grid)
    # Check the Gamma bump against trinomial payoff-grid spacing.
    assert abs(narrow.gamma) < 1e-8
    assert fixed.gamma is not None and fixed.analytic is not None
    assert fixed.gamma == pytest.approx(fixed.analytic["Gamma"], abs=.003)
    assert fixed.gamma_spot_bump_pct > 3.0
    assert fixed.evaluations == 10
    assert "spot_gamma+" in fixed.scenario_prices


def test_gamma_bump_independent_of_delta_and_metadata() -> None:
    def quadratic(req: PricingRequest):
        return SimpleNamespace(price=req.spot**2+req.volatility+req.rate+req.maturity,elapsed_ms=1.0)
    result=calculate_greeks(PricingRequest(),GreekBumps(),pricer=quadratic)
    assert result.delta == pytest.approx(200.0)
    assert result.gamma == pytest.approx(2.0)
    assert result.gamma_spot_bump_pct > result.bumps.spot_pct


@pytest.mark.parametrize(
    ("gamma_choice", "expected"),
    [
        ("Grid-aware (adaptive)", None),
        ("Fixed ±3.5%", 3.5),
        ("Fixed ±5%", 5.0),
    ],
)
# Check the numerical Greek or hedge rule independently of UI formatting.
def test_active_risk_numerics_forwards_gamma_choice(monkeypatch: pytest.MonkeyPatch, gamma_choice: object, expected: object) -> None:
    """The live workspace, not the retired Greeks page, forwards Gamma's N bump."""
    from contextlib import nullcontext

    request = PricingRequest()
    observed = {}
    state = {}

    monkeypatch.setattr(
        views, "st",
        SimpleNamespace(
            columns=lambda *_a, **_kw: (nullcontext(), nullcontext()),
            markdown=lambda *_a, **_kw: None,
            spinner=lambda *_a, **_kw: nullcontext(),
            session_state=state,
        ),
    )
    monkeypatch.setattr(
        views, "_greeks_stability_input_form",
        lambda: ({"example": "saved"}, "Trinomial · Allocation pruning",
                 (1.0, 0.5), (80, 40), 1e-18, gamma_choice, True),
    )
    # Check the numerical Greek or hedge rule independently of UI formatting.
    monkeypatch.setattr(views, "_get_request", lambda *_a: request)

    def record_study(actual_request: object, bumps: object, depths: object, *, gamma_bump_pct: object):
        observed.update(request=actual_request, bumps=bumps,
                        depths=depths, gamma_bump_pct=gamma_bump_pct)
        return ()

    monkeypatch.setattr(views, "greeks_stability", record_study)
    monkeypatch.setattr(views, "_render_stability_results", lambda saved: None)

    views.render_greeks_stability_tab()
    assert observed["request"] is request
    assert observed["bumps"] == (0.5, 1.0)
    assert observed["depths"] == (40, 80)
    assert observed["gamma_bump_pct"] == expected
    assert state["qa_stab_result"] == ((), request)


def test_active_risk_numerics_does_not_reprice_on_view_rerun(monkeypatch: pytest.MonkeyPatch) -> None:
    # Check the numerical Greek or hedge rule independently of UI formatting.
    """Charts and display state must never launch fresh lattice pricing."""
    from contextlib import nullcontext

    previous = (("cached",), PricingRequest())
    displayed = []
    monkeypatch.setattr(
        views, "st",
        SimpleNamespace(
            columns=lambda *_a, **_kw: (nullcontext(), nullcontext()),
            markdown=lambda *_a, **_kw: None,
            session_state={"qa_stab_result": previous},
        ),
    )
    monkeypatch.setattr(
        views, "_greeks_stability_input_form",
        lambda: ({}, "Trinomial · Allocation pruning",
                 (1.0,), (40, 80), 1e-18, "Grid-aware (adaptive)", False),
    )
    monkeypatch.setattr(
        views, "greeks_stability",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("Unexpected repricing")),
    )
    # Check the numerical Greek or hedge rule independently of UI formatting.
    monkeypatch.setattr(views, "_render_stability_results", displayed.append)
    views.render_greeks_stability_tab()
    assert displayed == [previous]


def test_invalid_manual_width_rejected() -> None:
    with pytest.raises(ValueError,match="Gamma"):
        GreekBumps(gamma_spot_pct=0.0).validate()


# test_quant_greeks.py (tests preserved)
def _request(**changes: object):
    return replace(PricingRequest(), **changes)


def _polynomial_pricer(request: PricingRequest):
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


def test_greek_stencils_and_reporting_units() -> None:
    result = calculate_greeks(_request(), GreekBumps(), pricer=_polynomial_pricer)
    # Verify the expected numerical or workflow invariant on this case.
    assert result.delta == pytest.approx(2.7, abs=1e-10)
    assert result.gamma == pytest.approx(0.02, abs=1e-10)
    assert result.vega_1pp == pytest.approx(0.035, abs=1e-10)
    assert result.rho_1bp == pytest.approx(0.0014, abs=1e-10)
    # Greek values must respect the market-reporting units of each shock.
    assert result.theta_1day == pytest.approx(-5*DAY, abs=1e-10)
    assert result.evaluations == 10
    assert result.total_runtime_ms == pytest.approx(20.0)
    assert result.volatility_scheme == "central"


def test_near_zero_volatility_uses_forward_second_order() -> None:
    result = calculate_greeks(_request(volatility=0.001), pricer=_polynomial_pricer)
    assert result.vega_1pp == pytest.approx(0.035, abs=1e-10)
    assert result.volatility_scheme.startswith("forward")
    assert "volatility-" not in result.scenario_prices
    assert "volatility++" in result.scenario_prices


def test_zero_volatility_remains_valid_for_mocked_engine() -> None:
    result = calculate_greeks(_request(volatility=0.0), pricer=_polynomial_pricer)
    assert result.vega_1pp == pytest.approx(0.035, abs=1e-10)
    # The mocked pricing engine supports the zero-volatility limit.
    assert result.analytic is None


def test_calendar_roll_advances_dividend_and_maturity() -> None:
    request = _request(model="Hybrid", dividends=(DividendInput(0.5, 3.0),))
    rolled = roll_calendar_day(request)
    assert rolled is not None
    assert rolled.maturity == pytest.approx(request.maturity - DAY)
    assert rolled.dividends[0].time == pytest.approx(0.5-DAY)
    result = calculate_greeks(request, pricer=_polynomial_pricer)
    assert result.theta_1day == pytest.approx(-9.0*DAY, abs=1e-10)


def test_theta_unavailable_near_cash_ex_date_without_faking_zero() -> None:
    request = _request(
        model="Hybrid", dividends=(DividendInput(DAY/2, 0.25),),
    )
    result = calculate_greeks(request, pricer=_polynomial_pricer)
    assert result.theta_1day is None
    # Verify the expected numerical or workflow invariant on this case.
    assert result.delta is not None
    assert result.evaluations == 9
    assert any("Theta" in issue for issue in result.issues)


def test_theta_unavailable_when_maturity_under_one_day() -> None:
    request = _request(maturity=DAY/2)
    result = calculate_greeks(request, pricer=_polynomial_pricer)
    assert result.theta_1day is None
    # At very short maturity the automatic grid spacing is below the Delta
    # bump, so the same two spot shocks are reused for Gamma.
    assert result.evaluations == 7


def test_one_failed_shock_does_not_zero_out_other_greeks() -> None:
    def partial(request: PricingRequest):
        if request.rate > 0.02:
            raise RuntimeError("Synthetic shock failure")
        return _polynomial_pricer(request)

    result = calculate_greeks(_request(), pricer=partial)
    assert result.rho_1bp is None
    assert result.delta == pytest.approx(2.7)
    assert result.gamma == pytest.approx(0.02)
    assert len(result.issues) == 1
    assert "rate+" in result.issues[0]


# Failure of the unshocked base price must remain explicit.
def test_base_failure_is_not_silenced() -> None:
    def fail(_: object):
        raise RuntimeError("Synthetic base failure")

    with pytest.raises(PricingInputError, match="Base contract"):
        calculate_greeks(_request(), pricer=fail)


def test_perturbation_input_validation() -> None:
    with pytest.raises(PricingInputError):
        GreekBumps(spot_pct=0).validate()
    with pytest.raises(PricingInputError):
        GreekBumps(vol_pp=float("nan")).validate()
    with pytest.raises(PricingInputError):
        GreekBumps(rate_bp=101).validate()


# Analytical Greeks are unavailable for unsupported exercise or dividend terms.
def test_black_scholes_reference_restricted_to_applicable_contracts() -> None:
    assert analytic_bs_greeks(_request()) is not None
    assert analytic_bs_greeks(_request(exercise="American")) is None
    assert analytic_bs_greeks(_request(volatility=0.0)) is None
    assert analytic_bs_greeks(_request(strike=0.0)) is None
    assert analytic_bs_greeks(_request(model="Hybrid", dividends=(DividendInput(0.5, 2.0),))) is None


def test_black_scholes_put_call_greeks_obey_parity() -> None:
    call = analytic_bs_greeks(_request(option_type="Call"))
    put = analytic_bs_greeks(_request(option_type="Put"))
    assert call is not None and put is not None
    assert call["Delta"] - put["Delta"] == pytest.approx(1.0)
    assert call["Gamma"] == pytest.approx(put["Gamma"])
    assert call["Vega / 1pp"] == pytest.approx(put["Vega / 1pp"])
    assert call["Rho / 1bp"] - put["Rho / 1bp"] == pytest.approx(
        100.0 * 1.0 * __import__("math").exp(-0.02) * 0.0001
    )
    # Analytic put and call Greeks must obey their parity identities.
    assert call["Theta / 1d"] - put["Theta / 1d"] == pytest.approx(
        -(0.02 * 100.0 * __import__("math").exp(-0.02))/365
    )


def test_export_contains_bumps_and_all_five_greeks() -> None:
    result = calculate_greeks(_request(), pricer=_polynomial_pricer)
    output = greeks_csv(result)
    assert "spot_bump_pct" in output
    assert "vol_bump_pp" in output
    assert "rate_bump_bp" in output
    assert "Theta / 1d" in output
    assert "analytic_BS" in output
    assert "contract,model" in output


def test_fd_greeks_against_independent_black_scholes_repricing() -> None:
    """Finance-scale check independent of lattice numerical convergence."""
    import math

    # Check the numerical Greek or hedge rule independently of UI formatting.
    def n(x: float):
        return 0.5 * math.erfc(-x / math.sqrt(2))

    def bs_engine(request: PricingRequest):
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
        # Compare finite-difference sensitivities to analytic model identities.
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


# test_research_risk_numerics.py (tests preserved)
def _obs(
    dim: str, x: float, delta: float = .5,
    gamma: float | None = .02, status: str = "ok",
    gamma_bump: float = 3.5,
) -> StabilityObservation:
    """Create a deterministic observation without running the pricer."""
    return StabilityObservation(
        dim, x, int(x) if dim == "steps" else 100,
        1.0, gamma_bump, delta, gamma, status,
    )


def test_adaptive_n_stencil_is_explicit() -> None:
    assert n_sweep_gamma_bump("Grid-aware (adaptive)") is None


@pytest.mark.parametrize("mode,expected", [
    ("Fixed ±1%", 1.), ("Fixed ±2%", 2.),
    ("Fixed ±3.5%", 3.5), ("Fixed ±5%", 5.),
])
# Check the numerical Greek or hedge rule independently of UI formatting.
def test_fixed_n_stencil_choices(mode: str, expected: float) -> None:
    assert n_sweep_gamma_bump(mode) == expected


def test_unknown_n_stencil_is_rejected() -> None:
    with pytest.raises(ValueError):
        n_sweep_gamma_bump("fixed unknown")


# Explicitly omit unavailable observations instead of plotting zeros.
def test_failed_and_missing_points_are_omitted_not_zero_filled() -> None:
    rows = (_obs("steps", 40, status="error"),
            _obs("steps", 80, gamma=None),
            _obs("steps", 100, gamma=.01))
    assert [p.coordinate for p in observations(rows, dimension="steps", greek="Gamma")] == [100.]


def test_deviation_requires_analytical_comparator() -> None:
    with pytest.raises(ValueError, match="analytical"):
        observations((_obs("steps", 40),), dimension="steps", greek="Delta", deviation=True)


def test_absolute_and_deviation_are_distinct() -> None:
    rows = (_obs("steps", 40, delta=.4), _obs("steps", 80, delta=.45))
    out = observations(rows, dimension="steps", greek="Delta", analytical=.5, deviation=True)
    assert [p.plotted for p in out] == pytest.approx([-.1, -.05])
    assert [p.measured for p in out] == pytest.approx([.4, .45])


# Check the numerical Greek or hedge rule independently of UI formatting.
def test_spot_and_n_dimensions_remain_separate() -> None:
    rows = (_obs("spot bump", .5, gamma=.01), _obs("steps", 80, gamma=.02))
    out = observations(rows, dimension="spot bump", greek="Gamma")
    assert len(out) == 1 and out[0].coordinate == .5


# These summaries describe numerical stability rather than accuracy.
def test_summaries_are_spreads_not_accuracy_claims() -> None:
    rows = (_obs("steps", 40, .41, .01), _obs("steps", 80, .44, .03))
    result = summarize_stability(rows, {"Delta": .5, "Gamma": .02})
    assert result.successful == result.total == 2
    assert result.n_delta_span == pytest.approx(.03)
    assert result.n_gamma_span == pytest.approx(.02)
    assert result.n_delta_max_bs_gap == pytest.approx(.09)
    assert result.n_gamma_max_bs_gap == pytest.approx(.01)


def test_near_zero_gamma_flag_only_for_valid_bs_reference() -> None:
    rows = (_obs("steps", 40, gamma=0.), _obs("steps", 80, gamma=.03))
    assert summarize_stability(rows, {"Delta": .5, "Gamma": .02}).near_zero_n_gamma == 1
    assert summarize_stability(rows, None).near_zero_n_gamma == 0


# Invalid plotted Greeks must never appear as valid measurements.
def test_nonfinite_risk_estimate_is_not_plotted() -> None:
    rows = (_obs("steps", 40, gamma=float("nan")), _obs("steps", 80, gamma=.03))
    result = observations(rows, dimension="steps", greek="Gamma")
    assert len(result) == 1 and result[0].coordinate == 80.


def test_invalid_greek_dimension_rejected() -> None:
    with pytest.raises(ValueError):
        observations((), dimension="wrong", greek="Vega")


def test_mode_list_is_stable() -> None:
    assert len(GAMMA_N_MODES) == len(set(GAMMA_N_MODES)) == 5
