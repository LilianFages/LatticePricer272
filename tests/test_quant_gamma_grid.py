"""Regression: Greeks should not report a false zero Gamma from a too-small spot bump."""

from dataclasses import replace
from types import SimpleNamespace
import math

import pytest

from frontend.greeks_service import GreekBumps, calculate_greeks, select_gamma_bump
from frontend.pricing_service import PricingRequest
from frontend import analytics_views as views


def test_grid_spacing_trinomial_atm_100_steps():
    request = PricingRequest()
    pct, scheme, clipped = select_gamma_bump(request, GreekBumps())
    assert scheme == "grid-aware" and not clipped
    assert pct == pytest.approx(100*math.expm1(.2*math.sqrt(3/100)))
    assert 3.4 < pct < 3.6


def test_manual_gamma_width_is_respected():
    pct, scheme, clipped = select_gamma_bump(
        PricingRequest(), GreekBumps(gamma_spot_pct=5.0))
    assert (pct, scheme, clipped) == (5.0, "manual", False)


def test_default_gamma_stencil_spans_fixed_trinomial_payoff_kink():
    """Separate demonstrator grid, independent of production engine.

    It implements an ideal no-dividend recombining trinomial with probabilities
    from the stated course lattice, not production pricing-service internals.
    """
    def price_on_grid(request):
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


def test_gamma_bump_independent_of_delta_and_metadata():
    def quadratic(req):
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
def test_active_risk_numerics_forwards_gamma_choice(monkeypatch, gamma_choice, expected):
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
    monkeypatch.setattr(views, "_get_request", lambda *_a: request)

    def record_study(actual_request, bumps, depths, *, gamma_bump_pct):
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


def test_active_risk_numerics_does_not_reprice_on_view_rerun(monkeypatch):
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
    monkeypatch.setattr(views, "_render_stability_results", displayed.append)
    views.render_greeks_stability_tab()
    assert displayed == [previous]


def test_invalid_manual_width_rejected():
    with pytest.raises(ValueError,match="Gamma"):
        GreekBumps(gamma_spot_pct=0.0).validate()
