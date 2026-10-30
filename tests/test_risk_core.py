"""Risk kernel has no frontend dependency, preserves D3 Greek/hedge units."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import pytest

from src.risk.hedging import StockHedge, calculate_stock_hedge
from src.risk.numerics import (
    black_scholes_vanna_vomma, vanna_from_samples, vomma_from_samples,
)


def _polynomial(s: float, sigma: float) -> float:
    return 4.0 + 0.5*s + 3*s*sigma + 7*sigma*sigma + 0.8*s*sigma*sigma


@pytest.mark.parametrize("s,sigma", [(100., .2), (115., .3)])
def test_vanna_centered_polynomial(s: float, sigma: float) -> None:
    ds, dv = 1., .005
    actual = vanna_from_samples(
        plus_high=_polynomial(s+ds,sigma+dv),
        minus_high=_polynomial(s-ds,sigma+dv),
        plus_low=_polynomial(s+ds,sigma-dv),
        minus_low=_polynomial(s-ds,sigma-dv),
        spot_bump=ds,vol_bump=dv,centered=True,
    )
    assert actual == pytest.approx(.01*(3+1.6*sigma), abs=1e-11)


# Validate both centered and forward finite-difference Vanna stencils.
def test_vanna_forward_polynomial() -> None:
    s,sigma,ds,dv = 100., .0001, 1., .005
    actual = vanna_from_samples(
        plus_high=_polynomial(s+ds,sigma+dv),
        minus_high=_polynomial(s-ds,sigma+dv),
        plus_low=_polynomial(s+ds,sigma),
        minus_low=_polynomial(s-ds,sigma),
        spot_bump=ds,vol_bump=dv,centered=False,
    )
    assert actual == pytest.approx(.01*(3+1.6*sigma+.8*dv))


@pytest.mark.parametrize("centered", [True,False])
def test_vomma_polynomial(centered: bool) -> None:
    s,sigma,dv = 100., .2, .005
    lo_sigma = sigma-dv if centered else sigma+2*dv
    actual = vomma_from_samples(
        base=_polynomial(s,sigma),high=_polynomial(s,sigma+dv),
        low=_polynomial(s,lo_sigma),vol_bump=dv,centered=centered,
    )
    assert actual == pytest.approx(0.0001*(14+1.6*s),abs=1e-10)


@pytest.mark.parametrize("fn,kwargs", [
    (vanna_from_samples,dict(plus_high=None,minus_high=1.,plus_low=2.,
                             minus_low=3.,spot_bump=1.,vol_bump=.01,centered=True)),
    (vomma_from_samples,dict(base=5.,high=6.,low=None,
                             vol_bump=.01,centered=True)),
])
def test_missing_shock_is_unavailable(
    fn: Callable[..., float | None], kwargs: dict[str, object]
) -> None:
    assert fn(**kwargs) is None


@pytest.mark.parametrize("bad_bump", [0.,-1.,float("nan"),float("inf")])
def test_invalid_vanna_bump(bad_bump: float) -> None:
    with pytest.raises(ValueError):
        vanna_from_samples(plus_high=1.,minus_high=2.,plus_low=3.,
                           minus_low=4.,spot_bump=1.,vol_bump=bad_bump,
                           centered=True)


@pytest.mark.parametrize("bad_bump", [0.,-1.,float("nan"),float("inf")])
def test_invalid_vomma_bump(bad_bump: float) -> None:
    with pytest.raises(ValueError):
        vomma_from_samples(base=1.,high=2.,low=3.,vol_bump=bad_bump,
                           centered=True)


# The analytic benchmark must have the expected zero cross-curvatures.
def test_black_scholes_analytic_zero_cross_curvatures() -> None:
    vanna,vomma = black_scholes_vanna_vomma(
        spot=100.,strike=100.,rate=.02,volatility=.2,maturity=1.,
    )
    assert abs(vanna) < 1e-13
    assert abs(vomma) < 1e-13


@pytest.mark.parametrize("bad", [0., -1., float("nan")])
def test_analytic_rejects_nonpositive_volatility(bad: float) -> None:
    with pytest.raises(ValueError):
        black_scholes_vanna_vomma(spot=100.,strike=100.,rate=.02,
                                   volatility=bad,maturity=1.)


@pytest.mark.parametrize("delta,side,quantity,action", [
    (.6,"Long",-600,"SELL"),(.6,"Short",600,"BUY"),
    (-.4,"Long",400,"BUY"),(-.4,"Short",-400,"SELL"),
])
def test_delta_hedge_signs(
    delta: float, side: str, quantity: int, action: str
) -> None:
    result = calculate_stock_hedge(delta,side,10,100)
    assert isinstance(result,StockHedge)
    assert result.rounded_trade == quantity
    assert result.instruction == action
    assert result.residual_delta_shares == pytest.approx(0)


# Net trades must account for shares already held in the portfolio.
def test_stock_hedge_net_existing_shares() -> None:
    x = calculate_stock_hedge(-.4,"Long",10,100,existing_shares=150)
    assert x.rounded_trade == 250
    assert x.option_delta_shares == pytest.approx(-400)


def test_stock_hedge_rounding() -> None:
    h = calculate_stock_hedge(.253,"Long",1,10)
    assert h.rounded_trade == -3
    assert h.residual_delta_shares == pytest.approx(-.47)


# The risk kernel remains independently importable from the UI.
def test_risk_core_import_without_frontend() -> None:
    # Kernel modules themselves import only stdlib; frontend import is not required.
    hedge = importlib.import_module("src.risk.hedging")
    maths = importlib.import_module("src.risk.numerics")
    assert not any(name.startswith("frontend.") for name in
                   (hedge.__name__, maths.__name__))


def test_frontend_compatibility_reexports() -> None:
    from frontend.trading_risk import (
        StockHedge as FrontendHedge,
        calculate_stock_hedge as frontend_hedge,
    )
    assert FrontendHedge is StockHedge
    assert frontend_hedge is calculate_stock_hedge
