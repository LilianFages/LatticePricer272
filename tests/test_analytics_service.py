"""Quant Analytics 3.2 tests: deterministic pricing stub, no outside network."""
import csv
import io
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from frontend.analytics_service import (
    convergence_study, greeks_stability, observations_csv,
    sensitivity_sweep, stability_csv,
)
from frontend.pricing_service import DividendInput, PricingInputError, PricingRequest


# The deterministic quadratic surrogate separates analytics checks from market engines.
def _fake_pricer(request: PricingRequest) -> SimpleNamespace:
    if request.spot < 85:
        raise RuntimeError("Synthetic spot out of supported range")
    if request.dividends and request.maturity < max(d.time for d in request.dividends):
        raise PricingInputError("Dividend after maturity")
    if request.dividends and request.model == "Binomial":
        raise PricingInputError("Binomial cannot price dividends")
    price = (0.0002*request.spot**2 + 5*request.volatility
             + 10*request.maturity + 3*request.rate + 1/(request.steps+1))
    bs = 8.916 if request.exercise == "European" and not request.dividends else None
    return SimpleNamespace(price=price, elapsed_ms=request.steps/10,
                           black_scholes=bs)


def _request(**values: Any) -> PricingRequest:
    return replace(PricingRequest(), **values)


# Check scenario cardinality and symmetry around the base market input.
def test_sensitivity_counts_and_center():
    rows = sensitivity_sweep(_request(), "spot", ("Binomial", "Hybrid"),
                             count=5, amplitude=0.1, pricer=_fake_pricer)
    assert len(rows) == 10
    assert [p.coordinate for p in rows[:5]] == pytest.approx([90,95,100,105,110])
    assert all(p.status == "ok" and p.price is not None for p in rows)


# Pricing failure must produce missing values rather than fabricated zero prices.
def test_sensitivity_isolated_failure_and_no_false_zero():
    rows = sensitivity_sweep(_request(), "spot", ("Binomial",), count=5,
                             amplitude=0.2, pricer=_fake_pricer)
    assert rows[0].price is None
    assert rows[0].runtime_ms is None
    assert "Synthetic spot" in rows[0].status
    assert rows[-1].price is not None


# Perturbations must respect the user-facing units of model parameters.
def test_sensitivity_volatility_and_rate_units():
    vol = sensitivity_sweep(_request(), "volatility", ("Hybrid",), count=3,
                            amplitude=0.05, pricer=_fake_pricer)
    assert [p.coordinate for p in vol] == pytest.approx([0.15,0.20,0.25])
    rate = sensitivity_sweep(_request(), "rate", ("Hybrid",), count=3,
                             amplitude=0.02, pricer=_fake_pricer)
    assert [p.coordinate for p in rate] == pytest.approx([0.,0.02,0.04])


# Invalid dividend timing must remain an explicit scenario failure.
def test_sensitivity_dividend_maturity_failure():
    request = _request(dividends=(DividendInput(time=0.95,amount=1),), model="Hybrid")
    rows = sensitivity_sweep(request,"maturity",("Hybrid",),count=3,
                             amplitude=0.25,pricer=_fake_pricer)
    assert rows[0].price is None
    assert rows[1].price is not None


# Validate the sweep grid before creating observations.
@pytest.mark.parametrize("parameter,models,count,amplitude", [
    ("wrong", ("Hybrid",),5,None), ("spot",(),5,None),
    ("spot",("Hybrid","Hybrid"),5,None),
    ("spot",("Hybrid",),6,None), ("spot",("Hybrid",),5,0),
    ("spot",("Hybrid",),5,100),
])
def test_sensitivity_rejects_invalid_inputs(parameter,models,count,amplitude):
    with pytest.raises(PricingInputError):
        sensitivity_sweep(_request(),parameter,models,count,amplitude,pricer=_fake_pricer)


# Use the analytical reference only for eligible European contracts.
def test_convergence_analytic_reference():
    result = convergence_study(_request(), ("Binomial","Hybrid"), (20,40,80),
                                pricer=_fake_pricer)
    assert result.reference_type == "analytic"
    assert result.reference_label == "Black–Scholes"
    assert result.reference_price == pytest.approx(8.916)
    assert len(result.points) == 6


# A numerical reference is defined by the selected largest grid.
def test_convergence_numeric_reference_at_max_n():
    request=_request(exercise="American", model="Hybrid")
    result=convergence_study(request,("Binomial","Hybrid"),(20,40),
                              numerical_reference_model="Hybrid",pricer=_fake_pricer)
    assert result.reference_type == "numerical"
    assert result.reference_price == pytest.approx(
        _fake_pricer(replace(request,steps=40)).price)
    assert result.reference_label.endswith("N=40")


# Numerical reference selection must not depend on display ordering.
def test_convergence_numeric_unselected_ref_independent():
    result=convergence_study(_request(exercise="American"),("Binomial",),(20,40),
                              numerical_reference_model="Hybrid",pricer=_fake_pricer)
    assert result.reference_type == "numerical"
    assert len(result.points) == 2


# Failed points must not invalidate otherwise successful convergence rows.
def test_convergence_partial_failure():
    request = _request(exercise="American", dividends=(DividendInput(0.5,3),),model="Hybrid")
    out=convergence_study(request,("Binomial","Hybrid"),(20,40),
                           numerical_reference_model="Hybrid",pricer=_fake_pricer)
    assert out.points[0].price is None
    assert out.points[2].price is not None
    assert out.reference_type == "numerical"


# Never infer a reference from an unavailable computed value.
def test_convergence_unavailable_numeric_reference():
    request=_request(exercise="American",dividends=(DividendInput(0.5,3),),model="Hybrid")
    out=convergence_study(request,("Hybrid",),(20,40),
                           numerical_reference_model="Binomial",pricer=_fake_pricer)
    assert out.reference_type == "unavailable"
    assert out.reference_price is None


# Reject discretization grids inconsistent with the study protocol.
@pytest.mark.parametrize("grid",[(20,), (20,20),(40,20),(20,1000)])
def test_convergence_grid_validation(grid):
    with pytest.raises(PricingInputError):
        convergence_study(_request(),("Hybrid",),grid,pricer=_fake_pricer)


# The quadratic proxy provides exact finite-difference derivative identities.
def test_greek_stability_exact_quadratic_derivatives():
    rows = greeks_stability(_request(),(1.,3.5,5.),(40,100),pricer=_fake_pricer)
    assert len(rows)==5
    for row in rows:
        assert row.status=="ok"
        assert row.delta==pytest.approx(0.04, rel=1e-7)
        assert row.gamma==pytest.approx(0.0004, rel=1e-7)
    assert rows[-1].gamma_bump_pct>0


# One unavailable shock should retain its failure status.
def test_greek_stability_records_failed_shocks():
    rows = greeks_stability(_request(spot=90),(5.,10.),(40,100),
                            gamma_bump_pct=10,pricer=_fake_pricer)
    assert any(x.status!="ok" for x in rows)
    assert any(x.gamma is None for x in rows)


# The Greek sensitivity grid must be numerically admissible.
def test_greek_stability_bad_grid():
    with pytest.raises(PricingInputError):
        greeks_stability(_request(),(2.,1.),(40,100),pricer=_fake_pricer)


# CSV missing values must not be serialized as financial zeros.
def test_csv_blank_failures_no_reinterpretation():
    rows=sensitivity_sweep(_request(),"spot",("Binomial",),count=5,
                           amplitude=.20,pricer=_fake_pricer)
    records=list(csv.DictReader(io.StringIO(observations_csv(rows))))
    assert records[0]["price"] == ""
    assert records[0]["status"] != "ok"
    greek=greeks_stability(_request(),(1.,3.5),(40,100),pricer=_fake_pricer)
    records=list(csv.DictReader(io.StringIO(stability_csv(greek))))
    assert len(records)==4 and "gamma_bump_pct" in records[0]


# Model-specific limitations must not block independent supported models.
def test_unsupported_first_model_does_not_block_supported_dividend_models():
    # The selected Binomial is unsupported, but the trinomial must still run.
    request=_request(model="Binomial", dividends=(DividendInput(0.5,3),))
    prices=sensitivity_sweep(request,"spot",("Binomial","Hybrid"),
                              count=3,amplitude=0.05,pricer=_fake_pricer)
    assert all(x.price is None for x in prices[:3])
    assert all(x.price is not None for x in prices[3:])
    conv=convergence_study(request,("Binomial","Hybrid"),(20,40),
                           numerical_reference_model="Hybrid",pricer=_fake_pricer)
    assert conv.reference_type=="numerical"
    assert conv.points[0].price is None and conv.points[2].price is not None
