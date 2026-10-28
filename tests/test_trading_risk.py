"""Desk risk: derivatives, curve isolation and signed stock hedge controls."""

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

import frontend.trading_risk as risk
from frontend.greeks_service import GreekBumps
from frontend.pricing_service import PricingRequest


BASE = PricingRequest(
    spot=100.0, strike=100.0, rate=0.02, volatility=0.20,
    maturity=1.0, steps=30, model="Binomial",
)


# Closed-form polynomial: mixed S/volatility terms expose Vanna and Vomma stencil errors.
def polynomial_price(req: PricingRequest) -> float:
    s, v = req.spot, req.volatility
    return 4.0 + 0.5 * s + 3.0 * s * v + 7.0 * v * v + 0.8 * s * v * v


def polynomial_pricer(req: PricingRequest) -> SimpleNamespace:
    return SimpleNamespace(price=polynomial_price(req))


def fake_greeks(request: PricingRequest, bumps: GreekBumps, pricer: object) -> SimpleNamespace:
    h_s = request.spot * bumps.spot_pct / 100.0
    h_v = bumps.vol_pp / 100.0
    initial = polynomial_price(request)
    scenarios = {
        "spot+": polynomial_price(replace(request, spot=request.spot + h_s)),
        "spot-": polynomial_price(replace(request, spot=request.spot - h_s)),
        "volatility+": polynomial_price(replace(request, volatility=request.volatility + h_v)),
        "volatility++": polynomial_price(replace(request, volatility=request.volatility + 2 * h_v)),
    }
    if request.volatility >= h_v:
        scenarios["volatility-"] = polynomial_price(
            replace(request, volatility=request.volatility - h_v)
        )
    return SimpleNamespace(
        price=initial, bumps=bumps, scenario_prices=scenarios,
        measured=lambda: {"Delta": 0.6, "Gamma": 0.03,
                          "Vega / 1pp": 0.12, "Theta / 1d": -0.02, "Rho / 1bp": 0.001},
        analytic=None, issues=(),
    )


@pytest.fixture
def synthetic(monkeypatch: pytest.MonkeyPatch) -> risk.DeskRiskSnapshot:
    monkeypatch.setattr(risk, "calculate_greeks", fake_greeks)
    return risk.compute_desk_risk(BASE, pricer=polynomial_pricer)


# The table must retain a stable risk-factor order across all seven Greeks.
def test_greek_order() -> None:
    assert risk.GREEK_NAMES == ("Delta", "Gamma", "Vega", "Vanna", "Vomma", "Theta", "Rho")


def test_vanna_polynomial_centered(synthetic: risk.DeskRiskSnapshot) -> None:
    assert synthetic.vanna_1pp == pytest.approx((3 + 1.6 * 0.2) * 0.01, abs=1e-10)


def test_vomma_polynomial_centered(synthetic: risk.DeskRiskSnapshot) -> None:
    assert synthetic.vomma_1pp2 == pytest.approx((14 + 1.6 * 100) * 1e-4, abs=1e-10)


def test_extra_pricing_count_centered(synthetic: risk.DeskRiskSnapshot) -> None:
    assert synthetic.extra_evaluations == 4


# Near zero volatility, the risk routine must switch to forward shocks.
def test_volatility_forward_vanna_vomma(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(risk, "calculate_greeks", fake_greeks)
    req = replace(BASE, volatility=0.0001)
    result = risk.compute_desk_risk(req, pricer=polynomial_pricer)
    expected_cross_forward = 3.0 + 0.8 * (2 * req.volatility + 0.005)
    assert result.vanna_1pp == pytest.approx(expected_cross_forward * 0.01)
    assert result.vomma_1pp2 == pytest.approx((14 + 1.6 * req.spot) * 0.0001)
    assert result.extra_evaluations == 2
    assert len(result.extra_issues) == 2


def test_existing_delta_and_rho_preserved(synthetic: risk.DeskRiskSnapshot) -> None:
    assert synthetic.value("Delta") == 0.6
    assert synthetic.value("Rho") == 0.001


def test_extra_greek_value(synthetic: risk.DeskRiskSnapshot) -> None:
    assert synthetic.value("Vanna") == synthetic.vanna_1pp
    assert synthetic.value("Vomma") == synthetic.vomma_1pp2


def test_unknown_greek_rejected(synthetic: risk.DeskRiskSnapshot) -> None:
    with pytest.raises(ValueError, match="Unsupported Greek"):
        synthetic.value("Charm")


# Analytical references exist only for the mathematically eligible cases.
def test_greek_table_has_seven_rows_and_units(synthetic: risk.DeskRiskSnapshot) -> None:
    table = risk.greek_table(synthetic)
    assert len(table) == 7
    assert [row["Greek"] for row in table] == list(risk.GREEK_NAMES)
    assert all(row["Unit"] for row in table)
    assert all(
        row["Black-Scholes"] is None for row in table
        if row["Greek"] not in ("Vanna", "Vomma")
    )
    assert all(
        row["Black-Scholes"] is not None for row in table
        if row["Greek"] in ("Vanna", "Vomma")
    )


def test_bs_vanna_vomma_at_positive_vol() -> None:
    va, vo = risk._bs_second_order(BASE)
    assert math.isfinite(va) and math.isfinite(vo)


@pytest.mark.parametrize("exercise,dividend,vol,strike", [
    ("American", (), 0.20, 100.0),
    ("European", (SimpleNamespace(time=0.5, amount=1.0),), 0.20, 100.0),
    ("European", (), 0.0, 100.0),
    ("European", (), 0.20, 0.0),
])
def test_bs_second_order_excludes_nonstandard_cases(exercise: str, dividend: tuple, vol: float, strike: float) -> None:
    req = replace(BASE, exercise=exercise, dividends=dividend, volatility=vol, strike=strike)
    assert risk._bs_second_order(req) == (None, None)


# Selecting the Greek must never trigger hidden batches of repricing.
def test_greek_curve_requires_explicit_call(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = {"n": 0}

    def counted(request, bumps, pricer):
        counter["n"] += 1
        return fake_greeks(request, bumps, pricer)

    monkeypatch.setattr(risk, "calculate_greeks", counted)
    assert counter["n"] == 0
    points = risk.compute_greek_curve(BASE, "Delta", count=5, pricer=polynomial_pricer)
    assert len(points) == 5
    assert counter["n"] == 5
    assert [p.spot for p in points] == pytest.approx([90, 95, 100, 105, 110])
    assert all(p.value == 0.6 and p.error is None for p in points)


def test_greek_curve_vomma_runs_full_extra_bumps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(risk, "calculate_greeks", fake_greeks)
    points = risk.compute_greek_curve(BASE, "Vomma", count=3, pricer=polynomial_pricer)
    assert [p.value for p in points] == pytest.approx(
        [(14 + 1.6 * s) * 1e-4 for s in (90, 100, 110)]
    )


# Input validation rejects unsupported point grids and span ranges.
@pytest.mark.parametrize("count", [0, 2, 4, 8, True])
def test_invalid_curve_point_count(count: object) -> None:
    with pytest.raises(ValueError):
        risk.compute_greek_curve(BASE, "Delta", count=count, pricer=polynomial_pricer)


@pytest.mark.parametrize("span", [-1, 0, 31, float("nan")])
def test_invalid_curve_width(span: float) -> None:
    with pytest.raises(ValueError):
        risk.compute_greek_curve(BASE, "Delta", span_pct=span, pricer=polynomial_pricer)


def test_invalid_curve_greek() -> None:
    with pytest.raises(ValueError):
        risk.compute_greek_curve(BASE, "Charm", pricer=polynomial_pricer)


def test_curve_failure_is_not_zero_filled(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_on_left(request, bumps, pricer):
        if request.spot < 100:
            raise ValueError("synthetic failure")
        return fake_greeks(request, bumps, pricer)

    monkeypatch.setattr(risk, "calculate_greeks", fail_on_left)
    points = risk.compute_greek_curve(BASE, "Delta", pricer=polynomial_pricer)
    assert points[0].value is None
    assert "synthetic failure" in points[0].error
    assert points[-1].value == 0.6


# Stock trades offset signed option delta; long/short are not interchangeable.
@pytest.mark.parametrize("delta,side,expected", [
    (0.6, "Long", -600),
    (0.6, "Short", 600),
    (-0.4, "Long", 400),
    (-0.4, "Short", -400),
])
def test_delta_hedge_four_position_signs(delta: float, side: str, expected: int) -> None:
    h = risk.calculate_stock_hedge(delta, side, contracts=10, multiplier=100)
    assert h.rounded_trade == expected
    assert h.instruction == ("BUY" if expected > 0 else "SELL")
    assert h.residual_delta_shares == pytest.approx(0)


def test_existing_long_shares_reduce_buy_hedge() -> None:
    h = risk.calculate_stock_hedge(-0.4, "Long", 10, 100, existing_shares=150)
    assert h.rounded_trade == 250
    assert h.target_shares == pytest.approx(400)


def test_existing_short_shares_reduce_sell_hedge() -> None:
    h = risk.calculate_stock_hedge(0.6, "Long", 10, 100, existing_shares=-200)
    assert h.rounded_trade == -400


def test_existing_stock_fully_hedged_results_hold() -> None:
    h = risk.calculate_stock_hedge(0.5, "Long", 1, 100, existing_shares=-50)
    assert h.rounded_trade == 0
    assert h.instruction == "HOLD"


# Whole-share rounding must retain the unrounded residual risk.
def test_small_trade_rounded_and_residual() -> None:
    h = risk.calculate_stock_hedge(0.253, "Long", 1, 10)
    assert h.exact_trade == pytest.approx(-2.53)
    assert h.rounded_trade == -3
    assert h.residual_delta_shares == pytest.approx(-0.47)


def test_round_half_away_from_zero() -> None:
    assert risk.calculate_stock_hedge(0.25, "Long", 1, 10).rounded_trade == -3
    assert risk.calculate_stock_hedge(-0.25, "Long", 1, 10).rounded_trade == 3


# Reject invalid position sizing instead of silently coercing quantities.
@pytest.mark.parametrize("side,contracts,multiplier,shares", [
    ("Buy", 1, 100, 0), ("Long", 0, 100, 0),
    ("Long", 1.0, 100, 0), ("Long", True, 100, 0),
    ("Long", 1, 0, 0), ("Long", 1, 100.0, 0),
    ("Long", 1, 100, float("inf")),
])
def test_hedge_rejects_invalid_position_inputs(side: object, contracts: object, multiplier: object, shares: object) -> None:
    with pytest.raises(ValueError):
        risk.calculate_stock_hedge(0.5, side, contracts, multiplier, shares)


def test_hedge_rejects_nonfinite_delta() -> None:
    with pytest.raises(ValueError):
        risk.calculate_stock_hedge(float("nan"), "Long", 10)


def test_hedge_is_independent_of_option_call_put() -> None:
    # Same signed Delta maps to the same hedge; option type is not the sign switch.
    long = risk.calculate_stock_hedge(-0.4, "Long", 2, 100)
    short = risk.calculate_stock_hedge(-0.4, "Short", 2, 100)
    assert long.rounded_trade == 80
    assert short.rounded_trade == -80

# LOT_5C3_GUARDED_REFACTOR
