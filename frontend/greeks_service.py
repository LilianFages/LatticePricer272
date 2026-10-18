"""Greek risk measures from independent repricings of existing lattice engines.

Conventions:
    Delta = dV/dS (EUR premium per EUR of spot)
    Gamma = d²V/dS² (premium per EUR² of spot)
    Vega = 0.01*dV/dsigma (premium for +1 volatility percentage point)
    Rho = 1e-4*dV/dr (premium for +1 basis point)
    Theta = one-calendar-day roll-down P&L (premium change at unchanged spot)

Bumps are used for DIFFERENTIATION and may differ from reporting units.
The pricer is never approximated/reimplemented in this module.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, replace
from typing import Callable

from frontend.pricing_service import (
    DividendInput,
    PricingInputError,
    PricingRequest,
    price_contract,
    validate_request,
)

DAY = 1.0 / 365.0
BP = 0.0001
VOL_POINT = 0.01


@dataclass(frozen=True, slots=True)
class GreekBumps:
    spot_pct: float = 1.0   # percent of initial spot
    vol_pp: float = 0.5     # percentage points of volatility
    rate_bp: float = 10.0   # basis points of continuously compounded rate
    gamma_spot_pct: float | None = None  # None = grid-aware automatic stencil

    def validate(self) -> None:
        if not all(math.isfinite(x) for x in (self.spot_pct, self.vol_pp, self.rate_bp)):
            raise PricingInputError("Greek perturbations must be finite numbers.")
        if not 0 < self.spot_pct <= 20:
            raise PricingInputError("Spot perturbation must be in (0, 20]%.")
        if not 0 < self.vol_pp <= 10:
            raise PricingInputError("Volatility perturbation must be in (0, 10] pp.")
        if not 0 < self.rate_bp <= 100:
            raise PricingInputError("Rate perturbation must be in (0, 100] bp.")
        if self.gamma_spot_pct is not None and (
            not math.isfinite(self.gamma_spot_pct)
            or not 0 < self.gamma_spot_pct <= 20
        ):
            raise PricingInputError("Gamma spot perturbation must be in (0, 20]%.")


@dataclass(frozen=True, slots=True)
class GreeksResult:
    request: PricingRequest
    bumps: GreekBumps
    price: float
    delta: float | None
    gamma: float | None
    vega_1pp: float | None
    theta_1day: float | None
    rho_1bp: float | None
    analytic: dict[str, float] | None
    scenario_prices: dict[str, float]
    issues: tuple[str, ...]
    evaluations: int
    total_runtime_ms: float
    volatility_scheme: str
    gamma_spot_bump_pct: float = 0.0
    gamma_scheme: str = "central"

    def measured(self) -> dict[str, float | None]:
        return {
            "Delta": self.delta,
            "Gamma": self.gamma,
            "Vega / 1pp": self.vega_1pp,
            "Theta / 1d": self.theta_1day,
            "Rho / 1bp": self.rho_1bp,
        }


def analytic_bs_greeks(request: PricingRequest) -> dict[str, float] | None:
    """Black--Scholes Greeks, restricted to smooth European, no-cash-dividend cases.

    Independent diagnostic/reference; not the engine used for finite differences.
    """
    if (request.exercise != "European" or request.dividends
            or request.volatility <= 0 or request.strike <= 0):
        return None
    s, k, r, sigma, t = (
        request.spot, request.strike, request.rate, request.volatility,
        request.maturity,
    )
    root_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r + sigma * sigma / 2) * t) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    normal_cdf = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))
    density = math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi)
    disc_strike = k * math.exp(-r * t)
    if request.option_type == "Call":
        delta = normal_cdf(d1)
        theta_y = -s * density * sigma / (2 * root_t) - r * disc_strike * normal_cdf(d2)
        rho_y = disc_strike * t * normal_cdf(d2)
    else:
        delta = normal_cdf(d1) - 1.0
        theta_y = -s * density * sigma / (2 * root_t) + r * disc_strike * normal_cdf(-d2)
        rho_y = -disc_strike * t * normal_cdf(-d2)
    return {
        "Delta": delta,
        "Gamma": density / (s * sigma * root_t),
        "Vega / 1pp": VOL_POINT * s * density * root_t,
        "Theta / 1d": theta_y * DAY,
        "Rho / 1bp": rho_y * BP,
    }


def roll_calendar_day(request: PricingRequest) -> PricingRequest | None:
    """Roll observation time by one day with the fixed *calendar* ex-dates.

    Holding the current ex-dividend spot fixed across an ex-date would not be
    an interpretable theta scenario. Return None if an event falls in the day.
    """
    if request.maturity <= DAY:
        return None
    if any(div.time <= DAY for div in request.dividends):
        return None
    return replace(
        request,
        maturity=request.maturity - DAY,
        dividends=tuple(
            DividendInput(time=div.time - DAY, amount=div.amount)
            for div in request.dividends
        ),
    )



def select_gamma_bump(request: PricingRequest, bumps: GreekBumps) -> tuple[float, str, bool]:
    """Return a Gamma stencil width independent of the Delta bump.

    An option value on a fixed finite lattice is locally affine in spot until
    a terminal node crosses the payoff kink. A too-small Gamma bump can thus
    produce a spurious zero. Scale the default width with one *terminal*
    spacing of the chosen lattice, without using the Black--Scholes answer.

    This is a numerical heuristic, not a guarantee of convergence. Dividend
    transitions can change effective spacing. The 20% limit prevents an
    excessively large shock that would alter the contract's risk interpretation.
    """
    if bumps.gamma_spot_pct is not None:
        return bumps.gamma_spot_pct, "manual", False

    sigma = request.volatility
    dt = request.maturity / request.steps
    if request.model.startswith("Trinomial"):
        # Trinomial geometric grid: alpha = exp(sigma*sqrt(3*dt)).
        log_step = sigma * math.sqrt(3.0 * dt)
    else:
        # Binomial terminal levels differ by approximately exp(2*sigma*sqrt(dt)).
        # Hybrid uses binomial steps outside dividend dates; the approximation
        # is only a guide if cash dividends are present.
        log_step = 2.0 * sigma * math.sqrt(dt)

    upper = math.log1p(0.20)
    clipped = log_step > upper
    width_pct = 100.0 * math.expm1(min(log_step, upper))
    return max(bumps.spot_pct, width_pct), "grid-aware", clipped


def calculate_greeks(
    request: PricingRequest,
    bumps: GreekBumps | None = None,
    pricer: Callable = price_contract,
) -> GreeksResult:
    """Calculate independent finite-difference lattice repricings.

    A failed shock does not discard all other Greeks; affected fields are None.
    Base price failure is propagated to the user. Timings include only pricer
    calls (no chart rendering), not separate calibration or volatility fitting.
    """
    validate_request(request)
    bumps = bumps or GreekBumps()
    bumps.validate()

    prices: dict[str, float] = {}
    issues: list[str] = []
    n_evaluations = 0
    runtime_ms = 0.0

    def eval_shock(name: str, shocked: PricingRequest) -> float | None:
        nonlocal n_evaluations, runtime_ms
        n_evaluations += 1
        try:
            result = pricer(shocked)
            value = float(result.price)
            if not math.isfinite(value):
                raise ArithmeticError("non-finite price")
            runtime_ms += float(getattr(result, "elapsed_ms", 0.0))
            prices[name] = value
            return value
        except (PricingInputError, ValueError, RuntimeError,
                ArithmeticError, OverflowError, ZeroDivisionError) as exc:
            issues.append(f"{name}: {type(exc).__name__}: {exc}")
            return None

    base = eval_shock("base", request)
    if base is None:
        raise PricingInputError("Base contract cannot be priced: " + issues[-1])

    def central_sides(field: str, h: float) -> tuple[float | None, float | None]:
        x = getattr(request, field)
        minus = eval_shock(f"{field}-", replace(request, **{field: x-h}))
        plus = eval_shock(f"{field}+", replace(request, **{field: x+h}))
        return minus, plus

    h_s = request.spot * bumps.spot_pct / 100.0
    s_minus, s_plus = central_sides("spot", h_s)
    delta = ((s_plus - s_minus) / (2 * h_s)
             if s_plus is not None and s_minus is not None else None)

    gamma_pct, gamma_scheme, clipped_gamma = select_gamma_bump(request, bumps)
    h_gamma = request.spot * gamma_pct / 100.0
    if math.isclose(h_gamma, h_s, rel_tol=1e-12):
        gamma_minus, gamma_plus = s_minus, s_plus
    else:
        gamma_minus = eval_shock(
            "spot_gamma-", replace(request, spot=request.spot-h_gamma)
        )
        gamma_plus = eval_shock(
            "spot_gamma+", replace(request, spot=request.spot+h_gamma)
        )
    gamma = ((gamma_plus - 2 * base + gamma_minus) / (h_gamma * h_gamma)
             if gamma_plus is not None and gamma_minus is not None else None)
    if clipped_gamma:
        issues.append(
            "Gamma / grid: suggested spacing exceeds the 20% spot bump cap; "
            "Gamma may be unreliable in this coarse or high-volatility lattice."
        )
    if request.dividends:
        issues.append(
            "Gamma / dividend: grid-aware bump is a spacing heuristic only; "
            "check sensitivity to bump and N near cash dividend dates."
        )
    reference_greeks = analytic_bs_greeks(request)
    if reference_greeks is not None and gamma is not None:
        reference_gamma = reference_greeks["Gamma"]
        if reference_gamma > 1e-5 and abs(gamma) < 0.05 * reference_gamma:
            issues.append(
                "Gamma / grid: near-zero curvature despite positive Black--Scholes "
                "Gamma. Try a larger Gamma bump or more steps."
            )

    h_vol = bumps.vol_pp / 100.0
    if request.volatility >= h_vol:
        vol_minus, vol_plus = central_sides("volatility", h_vol)
        vega = (VOL_POINT * (vol_plus-vol_minus) / (2*h_vol)
                if vol_plus is not None and vol_minus is not None else None)
        vol_scheme = "central"
    else:
        # A negative volatility is invalid. Use a second-order one-sided
        # derivative rather than implicitly clipping sigma to zero.
        p1 = eval_shock("volatility+", replace(request, volatility=request.volatility+h_vol))
        p2 = eval_shock("volatility++", replace(request, volatility=request.volatility+2*h_vol))
        vega = (VOL_POINT * (-3*base + 4*p1 - p2)/(2*h_vol)
                if p1 is not None and p2 is not None else None)
        vol_scheme = "forward 2nd-order (near zero vol)"

    h_r = bumps.rate_bp * BP
    r_minus, r_plus = central_sides("rate", h_r)
    rho = (BP * (r_plus-r_minus)/(2*h_r)
           if r_minus is not None and r_plus is not None else None)

    rolled = roll_calendar_day(request)
    if rolled is None:
        theta = None
        issues.append("Theta / 1d: unavailable within one day of expiry or a cash ex-date.")
    else:
        next_price = eval_shock("calendar+1d", rolled)
        theta = next_price - base if next_price is not None else None

    return GreeksResult(
        request=request, bumps=bumps, price=base,
        delta=delta, gamma=gamma, vega_1pp=vega,
        theta_1day=theta, rho_1bp=rho,
        analytic=reference_greeks,
        scenario_prices=prices, issues=tuple(issues),
        evaluations=n_evaluations, total_runtime_ms=runtime_ms,
        volatility_scheme=vol_scheme,
        gamma_spot_bump_pct=gamma_pct,
        gamma_scheme=gamma_scheme,
    )


def greeks_csv(result: GreeksResult) -> str:
    """CSV with one line per Greek and transparent input/perturbation metadata."""
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["section", "field", "value", "analytic_BS", "model_minus_BS"])
    req, b = result.request, result.bumps
    fields = {
        "model": req.model, "option": req.option_type, "exercise": req.exercise,
        "spot": req.spot, "strike": req.strike, "volatility": req.volatility,
        "rate": req.rate, "maturity": req.maturity, "steps": req.steps,
        "pruning_threshold": req.pruning_threshold,
        "dividends": ";".join(f"{d.time}:{d.amount}" for d in req.dividends),
        "spot_bump_pct": b.spot_pct, "gamma_bump_pct_used": result.gamma_spot_bump_pct,
        "gamma_scheme": result.gamma_scheme, "vol_bump_pp": b.vol_pp,
        "rate_bump_bp": b.rate_bp, "theta_roll_days": 1,
        "vol_scheme": result.volatility_scheme,
        "evaluations": result.evaluations,
        "total_runtime_ms": result.total_runtime_ms,
        "base_price": result.price,
    }
    for key, value in fields.items():
        writer.writerow(["contract", key, value, "", ""])
    for label, estimate in result.measured().items():
        bs = result.analytic.get(label) if result.analytic else None
        writer.writerow(["greek", label,
                         "" if estimate is None else estimate,
                         "" if bs is None else bs,
                         "" if estimate is None or bs is None else estimate-bs])
    for label, price in result.scenario_prices.items():
        writer.writerow(["shock_price", label, price, "", ""])
    for problem in result.issues:
        writer.writerow(["warning", "note", problem, "", ""])
    return stream.getvalue()
