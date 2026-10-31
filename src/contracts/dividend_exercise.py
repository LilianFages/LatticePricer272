"""American ex-date exercise on a fixed lattice without future information.

An ex-date aligned with a lattice layer is priced using the exact pre/post
cash-jump choice at each child. For a genuine off-grid ex-date, the parent
also compares its lattice continuation with two discounted expected payoffs
for strategies exercising just BEFORE or just AFTER the real cash jump.
Before-ex payoff uses strike K; after-ex payoff uses strike K+D on the
pre-dividend diffusion. Both strategies are nonanticipative: exercise is
precommitted from the parent node. This is NOT an exact stochastic event layer.

At zero volatility the event-time exercise decision is exact, recovering the
independent deterministic American benchmark and stable one-day Theta.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from src.pricing.black_scholes_terms import standardized_terms
from src.contracts.dividend import CashDividend
from src.contracts.dividend_timing import dividend_grid_step, dividend_time_in_step
from src.lattices.node import Node
from src.contracts.option import CallOption, Option, PutOption


class DividendTree(Protocol):
    maturity: float
    nb_steps: int
    dt: float
    rate: float
    volatility: float
    dividends: list[CashDividend]


@dataclass(frozen=True, slots=True)
class DividendStepEvent:
    """A cash jump in a lattice interval, retaining its actual timing."""

    amount: float
    elapsed_before_event: float
    remaining_time: float
    rate: float
    volatility: float


def aligned_event_step(time: float, maturity: float, steps: int) -> int | None:
    """Identify a lattice endpoint up to binary floating-point roundoff."""
    step = dividend_grid_step(time, maturity, steps)
    boundary = step * (maturity / steps)
    tolerance = 8 * max(math.ulp(time), math.ulp(boundary))
    return step if abs(time - boundary) <= tolerance else None


def dividend_exercise_steps(tree: DividendTree) -> dict[int, DividendStepEvent]:
    """Map every ex-date to its step without silently snapping a real date."""
    events: dict[int, DividendStepEvent] = {}
    for dividend in getattr(tree, "dividends", ()):
        step = dividend_grid_step(dividend.time, tree.maturity, tree.nb_steps)
        if step in events:
            raise ValueError("More than one dividend occurs in a lattice step.")
        start = (step - 1) * tree.dt
        end = step * tree.dt
        actual_time = dividend_time_in_step(dividend.time, start, end)
        events[step] = DividendStepEvent(
            amount=dividend.amount,
            elapsed_before_event=max(0.0, actual_time - start),
            remaining_time=max(0.0, end - actual_time),
            rate=tree.rate,
            volatility=tree.volatility,
        )
    return events


def before_dividend_value(
    child: Node,
    post_dividend_value: float,
    option: Option,
    dividend: DividendStepEvent | None
) -> float:
    """Exact pre/post cash-jump choice ONLY when the ex-date is on the grid.

    Applying this choice to an off-grid child would make the option holder
    anticipate future market information, spuriously inflating its value.
    """
    if (
        dividend is None or not option.is_american
        or dividend.remaining_time > 0.0
    ):
        return post_dividend_value
    return max(post_dividend_value, option.payoff(child.price + dividend.amount))


def _exercise_at_ex_date(
    spot: float,
    option: Option,
    rate: float,
    volatility: float,
    time: float,
    *,
    cash_adjustment: float = 0.0
) -> float:
    """Price forced exercise on either side of the actual cash jump.

    Before ex-date, cash_adjustment=0. After ex-date, the stock becomes
    S_plus=S_minus-D and the call/put payoff is equivalent to using K+D
    on the pre-dividend diffusion. Discounting is always to the parent time.
    The decision to exercise is precommitted at the parent and uses no
    post-event market observation.
    """
    effective_strike = option.strike + cash_adjustment
    discount_strike = effective_strike * math.exp(-rate * time)
    if time <= 0.0 or volatility == 0.0:
        if isinstance(option, CallOption):
            return max(spot - discount_strike, 0.0)
        if isinstance(option, PutOption):
            return max(discount_strike - spot, 0.0)
        raise TypeError("Unsupported vanilla option type.")

    # The zero-strike option is a legitimate limiting contract.
    if effective_strike == 0.0:
        if isinstance(option, CallOption):
            return spot
        if isinstance(option, PutOption):
            return 0.0
        raise TypeError("Unsupported vanilla option type.")
    # A precommitted ex-date exercise is a European payoff to that date.
    d1, d2 = standardized_terms(
        spot, effective_strike, rate, volatility, time,
    )
    cdf = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))
    if isinstance(option, CallOption):
        return max(0.0, spot * cdf(d1) - discount_strike * cdf(d2))
    if isinstance(option, PutOption):
        return max(0.0, discount_strike * cdf(-d2) - spot * cdf(-d1))
    raise TypeError("Unsupported vanilla option type.")


def offgrid_exercise_bound(
    parent: Node,
    hold_value: float,
    option: Option,
    dividend: DividendStepEvent | None
) -> float:
    """Add a nonanticipative event-time exercise policy to the parent value.

    The two added strategies force exercise immediately before or immediately
    after the cash jump. With volatility, they do not condition exercise on
    the stock price at that date: neither uses future price information.
    The deterministic event-time decision is exact for sigma=0.
    """
    if (
        dividend is None or not option.is_american
        or dividend.remaining_time == 0.0
    ):
        return hold_value
    # The parent chooses to exercise before the future cash jump.
    pre_event_value = _exercise_at_ex_date(
        parent.price,
        option,
        dividend.rate,
        dividend.volatility,
        dividend.elapsed_before_event,
    )
    # Exercise after the jump is represented by the shifted strike K + D.
    post_event_value = _exercise_at_ex_date(
        parent.price,
        option,
        dividend.rate,
        dividend.volatility,
        dividend.elapsed_before_event,
        cash_adjustment=dividend.amount,
    )
    return max(hold_value, pre_event_value, post_event_value)
