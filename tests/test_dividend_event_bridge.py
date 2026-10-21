"""D4.3c regression: event-time cash exercise, stability and engine parity."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from frontend.greeks_service import roll_calendar_day
from frontend.pricing_service import DividendInput, PricingRequest, price_contract
from frontend.trading_risk import compute_desk_risk
from src.dividend_exercise import (
    DividendStepEvent,
    offgrid_exercise_bound,
    aligned_event_step,
    before_dividend_value,
    dividend_exercise_steps,
)
from src.dividend import CashDividend
from src.node import Node
from src.option import CallOption, PutOption
from src.black_scholes import BlackScholesPricer
from src.risk.dividend_events import (
    assess_dividend_exercise,
    deterministic_american_reference,
)
from src.trinomial_tree import TrinomialTree
from src.hybrid_tree import HybridTree

MODELS = (
    "Hybrid", "Trinomial · Full", "Trinomial · Branch pruning",
    "Trinomial · Allocation pruning",
)
ALGORITHMS = ("Recursive", "Backward")


def make_request(*, ex_date: float = 0.6, **kwargs) -> PricingRequest:
    """Deterministic American call with exercise at the actual ex-date."""
    basic = PricingRequest(
        spot=120.0, strike=100.0, rate=0.05, volatility=0.0,
        maturity=1.0, steps=4, model="Trinomial · Full",
        option_type="Call", exercise="American",
        dividends=(DividendInput(time=ex_date, amount=20.0),),
    )
    return replace(basic, **kwargs)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("ex_date", (0.26, 0.49, 0.50, 0.51, 0.60, 0.74))
def test_deterministic_offgrid_cash_exercise_matches_independent_reference(
    model: str, algorithm: str, ex_date: float,
) -> None:
    request = make_request(ex_date=ex_date, model=model)
    exact, event = deterministic_american_reference(request)
    assert event == "pre-dividend"
    actual = price_contract(request, algorithm=algorithm).price
    assert actual == pytest.approx(exact, abs=1e-8, rel=1e-10)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_offgrid_stochastic_american_is_not_below_european(
    model: str, algorithm: str,
) -> None:
    # 0.60 with N=20 is *on* the grid (step 12); use a true off-grid date.
    request = make_request(ex_date=0.61, model=model, volatility=0.20, steps=20)
    american = price_contract(request, algorithm=algorithm).price
    european = price_contract(
        replace(request, exercise="European"), algorithm=algorithm,
    ).price
    assert american >= european - 1e-9
    warning = assess_dividend_exercise(request, american).warning
    assert warning is not None and "nonanticipative" in warning


@pytest.mark.parametrize("model", MODELS)
def test_offgrid_american_recursive_backward_parity(model: str) -> None:
    request = make_request(model=model, volatility=0.18, steps=16)
    a = price_contract(request, algorithm="Recursive").price
    b = price_contract(request, algorithm="Backward").price
    assert a == pytest.approx(b, abs=1e-10, rel=1e-11)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_calendar_one_day_theta_no_longer_jumps_at_offgrid_event(algorithm: str) -> None:
    ex_date = 182 / 365
    request = make_request(
        ex_date=ex_date, model="Hybrid", maturity=364 / 365, steps=4,
    )
    shifted = roll_calendar_day(request)
    assert shifted is not None
    assert aligned_event_step(ex_date, request.maturity, request.steps) == 2
    assert aligned_event_step(shifted.dividends[0].time, shifted.maturity, 4) is None

    initial = price_contract(request, algorithm=algorithm).price
    later = price_contract(shifted, algorithm=algorithm).price
    exact_initial, _ = deterministic_american_reference(request)
    exact_later, _ = deterministic_american_reference(shifted)
    assert initial == pytest.approx(exact_initial, abs=1e-9)
    assert later == pytest.approx(exact_later, abs=1e-9)
    assert later - initial == pytest.approx(exact_later - exact_initial, abs=1e-9)
    assert abs(later - initial) < 0.05


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_full_desk_risk_theta_matches_deterministic_reference(algorithm: str) -> None:
    from functools import partial
    request = make_request(
        ex_date=182 / 365, model="Hybrid", maturity=364 / 365,
    )
    snap = compute_desk_risk(
        request, pricer=partial(price_contract, algorithm=algorithm),
    )
    rolled = roll_calendar_day(request)
    assert rolled is not None
    exact_initial, _ = deterministic_american_reference(request)
    exact_later, _ = deterministic_american_reference(rolled)
    assert snap.value("Theta") == pytest.approx(exact_later-exact_initial, abs=1e-7)
    assert abs(snap.value("Theta")) < 0.05


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("kind, strike", (("Call", 100.0), ("Put", 160.0)))
def test_two_offgrid_dividends_match_deterministic_reference(
    model: str, algorithm: str, kind: str, strike: float,
) -> None:
    request = make_request(
        model=model, option_type=kind, strike=strike, spot=130.0,
        steps=8, dividends=(
            DividendInput(time=0.30, amount=7.0),
            DividendInput(time=0.70, amount=13.0),
        ),
    )
    exact, _ = deterministic_american_reference(request)
    actual = price_contract(request, algorithm=algorithm).price
    assert actual == pytest.approx(exact, abs=1e-8, rel=1e-10)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_dividend_at_expiry_preserves_pre_jump_exercise(
    model: str, algorithm: str,
) -> None:
    request = make_request(ex_date=1.0, model=model)
    exact, event = deterministic_american_reference(request)
    assert event == "pre-dividend"
    actual = price_contract(request, algorithm=algorithm).price
    assert actual == pytest.approx(exact, abs=1e-8, rel=1e-10)


def test_event_mapping_keeps_actual_time_without_adding_extra_dividends() -> None:
    for tree_class in (HybridTree, TrinomialTree):
        tree = tree_class(
            spot=120.0, rate=0.05, volatility=0.2, maturity=1.0,
            nb_steps=4, dividends=[CashDividend(time=0.6, amount=20.0)],
        )
        steps = dividend_exercise_steps(tree)
        assert tuple(steps) == (3,)
        assert steps[3].remaining_time == pytest.approx(0.15)
        assert steps[3].amount == 20.0
        assert steps[3].elapsed_before_event == pytest.approx(0.10)


def test_event_bridge_is_identity_for_european_or_no_cash() -> None:
    child = Node(100.0)
    event = DividendStepEvent(amount=15.0, elapsed_before_event=0.1, remaining_time=0.15, rate=0.05, volatility=0.2)
    european = CallOption(strike=100.0, is_american=False)
    american = CallOption(strike=100.0, is_american=True)
    assert before_dividend_value(child, 9.1, european, event) == 9.1
    assert before_dividend_value(child, 9.1, american, None) == 9.1


def test_nonanticipative_exercise_uses_parent_spot_not_future_child() -> None:
    parent = Node(120.0)
    option = CallOption(strike=100.0, is_american=True)
    event = DividendStepEvent(
        amount=20.0, elapsed_before_event=0.10,
        remaining_time=0.15, rate=0.05, volatility=0.0,
    )
    bound = offgrid_exercise_bound(parent, 10.0, option, event)
    exact = 120.0 - 100.0 * math.exp(-0.05 * 0.10)
    assert bound == pytest.approx(exact)
    future_node = Node(200.0)
    assert before_dividend_value(future_node, 10.0, option, event) == 10.0


def test_stochastic_ex_date_lower_bound_has_no_lookahead() -> None:
    parent = Node(120.0)
    option = CallOption(strike=100.0, is_american=True)
    event = DividendStepEvent(
        amount=20.0, elapsed_before_event=0.10,
        remaining_time=0.15, rate=0.05, volatility=0.20,
    )
    first = offgrid_exercise_bound(parent, 5.0, option, event)
    second = offgrid_exercise_bound(parent, 5.0, option, event)
    assert first == second
    assert first > 20.0


def test_stochastic_warning_does_not_claim_exact_bridge() -> None:
    request = make_request(volatility=0.2)
    warning = assess_dividend_exercise(request, 23.0).warning
    assert warning and "conservative lower bound" in warning and "Check convergence" in warning


def test_zero_volatility_offgrid_audit_benchmark_has_no_model_gap() -> None:
    request = make_request(ex_date=0.6)
    result = price_contract(request, algorithm="Recursive")
    audit = assess_dividend_exercise(request, result.price)
    assert audit.deterministic_reference == pytest.approx(result.price, abs=1e-8)
    assert abs(audit.model_minus_reference) < 1e-8
    assert "model minus reference" not in (audit.warning or "")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_aligned_stochastic_exdate_has_no_offgrid_warning(
    model: str, algorithm: str,
) -> None:
    """N=20, t=0.60 is exactly the twelfth lattice endpoint."""
    request = make_request(ex_date=0.60, model=model, volatility=0.20, steps=20)
    result = price_contract(request, algorithm=algorithm)
    assert assess_dividend_exercise(request, result.price).warning is None


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_offgrid_single_dividend_put_post_jump_matches_reference(
    model: str, algorithm: str,
) -> None:
    """An in-the-money deterministic put should exercise after the cash jump."""
    request = make_request(
        ex_date=0.60, model=model, option_type="Put",
        spot=120.0, strike=140.0, volatility=0.0,
    )
    exact, event = deterministic_american_reference(request)
    assert event == "post-dividend"
    actual = price_contract(request, algorithm=algorithm).price
    assert actual == pytest.approx(exact, rel=1e-11, abs=1e-9)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_stochastic_put_with_offgrid_exdate_dominates_european(
    model: str, algorithm: str,
) -> None:
    request = make_request(
        ex_date=0.61, model=model, option_type="Put", strike=140.0,
        volatility=0.20, steps=20,
    )
    american = price_contract(request, algorithm=algorithm).price
    european = price_contract(
        replace(request, exercise="European"), algorithm=algorithm,
    ).price
    assert math.isfinite(american)
    assert american >= european - 1e-9
    assert "nonanticipative" in (assess_dividend_exercise(request, american).warning or "")


def test_post_dividend_put_uses_shifted_strike_in_stochastic_case() -> None:
    """Check the exercise-at-ex-date strategy against independent BS pricing."""
    from src.dividend_exercise import offgrid_exercise_bound

    parent = Node(120.0)
    option = PutOption(strike=140.0, is_american=True)
    event = DividendStepEvent(
        amount=20.0, elapsed_before_event=0.1,
        remaining_time=0.15, rate=0.05, volatility=0.2,
    )
    actual = offgrid_exercise_bound(parent, 0.0, option, event)
    expected = BlackScholesPricer().price(
        option=PutOption(strike=160.0), spot=120.0,
        rate=0.05, volatility=0.2, maturity=0.1,
    )
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_zero_strike_call_with_dividend_uses_shifted_strike() -> None:
    """The K=0 shortcut cannot ignore a positive ex-date cash adjustment."""
    from src.dividend_exercise import _exercise_at_ex_date

    actual = _exercise_at_ex_date(
        120.0, CallOption(strike=0.0), 0.05, 0.2, 0.1,
        cash_adjustment=20.0,
    )
    expected = BlackScholesPricer().price(
        option=CallOption(strike=20.0), spot=120.0,
        rate=0.05, volatility=0.2, maturity=0.1,
    )
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
