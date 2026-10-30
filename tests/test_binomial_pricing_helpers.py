"""Regression tests for the shared experimental European-call pricers."""

import pytest

from experiments import (
    binomial_convergence,
    binomial_convergence_order,
    binomial_error_gap,
    binomial_precision_target,
    trinomial_convergence,
)
from experiments.binomial_pricing_helpers import (
    backward_binomial_call_price,
    black_scholes_call_price,
)
from src.backward_pricer import BackwardPricer
from src.binomial_tree import BinomialTree
from src.black_scholes import BlackScholesPricer
from src.option import CallOption


@pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
def test_analytical_reference_matches_direct_pricer(strike: float) -> None:
    """The shared benchmark reproduces the original Black-Scholes call."""
    expected = BlackScholesPricer().price(
        option=CallOption(strike=strike),
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
    )
    assert black_scholes_call_price(
        spot=100.0,
        strike=strike,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
    ) == pytest.approx(expected)


@pytest.mark.parametrize("steps", [5, 25])
def test_backward_reference_matches_direct_pricer(steps: int) -> None:
    """The lattice helper preserves construction and backward induction."""
    expected = BackwardPricer().price(
        BinomialTree(
            spot=100.0,
            rate=0.02,
            volatility=0.20,
            maturity=1.0,
            nb_steps=steps,
        ),
        CallOption(strike=95.0),
    )
    assert backward_binomial_call_price(
        spot=100.0,
        strike=95.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=steps,
    ) == pytest.approx(expected)


def test_existing_binomial_wrappers_keep_argument_order() -> None:
    """The original public module functions keep their signatures."""
    n, strike = 20, 95.0
    reference = backward_binomial_call_price(
        spot=100.0,
        strike=strike,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=n,
    )
    assert binomial_convergence_order.tree_price(n, strike) == pytest.approx(reference)
    assert binomial_error_gap.tree_price(strike, n) == pytest.approx(reference)
    assert binomial_precision_target.tree_price(strike, n) == pytest.approx(reference)
    assert binomial_convergence.tree_price(n) == pytest.approx(
        backward_binomial_call_price(
            spot=100.0,
            strike=100.0,
            rate=0.02,
            volatility=0.20,
            maturity=1.0,
            nb_steps=n,
        )
    )


def test_existing_black_scholes_wrappers_keep_contracts() -> None:
    """All five experiments use the same reference-market assumptions."""
    reference = black_scholes_call_price(
        spot=100.0,
        strike=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
    )
    assert binomial_convergence.black_scholes_price() == pytest.approx(reference)
    assert trinomial_convergence.black_scholes_price() == pytest.approx(reference)
    for module in (
        binomial_convergence_order,
        binomial_error_gap,
        binomial_precision_target,
    ):
        assert module.black_scholes_price(100.0) == pytest.approx(reference)
