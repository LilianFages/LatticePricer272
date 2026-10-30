"""Shared no-dividend European call valuations for numerical experiments.

Create a fresh option and lattice per call to preserve study independence.
Recursive-pricing comparisons intentionally retain their own algorithm.
"""

from src.backward_pricer import BackwardPricer
from src.binomial_tree import BinomialTree
from src.black_scholes import BlackScholesPricer
from src.option import CallOption


def black_scholes_call_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float
) -> float:
    """Compute the independent analytical European call benchmark."""
    option = CallOption(strike=strike)
    return BlackScholesPricer().price(
        option=option,
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
    )


def backward_binomial_call_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    nb_steps: int
) -> float:
    """Compute a European call price using a fresh backward binomial tree."""
    tree = BinomialTree(
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        nb_steps=nb_steps,
    )
    option = CallOption(strike=strike)
    return BackwardPricer().price(tree, option)
