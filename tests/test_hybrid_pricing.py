import math

from src.binomial_tree import BinomialTree
from src.dividend import CashDividend
from src.hybrid_tree import HybridTree
from src.option import CallOption, PutOption
from src.recursive_pricer import RecursivePricer
from src.trinomial_tree import TrinomialTree


# Without dividends, hybrid pricing must reduce exactly to binomial pricing.
def test_hybrid_without_dividend_matches_binomial_price() -> None:
    option = CallOption(
        strike=100.0
    )

    binomial_price = RecursivePricer().price(
        BinomialTree(
            spot=100.0,
            rate=0.02,
            volatility=0.20,
            maturity=1.0,
            nb_steps=10,
        ),
        option,
    )

    hybrid_price = RecursivePricer().price(
        HybridTree(
            spot=100.0,
            rate=0.02,
            volatility=0.20,
            maturity=1.0,
            nb_steps=10,
        ),
        option,
    )

    assert math.isclose(
        hybrid_price,
        binomial_price,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


# At zero rates, a cash dividend must lower the hybrid European call price.
def test_hybrid_dividend_reduces_call_price() -> None:
    option = CallOption(
        strike=100.0
    )

    no_dividend_price = RecursivePricer().price(
        HybridTree(
            spot=100.0,
            rate=0.0,
            volatility=0.20,
            maturity=1.0,
            nb_steps=20,
        ),
        option,
    )

    dividend_price = RecursivePricer().price(
        HybridTree(
            spot=100.0,
            rate=0.0,
            volatility=0.20,
            maturity=1.0,
            nb_steps=20,
            dividends=[
                CashDividend(
                    time=0.5,
                    amount=3.0,
                )
            ],
        ),
        option,
    )

    assert dividend_price < no_dividend_price


# At zero rates, a cash dividend must raise the hybrid European put price.
def test_hybrid_dividend_increases_put_price() -> None:
    option = PutOption(
        strike=100.0
    )

    no_dividend_price = RecursivePricer().price(
        HybridTree(
            spot=100.0,
            rate=0.0,
            volatility=0.20,
            maturity=1.0,
            nb_steps=20,
        ),
        option,
    )

    dividend_price = RecursivePricer().price(
        HybridTree(
            spot=100.0,
            rate=0.0,
            volatility=0.20,
            maturity=1.0,
            nb_steps=20,
            dividends=[
                CashDividend(
                    time=0.5,
                    amount=3.0,
                )
            ],
        ),
        option,
    )

    assert dividend_price > no_dividend_price


# Hybrid European prices must satisfy discrete-dividend put-call parity.
def test_hybrid_put_call_parity_with_dividend() -> None:
    spot = 100.0
    strike = 100.0
    dividend_amount = 3.0

    dividends = [
        CashDividend(
            time=0.5,
            amount=dividend_amount,
        )
    ]

    call_price = RecursivePricer().price(
        HybridTree(
            spot=spot,
            rate=0.0,
            volatility=0.20,
            maturity=1.0,
            nb_steps=20,
            dividends=dividends,
        ),
        CallOption(
            strike=strike
        ),
    )

    put_price = RecursivePricer().price(
        HybridTree(
            spot=spot,
            rate=0.0,
            volatility=0.20,
            maturity=1.0,
            nb_steps=20,
            dividends=dividends,
        ),
        PutOption(
            strike=strike
        ),
    )

    expected_difference = (
        spot
        - dividend_amount
        - strike
    )

    assert math.isclose(
        call_price - put_price,
        expected_difference,
        rel_tol=1e-10,
        abs_tol=1e-10,
    )


# Hybrid and full trinomial prices should remain close with one cash dividend.
def test_hybrid_price_is_close_to_full_trinomial() -> None:
    option = CallOption(
        strike=100.0
    )

    dividends = [
        CashDividend(
            time=0.5,
            amount=3.0,
        )
    ]

    hybrid_price = RecursivePricer().price(
        HybridTree(
            spot=100.0,
            rate=0.02,
            volatility=0.20,
            maturity=1.0,
            nb_steps=50,
            dividends=dividends,
        ),
        option,
    )

    trinomial_price = RecursivePricer().price(
        TrinomialTree(
            spot=100.0,
            rate=0.02,
            volatility=0.20,
            maturity=1.0,
            nb_steps=50,
            dividends=dividends,
        ),
        option,
    )

    assert math.isclose(
        hybrid_price,
        trinomial_price,
        abs_tol=0.10,
    )


# Several dividends must also be priceable by the hybrid lattice.
def test_hybrid_prices_multiple_dividends() -> None:
    option = CallOption(
        strike=100.0
    )

    one_dividend_price = RecursivePricer().price(
        HybridTree(
            spot=100.0,
            rate=0.0,
            volatility=0.20,
            maturity=1.0,
            nb_steps=40,
            dividends=[
                CashDividend(
                    time=0.25,
                    amount=2.0,
                )
            ],
        ),
        option,
    )

    two_dividend_price = RecursivePricer().price(
        HybridTree(
            spot=100.0,
            rate=0.0,
            volatility=0.20,
            maturity=1.0,
            nb_steps=40,
            dividends=[
                CashDividend(
                    time=0.25,
                    amount=2.0,
                ),
                CashDividend(
                    time=0.75,
                    amount=3.0,
                ),
            ],
        ),
        option,
    )

    assert two_dividend_price < one_dividend_price