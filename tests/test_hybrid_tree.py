"""Hybrid-tree construction and pricing with discrete dividends.

Consolidated without dropping individual regression cases.
"""

from __future__ import annotations

import math
from src.binomial_tree import BinomialTree
from src.dividend import CashDividend
from src.hybrid_tree import HybridTree
from src.option import CallOption, PutOption
from src.recursive_pricer import RecursivePricer
from src.trinomial_tree import TrinomialTree
from src.node import Node


# test_hybrid_pricing.py (tests preserved)
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

    # Without dividends, the hybrid tree reduces to binomial pricing.
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

    # At zero rates, a positive cash dividend lowers the European call.
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

    # At zero rates, a cash dividend raises the European put.
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

    # The present value of cash dividends enters put-call parity explicitly.
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

    # The present value of cash dividends enters put-call parity explicitly.
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

    # Price differences must stay within tolerance for matched dividend inputs.
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

    # Price differences must stay within tolerance for matched dividend inputs.
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

    # Several ex-dates must be handled without losing a dividend payment.
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


# test_hybrid_tree.py (tests preserved)
def _build_hybrid(
    nb_steps: int = 3,
    dividends: list[CashDividend] | None = None
) -> HybridTree:
    """Build a standard hybrid test lattice."""
    return HybridTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=nb_steps,
        dividends=dividends,
    )


def _terminal_top(
    tree: BinomialTree | HybridTree
) -> Node:
    """Return the highest node of the terminal column."""
    current_node: Node = tree.root

    for _ in range(
        tree.nb_steps
    ):
        assert current_node.next_up is not None
        current_node = current_node.next_up

    return current_node


def _count_column_nodes(
    top_node: Node
) -> int:
    """Count the nodes of one lattice column."""
    node_count = 0
    current_node: Node | None = top_node

    while current_node is not None:
        node_count += 1

        current_node = (
            current_node.lower_neighbor
        )

    return node_count


# Without dividends, the hybrid lattice must reduce to the binomial lattice.
def test_hybrid_without_dividend_matches_binomial() -> None:
    hybrid_tree = _build_hybrid(
        nb_steps=4
    )

    binomial_tree = BinomialTree(
        spot=100.0,
        rate=0.02,
        volatility=0.20,
        maturity=1.0,
        nb_steps=4,
    )

    hybrid_tree.build()
    binomial_tree.build()

    # Every non-dividend hybrid step should match the pure binomial geometry.
    hybrid_node: Node | None = (
        _terminal_top(hybrid_tree)
    )

    binomial_node: Node | None = (
        _terminal_top(binomial_tree)
    )

    while hybrid_node is not None:
        assert binomial_node is not None

        assert math.isclose(
            hybrid_node.price,
            binomial_node.price,
            rel_tol=1e-12,
        )

        hybrid_node = (
            hybrid_node.lower_neighbor
        )

        binomial_node = (
            binomial_node.lower_neighbor
        )

    assert binomial_node is None


# A dividend step must switch from two branches to three branches.
def test_dividend_step_is_trinomial() -> None:
    tree = _build_hybrid(
        nb_steps=3,
        dividends=[
            CashDividend(
                time=2.0 / 3.0,
                amount=3.0,
            )
        ],
    )

    tree.build()

    first_up = tree.root.next_up

    assert first_up is not None

    # The first transition is a normal binomial step.
    assert tree.root.next_mid is None

    # The second transition contains the dividend.
    assert first_up.next_up is not None
    assert first_up.next_mid is not None
    assert first_up.next_down is not None


# The dividend bridge must use the spacing of a binomial column.
def test_dividend_bridge_uses_binomial_column_spacing() -> None:
    tree = _build_hybrid(
        nb_steps=3,
        dividends=[
            CashDividend(
                time=2.0 / 3.0,
                amount=3.0,
            )
        ],
    )

    tree.build()

    first_up = tree.root.next_up

    assert first_up is not None
    assert first_up.next_up is not None
    # Check the bridge level ratio against the model's binomial spacing.
    assert first_up.next_mid is not None

    bridge_ratio = (
        first_up.next_up.price
        / first_up.next_mid.price
    )

    assert math.isclose(
        bridge_ratio,
        tree.binomial_alpha ** 2,
        rel_tol=1e-12,
    )

    assert math.isclose(
        bridge_ratio,
        tree.level_ratio,
        rel_tol=1e-12,
    )


# Trinomial probabilities on a dividend step must be local and admissible.
def test_dividend_step_probabilities_are_local() -> None:
    tree = _build_hybrid(
        nb_steps=3,
        dividends=[
            CashDividend(
                time=2.0 / 3.0,
                amount=3.0,
            )
        ],
    )

    tree.build()

    upper_node = tree.root.next_up
    lower_node = tree.root.next_down

    assert upper_node is not None
    # Dividend transitions must match node-specific forward moments.
    assert lower_node is not None

    for node in [
        upper_node,
        lower_node,
    ]:
        assert node.up_probability is not None
        assert node.mid_probability is not None
        assert node.down_probability is not None

        probability_sum = (
            node.up_probability
            + node.mid_probability
            + node.down_probability
        )

        assert math.isclose(
            probability_sum,
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

    assert not math.isclose(
        upper_node.up_probability,
        lower_node.up_probability,
        rel_tol=1e-12,
    )


# After the dividend step, the lattice must return to binomial branching.
def test_tree_returns_to_binomial_after_dividend() -> None:
    tree = _build_hybrid(
        nb_steps=3,
        dividends=[
            CashDividend(
                time=2.0 / 3.0,
                amount=3.0,
            )
        ],
    )

    tree.build()

    first_up = tree.root.next_up

    assert first_up is not None
    assert first_up.next_mid is not None

    dividend_node = (
        first_up.next_mid
    )

    # The third transition contains no dividend.
    assert dividend_node.next_up is not None
    assert dividend_node.next_mid is None
    assert dividend_node.next_down is not None


# Binomial branching after a dividend must still recombine.
def test_binomial_step_recombines_after_dividend() -> None:
    tree = _build_hybrid(
        nb_steps=3,
        dividends=[
            CashDividend(
                time=2.0 / 3.0,
                amount=3.0,
            )
        ],
    )

    tree.build()

    first_up = tree.root.next_up

    assert first_up is not None
    assert first_up.next_mid is not None

    upper_node = (
        first_up.next_mid
    )

    lower_node = (
        upper_node.lower_neighbor
    )

    assert lower_node is not None

    assert upper_node.next_down is not None
    assert lower_node.next_up is not None

    # Down from the upper node and up from the lower node must recombine.
    assert (
        upper_node.next_down
        is lower_node.next_up
    )


# One moderate dividend adds one node versus a pure binomial terminal column.
def test_one_dividend_adds_one_terminal_node() -> None:
    tree = _build_hybrid(
        nb_steps=3,
        dividends=[
            CashDividend(
                time=2.0 / 3.0,
                amount=3.0,
            )
        ],
    )

    tree.build()

    terminal_top = _terminal_top(
        tree
    )

    assert (
        _count_column_nodes(
            terminal_top
        )
        == 5
    )


# Hybrid dividend bridges must use the same neighbor and geometric conventions.
def test_shared_hybrid_stencil_and_column_bottom() -> None:
    import pytest

    upper, middle, lower, bottom = (Node(value) for value in (160., 120., 90., 60.))
    upper.lower_neighbor = middle
    middle.lower_neighbor = lower
    lower.lower_neighbor = bottom
    assert HybridTree._bottom_node(upper) is bottom
    assert HybridTree._find_middle_successor(118., upper, middle) == (
        upper, middle, lower,
    )
    assert HybridTree._find_middle_successor(88., upper, middle) == (
        middle, lower, bottom,
    )
    with pytest.raises(RuntimeError, match="Hybrid grid is too narrow below"):
        HybridTree._find_middle_successor(60., upper, middle)


def test_hybrid_exponent_wrapper_uses_bridge_ratio() -> None:
    tree = _build_hybrid(nb_steps=3)
    assert tree._closest_exponent(100. * tree.level_ratio, 100.) == 1
