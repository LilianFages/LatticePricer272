"""Backward pricing across lattice families and structural error cases.

Consolidated without dropping individual regression cases.
"""

from __future__ import annotations

import math
import pytest
from src.lattices.allocation_pruned_trinomial_tree import AllocationPrunedTrinomialTree
from src.pricing.backward_pricer import BackwardPricer
from src.lattices.binomial_tree import BinomialTree
from src.contracts.dividend import CashDividend
from src.lattices.hybrid_tree import HybridTree
from src.lattices.node import Node
from src.contracts.option import CallOption, Option, PutOption
from src.pricing.recursive_pricer import RecursivePricer
from src.lattices.trinomial_tree import TrinomialTree


# test_backward_models.py (tests preserved)
FAMILIES = (
    "Binomial",
    "Hybrid",
    "Trinomial Full",
    "Trinomial Branch",
    "Trinomial Allocation",
)
DIVIDEND_FAMILIES = FAMILIES[1:]


# Build the same financial scenario for each pricing method.
def _make_tree(family: str,
    *,
    n: int = 12,
    spot: float = 100.0,
    rate: float = 0.02,
    sigma: float = 0.20,
    dividends: tuple[tuple[float, float], ...] = (),
) -> BinomialTree | HybridTree | TrinomialTree | AllocationPrunedTrinomialTree:
    """Create a fresh graph for each independent pricing method."""
    kwargs = dict(
        spot=spot,
        rate=rate,
        volatility=sigma,
        maturity=1.0,
        nb_steps=n,
    )
    events = [CashDividend(time=t, amount=a) for t, a in dividends]

    # Reject dividends only for the unsupported binomial family.
    if family == "Binomial":
        if events:
            raise ValueError("Binomial does not support cash dividends.")
        return BinomialTree(**kwargs)
    if family == "Hybrid":
        return HybridTree(**kwargs, dividends=events)
    if family == "Trinomial Full":
        return TrinomialTree(**kwargs, dividends=events, pruning_threshold=0.0)
    if family == "Trinomial Branch":
        return TrinomialTree(**kwargs, dividends=events, pruning_threshold=1e-18)
    if family == "Trinomial Allocation":
        return AllocationPrunedTrinomialTree(
            **kwargs, dividends=events, pruning_threshold=1e-18,
        )
    raise ValueError("Unknown model family.")


# Keep each implementation on a distinct, freshly built lattice.
def _compare_methods(family: str, option: Option, **kwargs: object) -> None:
    """Assert strict price parity on independently constructed trees."""
    recursive_tree = _make_tree(family, **kwargs)
    backward_tree = _make_tree(family, **kwargs)
    recursive = RecursivePricer().price(recursive_tree, option)
    backward = BackwardPricer().price(backward_tree, option)
    assert math.isfinite(recursive)
    assert math.isfinite(backward)
    assert math.isclose(recursive, backward, rel_tol=1e-12, abs_tol=1e-12)
    assert backward_tree.root.option_value == backward


# Compare all supported contracts, exercise rights and discretizations.
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", ("Call", "Put"))
@pytest.mark.parametrize("american", (False, True))
@pytest.mark.parametrize("n", (1, 7, 28))
def test_no_dividend_parity(family: str, kind: str, american: bool, n: int) -> None:
    option_type = CallOption if kind == "Call" else PutOption
    _compare_methods(family, option_type(strike=105.0, is_american=american), n=n)


@pytest.mark.parametrize("family", DIVIDEND_FAMILIES)
@pytest.mark.parametrize("kind", ("Call", "Put"))
@pytest.mark.parametrize("american", (False, True))
def test_single_dividend_parity(family: str, kind: str, american: bool) -> None:
    option_type = CallOption if kind == "Call" else PutOption
    _compare_methods(
        family, option_type(strike=100.0, is_american=american),
        n=20, dividends=((0.5, 3.0),),
    )


@pytest.mark.parametrize("family", DIVIDEND_FAMILIES)
@pytest.mark.parametrize("kind", ("Call", "Put"))
# Keep pricing comparisons on identical contracts and lattice settings.
def test_two_dividends_parity(family: str, kind: str) -> None:
    option_type = CallOption if kind == "Call" else PutOption
    _compare_methods(
        family, option_type(strike=100.0, is_american=True),
        n=24, dividends=((0.25, 2.0), (0.75, 3.0)),
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_american_call_negative_rate(family: str) -> None:
    _compare_methods(
        family, CallOption(strike=100.0, is_american=True),
        n=8, spot=150.0, rate=-0.10, sigma=0.15,
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_zero_volatility_parity(family: str) -> None:
    _compare_methods(family, PutOption(strike=100.0, is_american=True), n=8, sigma=0.0)


# Shared descendants of recombining branches must also be priced.
def test_shared_recombining_nodes_are_priced() -> None:
    tree = _make_tree("Trinomial Full", n=3)
    BackwardPricer().price(tree, CallOption(strike=100.0))
    root = tree.root
    assert root.next_up is not None
    assert root.next_mid is not None
    assert root.next_down is not None
    assert root.next_mid.option_value is not None


# Malformed lattice edges must fail explicitly instead of being skipped.
def test_incomplete_graph_fails_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = _make_tree("Binomial", n=2)

    def break_graph() -> None:
        tree.root.next_up = Node(110.0)
        tree.root.next_down = None

    monkeypatch.setattr(tree, "build", break_graph)
    with pytest.raises(RuntimeError, match="Incomplete lattice"):
        BackwardPricer().price(tree, CallOption(strike=100.0))


def test_time_layer_cycle_fails_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = _make_tree("Binomial", n=1)

    def make_cycle() -> None:
        tree.root.next_up = tree.root
        tree.root.next_down = Node(90.0)

    # Keep pricing comparisons on identical contracts and lattice settings.
    monkeypatch.setattr(tree, "build", make_cycle)
    with pytest.raises(RuntimeError, match="more than one time layer"):
        BackwardPricer().price(tree, CallOption(strike=100.0))


# Ensure backward valuation never silently invokes recursive pricing.
def test_backward_does_not_call_recursive_pricer(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("RecursivePricer unexpectedly called.")

    monkeypatch.setattr(RecursivePricer, "price", forbidden)
    value = BackwardPricer().price(
        _make_tree("Hybrid", n=8), CallOption(strike=100.0),
    )
    assert value >= 0.0


# test_backward_pricer.py (tests preserved)
# Centralize the common market parameters used by the pricing tests.
def _build_tree(
    nb_steps: int,
    spot: float = 100.0,
    rate: float = 0.02,
    volatility: float = 0.20
) -> BinomialTree:
    return BinomialTree(
        spot=spot,
        rate=rate,
        volatility=volatility,
        maturity=1.0,
        nb_steps=nb_steps,
    )


# Compare both algorithms on independent copies of the same lattice.
def _price_with_both_pricers(
    option: Option,
    nb_steps: int,
    spot: float = 100.0,
    rate: float = 0.02,
    volatility: float = 0.20
) -> tuple[float, float]:
    backward_tree = _build_tree(
        nb_steps,
        spot,
        rate,
        volatility,
    )
    recursive_tree = _build_tree(
        nb_steps,
        spot,
        rate,
        volatility,
    )

    # Keep pricing comparisons on identical contracts and lattice settings.
    backward_price = BackwardPricer().price(
        backward_tree,
        option,
    )

    # Use the recursive implementation as an independent pricing benchmark.
    recursive_price = RecursivePricer().price(
        recursive_tree,
        option,
    )

    return backward_price, recursive_price


# A one-step price must equal the discounted risk-neutral expectation.
def test_backward_one_step_call() -> None:
    tree = _build_tree(nb_steps=1)
    option = CallOption(strike=100.0)

    price = BackwardPricer().price(
        tree,
        option,
    )

    assert tree.root.next_up is not None
    assert tree.root.next_down is not None

    up_payoff = option.payoff(
        tree.root.next_up.price
    )
    down_payoff = option.payoff(
        tree.root.next_down.price
    )

    # Compute directly the theoretical one-step continuation value.
    expected_price = math.exp(
        -tree.rate * tree.dt
    ) * (
        tree.up_probability * up_payoff
        + tree.down_probability * down_payoff
    )

    assert math.isclose(
        price,
        expected_price,
        rel_tol=1e-12,
    )


# Backward and recursive induction must agree for a European call.
def test_backward_matches_recursive_call() -> None:
    option = CallOption(strike=100.0)

    backward_price, recursive_price = _price_with_both_pricers(
        option=option,
        nb_steps=5,
    )

    assert math.isclose(
        backward_price,
        recursive_price,
        rel_tol=1e-12,
    )


# The same equivalence must hold for a European put.
def test_backward_matches_recursive_put() -> None:
    option = PutOption(strike=100.0)

    backward_price, recursive_price = _price_with_both_pricers(
        option=option,
        nb_steps=6,
    )

    assert math.isclose(
        backward_price,
        recursive_price,
        rel_tol=1e-12,
    )


# Backward induction must store the final price at the root node.
def test_backward_pricer_stores_root_value() -> None:
    tree = _build_tree(nb_steps=4)
    option = CallOption(strike=100.0)

    price = BackwardPricer().price(
        tree,
        option,
    )

    assert tree.root.option_value is not None
    assert math.isclose(
        tree.root.option_value,
        price,
        rel_tol=1e-12,
    )


# Both algorithms must also agree when early exercise is allowed.
def test_backward_matches_recursive_american_put() -> None:
    option = PutOption(
        strike=100.0,
        is_american=True,
    )

    backward_price, recursive_price = _price_with_both_pricers(
        option=option,
        nb_steps=8,
        spot=90.0,
        rate=0.05,
        volatility=0.25,
    )

    assert math.isclose(
        backward_price,
        recursive_price,
        rel_tol=1e-12,
    )


# A deep in-the-money American put can optimally exercise immediately.
def test_backward_american_put_can_exercise_early() -> None:
    european_tree = _build_tree(
        nb_steps=1,
        spot=50.0,
        rate=0.10,
        volatility=0.01,
    )
    american_tree = _build_tree(
        nb_steps=1,
        spot=50.0,
        rate=0.10,
        volatility=0.01,
    )

    european_put = PutOption(
        strike=100.0,
        is_american=False,
    )
    american_put = PutOption(
        strike=100.0,
        is_american=True,
    )

    # Keep pricing comparisons on identical contracts and lattice settings.
    european_price = BackwardPricer().price(
        european_tree,
        european_put,
    )

    # Compare early exercise against the corresponding European contract.
    american_price = BackwardPricer().price(
        american_tree,
        american_put,
    )

    # Immediate exercise gives the intrinsic value K - S = 50.
    assert american_price > european_price
    assert math.isclose(
        american_price,
        50.0,
        rel_tol=1e-12,
    )


# Negative rates can make early exercise relevant for American calls.
def test_backward_matches_recursive_american_call_negative_rate() -> None:
    option = CallOption(
        strike=100.0,
        is_american=True,
    )

    backward_price, recursive_price = _price_with_both_pricers(
        option=option,
        nb_steps=8,
        spot=150.0,
        rate=-0.10,
        volatility=0.15,
    )

    assert math.isclose(
        backward_price,
        recursive_price,
        rel_tol=1e-12,
    )
