"""Parity and structural tests for the iterative multi-model backward pricer."""

from __future__ import annotations

import math

import pytest

from src.allocation_pruned_trinomial_tree import AllocationPrunedTrinomialTree
from src.backward_pricer import BackwardPricer
from src.binomial_tree import BinomialTree
from src.dividend import CashDividend
from src.hybrid_tree import HybridTree
from src.iterative_backward_pricer import IterativeBackwardPricer
from src.node import Node
from src.option import CallOption, Option, PutOption
from src.recursive_pricer import RecursivePricer
from src.trinomial_tree import TrinomialTree

FAMILIES = (
    "Binomial",
    "Hybrid",
    "Trinomial Full",
    "Trinomial Branch",
    "Trinomial Allocation",
)
DIVIDEND_FAMILIES = FAMILIES[1:]


def make_tree(
    family: str,
    *,
    n: int = 12,
    spot: float = 100.0,
    rate: float = 0.02,
    sigma: float = 0.20,
    dividends: tuple[tuple[float, float], ...] = (),
):
    """Create a fresh graph for each independent pricing method."""
    kwargs = dict(
        spot=spot,
        rate=rate,
        volatility=sigma,
        maturity=1.0,
        nb_steps=n,
    )
    events = [CashDividend(time=t, amount=a) for t, a in dividends]

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


def compare_methods(family: str, option: Option, **kwargs) -> None:
    """Assert strict price parity on independently constructed trees."""
    recursive_tree = make_tree(family, **kwargs)
    backward_tree = make_tree(family, **kwargs)
    recursive = RecursivePricer().price(recursive_tree, option)
    backward = IterativeBackwardPricer().price(backward_tree, option)
    assert math.isfinite(recursive)
    assert math.isfinite(backward)
    assert math.isclose(recursive, backward, rel_tol=1e-12, abs_tol=1e-12)
    assert backward_tree.root.option_value == backward


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", ("Call", "Put"))
@pytest.mark.parametrize("american", (False, True))
@pytest.mark.parametrize("n", (1, 7, 28))
def test_no_dividend_parity(
    family: str, kind: str, american: bool, n: int,
) -> None:
    option_type = CallOption if kind == "Call" else PutOption
    compare_methods(family, option_type(strike=105.0, is_american=american), n=n)


@pytest.mark.parametrize("family", DIVIDEND_FAMILIES)
@pytest.mark.parametrize("kind", ("Call", "Put"))
@pytest.mark.parametrize("american", (False, True))
def test_single_dividend_parity(
    family: str, kind: str, american: bool,
) -> None:
    option_type = CallOption if kind == "Call" else PutOption
    compare_methods(
        family, option_type(strike=100.0, is_american=american),
        n=20, dividends=((0.5, 3.0),),
    )


@pytest.mark.parametrize("family", DIVIDEND_FAMILIES)
@pytest.mark.parametrize("kind", ("Call", "Put"))
def test_two_dividends_parity(family: str, kind: str) -> None:
    option_type = CallOption if kind == "Call" else PutOption
    compare_methods(
        family, option_type(strike=100.0, is_american=True),
        n=24, dividends=((0.25, 2.0), (0.75, 3.0)),
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_american_call_negative_rate(family: str) -> None:
    compare_methods(
        family, CallOption(strike=100.0, is_american=True),
        n=8, spot=150.0, rate=-0.10, sigma=0.15,
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_zero_volatility_parity(family: str) -> None:
    compare_methods(family, PutOption(strike=100.0, is_american=True), n=8, sigma=0.0)


def test_existing_column_backward_parity() -> None:
    """Do not replace the separate optimized binomial backward algorithm."""
    params = dict(
        spot=100.0, rate=0.02, volatility=0.20,
        maturity=1.0, nb_steps=28,
    )
    option = PutOption(strike=105.0, is_american=True)
    column = BackwardPricer().price(BinomialTree(**params), option)
    graph = IterativeBackwardPricer().price(BinomialTree(**params), option)
    assert math.isclose(column, graph, rel_tol=1e-12, abs_tol=1e-12)


def test_shared_recombining_nodes_are_priced() -> None:
    tree = make_tree("Trinomial Full", n=3)
    IterativeBackwardPricer().price(tree, CallOption(strike=100.0))
    root = tree.root
    assert root.next_up is not None
    assert root.next_mid is not None
    assert root.next_down is not None
    assert root.next_mid.option_value is not None


def test_incomplete_graph_fails_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = make_tree("Binomial", n=2)

    def break_graph() -> None:
        tree.root.next_up = Node(110.0)
        tree.root.next_down = None

    monkeypatch.setattr(tree, "build", break_graph)
    with pytest.raises(RuntimeError, match="Incomplete lattice"):
        IterativeBackwardPricer().price(tree, CallOption(strike=100.0))


def test_time_layer_cycle_fails_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = make_tree("Binomial", n=1)

    def make_cycle() -> None:
        tree.root.next_up = tree.root
        tree.root.next_down = Node(90.0)

    monkeypatch.setattr(tree, "build", make_cycle)
    with pytest.raises(RuntimeError, match="more than one time layer"):
        IterativeBackwardPricer().price(tree, CallOption(strike=100.0))


def test_backward_does_not_call_recursive_pricer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("RecursivePricer unexpectedly called.")

    monkeypatch.setattr(RecursivePricer, "price", forbidden)
    value = IterativeBackwardPricer().price(
        make_tree("Hybrid", n=8), CallOption(strike=100.0),
    )
    assert value >= 0.0
