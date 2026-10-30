"""Phase 6B: one shared interpretation of dividends and reachable nodes."""
from __future__ import annotations

import math

import pytest

from src.dividend import CashDividend
from src.dividend_timing import dividend_for_lattice_step
from src.hybrid_tree import HybridTree
from src.node import count_reachable_nodes
from src.node import Node
from src.transition_moments import TransitionMoments
from src.trinomial_tree import TrinomialTree


def test_reachability_follows_priced_edges_not_allocated_column() -> None:
    root, a, b, unreachable = [Node(float(i + 1)) for i in range(4)]
    root.next_up = a
    root.next_mid = a  # Recombinant edge: do not double count.
    root.next_down = b
    a.next_mid = root  # Malformed cyclic graph must terminate.
    a.lower_neighbor = unreachable  # Merely allocated, not a priced successor.
    assert count_reachable_nodes(root) == 3
    assert count_reachable_nodes(unreachable) == 1


def test_same_dividend_lookup_in_hybrid_and_trinomial() -> None:
    dividends = [CashDividend(195 / 365, 2.0)]
    assert dividend_for_lattice_step(dividends, 39, 1.0, 73) is dividends[0]
    assert dividend_for_lattice_step(dividends, 38, 1.0, 73) is None
    for engine in (HybridTree, TrinomialTree):
        tree = engine(spot=100, rate=.02, volatility=.2, maturity=1,
                      nb_steps=73, dividends=dividends)
        assert tree._dividend_for_step(39) is dividends[0]
        assert tree._dividend_for_step(38) is None


def test_two_dividends_in_same_lattice_step_rejected_consistently() -> None:
    dividends = [CashDividend(.52, 2.0), CashDividend(.51, 1.0)]
    with pytest.raises(ValueError, match="Only one dividend per time step"):
        dividend_for_lattice_step(dividends, 3, 1.0, 4)
    for engine in (HybridTree, TrinomialTree):
        tree = engine(spot=100, rate=.02, volatility=.2, maturity=1,
                      nb_steps=4, dividends=dividends)
        with pytest.raises(ValueError, match="Only one dividend per time step"):
            tree._dividend_for_step(3)


@pytest.mark.parametrize("step, dt", [(1, .25), (2, .25), (39, 1/73)])
def test_shared_moments_same_as_original_interval(step: int, dt: float) -> None:
    spot, rate, volatility = 100.0, .03, .2
    dividend = None if step == 1 else CashDividend((step - .4) * dt, 1.0)
    expected = TransitionMoments.compute(
        spot=spot, rate=rate, volatility=volatility,
        step_start=(step - 1) * dt, step_end=step * dt, dividend=dividend,
    )
    observed = TransitionMoments.at_lattice_step(
        spot=spot, rate=rate, volatility=volatility,
        step=step, dt=dt, dividend=dividend,
    )
    assert observed == expected
    assert all(math.isfinite(x) for x in observed)
