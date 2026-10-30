"""Iterative backward induction on the existing recombining pricing graph.

This course-style backward induction values the existing lattice from
maturity to valuation. It applies the same transition probabilities and
exercise rules as RecursivePricer, without recursive function calls.
"""

from __future__ import annotations

import math

from src.binomial_tree import BinomialTree
from src.dividend_exercise import (
    before_dividend_value, dividend_exercise_steps, offgrid_exercise_bound,
)
from src.hybrid_tree import HybridTree
from src.node import Node
from src.option import Option
from src.trinomial_tree import TrinomialTree

LatticeTree = BinomialTree | TrinomialTree | HybridTree


class BackwardPricer:
    """Price binomial, hybrid, and trinomial trees by backward induction."""

    def price(self, tree: LatticeTree, option: Option) -> float:
        """Build once, then work backward through reachable time layers."""
        tree.build()
        event_dividends = (
            dividend_exercise_steps(tree) if option.is_american else {}
        )
        levels = self._reachable_levels(tree.root, tree.nb_steps)

        for node in levels[-1]:
            node.option_value = option.payoff(node.price)

        discount_factor = math.exp(-tree.rate * tree.dt)

        for step in range(tree.nb_steps - 1, -1, -1):
            layer = levels[step]
            ex_dividend = event_dividends.get(step + 1)
            for node in layer:
                self._price_node(node, option, discount_factor, ex_dividend)

        value = tree.root.option_value
        if value is None:
            raise RuntimeError("Root option value was not calculated.")
        return value

    @staticmethod
    def _transition_value(
        node: Node, option: Option, ex_dividend: object,
    ) -> float:
        """Collect discounted-successor inputs without changing edge order."""
        up, down = node.next_up, node.next_down
        if up is None or down is None:
            raise RuntimeError("Incomplete lattice.")
        if node.up_probability is None or node.down_probability is None:
            raise RuntimeError("Missing transition probabilities.")
        if up.option_value is None or down.option_value is None:
            raise RuntimeError("Unpriced successor in backward induction.")

        # Preserve the original up/down evaluation order and event adjustment.
        expected = (
            node.up_probability
            * before_dividend_value(up, up.option_value, option, ex_dividend)
            + node.down_probability
            * before_dividend_value(down, down.option_value, option, ex_dividend)
        )
        mid = node.next_mid
        if mid is not None:
            if node.mid_probability is None:
                raise RuntimeError("Missing middle transition probability.")
            if mid.option_value is None:
                raise RuntimeError("Unpriced middle successor.")
            expected += (
                node.mid_probability
                * before_dividend_value(mid, mid.option_value, option, ex_dividend)
            )
        return expected

    @staticmethod
    def _price_node(
        node: Node, option: Option, discount_factor: float, ex_dividend: object,
    ) -> None:
        """Discount expected payoff and compare continuation with exercise."""
        expected = BackwardPricer._transition_value(
            node, option, ex_dividend,
        )
        hold_value = offgrid_exercise_bound(
            node, discount_factor * expected, option, ex_dividend,
        )
        node.option_value = option.value_at_node(
            spot=node.price, hold_value=hold_value,
        )

    @staticmethod
    def _next_level(
        current: list[Node], seen_earlier: set[int],
    ) -> dict[int, Node]:
        """Deduplicate a time layer and reject cross-layer node reuse."""
        following: dict[int, Node] = {}
        for node in current:
            if node.next_up is None or node.next_down is None:
                raise RuntimeError("Incomplete lattice.")
            for child in (node.next_up, node.next_down, node.next_mid):
                if child is None:
                    continue
                identity = id(child)
                if identity in seen_earlier:
                    raise RuntimeError(
                        "A lattice node belongs to more than one time layer."
                    )
                following[identity] = child
        return following

    @staticmethod
    def _reachable_levels(root: Node, steps: int) -> list[list[Node]]:
        """Visit shared nodes exactly once per time layer, without recursion."""
        levels: list[list[Node]] = [[root]]
        seen_earlier: set[int] = {id(root)}
        for _ in range(steps):
            following = BackwardPricer._next_level(
                levels[-1], seen_earlier,
            )
            if not following:
                raise RuntimeError("Missing next lattice layer.")
            levels.append(list(following.values()))
            seen_earlier.update(following)
        return levels
