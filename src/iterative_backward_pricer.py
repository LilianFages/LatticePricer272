"""Iterative backward induction on the existing recombining pricing graph.

This algorithm shares the tree construction and transition probabilities with
RecursivePricer. It uses no Python recursion, but it materializes the entire
graph; it is NOT the memory-efficient column-rebuilding Binomial BackwardPricer.
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


class IterativeBackwardPricer:
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
                next_up = node.next_up
                next_down = node.next_down
                if next_up is None or next_down is None:
                    raise RuntimeError("Incomplete lattice.")
                if node.up_probability is None or node.down_probability is None:
                    raise RuntimeError("Missing transition probabilities.")
                if next_up.option_value is None or next_down.option_value is None:
                    raise RuntimeError("Unpriced successor in backward induction.")

                expected_value = (
                    node.up_probability
                    * before_dividend_value(
                        next_up, next_up.option_value, option, ex_dividend,
                    )
                    + node.down_probability
                    * before_dividend_value(
                        next_down, next_down.option_value, option, ex_dividend,
                    )
                )
                if node.next_mid is not None:
                    if node.mid_probability is None:
                        raise RuntimeError("Missing middle transition probability.")
                    if node.next_mid.option_value is None:
                        raise RuntimeError("Unpriced middle successor.")
                    expected_value += (
                        node.mid_probability
                        * before_dividend_value(
                            node.next_mid, node.next_mid.option_value,
                            option, ex_dividend,
                        )
                    )
                hold_value = discount_factor * expected_value
                hold_value = offgrid_exercise_bound(
                    node, hold_value, option, ex_dividend,
                )
                node.option_value = option.value_at_node(
                    spot=node.price,
                    hold_value=hold_value,
                )

        value = tree.root.option_value
        if value is None:
            raise RuntimeError("Root option value was not calculated.")
        return value

    @staticmethod
    def _reachable_levels(root: Node, steps: int) -> list[list[Node]]:
        """Visit shared nodes once per level, without recursive traversal.

        Objects referenced by an earlier level may not be reused at a later
        time; doing so would violate the acyclic time-layered lattice model.
        """
        levels: list[list[Node]] = [[root]]
        earlier: set[int] = {id(root)}

        for _ in range(steps):
            following: dict[int, Node] = {}
            for node in levels[-1]:
                if node.next_up is None or node.next_down is None:
                    raise RuntimeError("Incomplete lattice.")
                for child in (node.next_up, node.next_down, node.next_mid):
                    if child is None:
                        continue
                    child_id = id(child)
                    if child_id in earlier:
                        raise RuntimeError(
                            "A lattice node belongs to more than one time layer."
                        )
                    following[child_id] = child

            if not following:
                raise RuntimeError("Missing next lattice layer.")
            levels.append(list(following.values()))
            earlier.update(following)

        return levels
