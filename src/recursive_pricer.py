import math
from typing import Self

from src.binomial_tree import BinomialTree
from src.dividend_exercise import (
    DividendStepEvent, before_dividend_value, dividend_exercise_steps,
    offgrid_exercise_bound,
)
from src.node import Node
from src.option import Option
from src.trinomial_tree import TrinomialTree
from src.hybrid_tree import HybridTree


LatticeTree = BinomialTree | TrinomialTree | HybridTree


class RecursivePricer:
    """Prices vanilla options recursively on a recombining lattice."""

    def price(
        self: Self,
        tree: LatticeTree,
        option: Option
    ) -> float:
        """Build the lattice and return the option price."""
        tree.root.option_value = None

        # Recursive pricing requires the complete lattice in memory.
        tree.build()
        event_steps = dividend_exercise_steps(tree) if option.is_american else {}
        remaining_events = {
            tree.nb_steps - step + 1: amount
            for step, amount in event_steps.items()
        }

        discount_factor = math.exp(
            -tree.rate * tree.dt
        )

        return self._price_node(
            node=tree.root,
            remaining_steps=tree.nb_steps,
            option=option,
            discount_factor=discount_factor,
            event_dividends=remaining_events,
        )

    def _price_node(
        self: Self,
        node: Node,
        remaining_steps: int,
        option: Option,
        discount_factor: float,
        event_dividends: dict[int, DividendStepEvent],
    ) -> float:
        """Recursively price the option from one node."""

        # Recombining nodes are priced only once and then reused.
        if node.option_value is not None:
            return node.option_value

        # At maturity, the option value is simply its payoff.
        if remaining_steps == 0:
            node.option_value = option.payoff(
                node.price
            )

            return node.option_value

        if (
            node.next_up is None
            or node.next_down is None
        ):
            raise RuntimeError(
                "Incomplete lattice."
            )

        if (
            node.up_probability is None
            or node.down_probability is None
        ):
            raise RuntimeError(
                "Missing transition probabilities."
            )

        # Resolve the dividend adjustment on each outgoing transition.
        ex_dividend = event_dividends.get(remaining_steps)
        up_value = self._child_value(
            node.next_up, remaining_steps, option,
            discount_factor, event_dividends, ex_dividend,
        )
        down_value = self._child_value(
            node.next_down, remaining_steps, option,
            discount_factor, event_dividends, ex_dividend,
        )
        expected_option_value = (
            node.up_probability * up_value
            + node.down_probability * down_value
        )

        # A trinomial node adds its middle branch after the two outer ones.
        if node.next_mid is not None:
            if node.mid_probability is None:
                raise RuntimeError("Missing middle transition probability.")
            mid_value = self._child_value(
                node.next_mid, remaining_steps, option,
                discount_factor, event_dividends, ex_dividend,
            )
            expected_option_value += node.mid_probability * mid_value

        hold_value = (
            discount_factor
            * expected_option_value
        )

        # American options may exercise early at the current node.
        hold_value = offgrid_exercise_bound(
            node, hold_value, option, ex_dividend,
        )
        node.option_value = option.value_at_node(
            spot=node.price,
            hold_value=hold_value,
        )

        return node.option_value
    def _child_value(
        self: Self,
        child: Node,
        remaining_steps: int,
        option: Option,
        discount_factor: float,
        event_dividends: dict[int, DividendStepEvent],
        ex_dividend: DividendStepEvent | None,
    ) -> float:
        """Price a successor once and apply the same ex-date event rule."""
        child_value = self._price_node(
            node=child,
            remaining_steps=remaining_steps - 1,
            option=option,
            discount_factor=discount_factor,
            event_dividends=event_dividends,
        )
        return before_dividend_value(
            child, child_value, option, ex_dividend,
        )
