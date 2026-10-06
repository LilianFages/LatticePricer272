import math
from typing import Self

from src.binomial_tree import BinomialTree
from src.node import Node
from src.option import Option


class RecursivePricer:
    """Prices vanilla options recursively on a binomial lattice."""

    def price(
        self: Self,
        tree: BinomialTree,
        option: Option,
    ) -> float:
        """Build the lattice and return the option price."""
        # Clear a possible value left by a previous pricing run.
        tree.root.option_value = None

        # Recursive pricing requires the complete lattice in memory.
        tree.build()

        discount_factor = math.exp(
            -tree.rate * tree.dt
        )

        return self._price_node(
            node=tree.root,
            remaining_steps=tree.nb_steps,
            tree=tree,
            option=option,
            discount_factor=discount_factor,
        )

    def _price_node(
        self: Self,
        node: Node,
        remaining_steps: int,
        tree: BinomialTree,
        option: Option,
        discount_factor: float,
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

        # An intermediate node must have both binomial successors.
        if (
            node.next_up is None
            or node.next_down is None
        ):
            raise RuntimeError(
                "Incomplete lattice."
            )

        # Recursively price the two successor nodes.
        up_value = self._price_node(
            node=node.next_up,
            remaining_steps=remaining_steps - 1,
            tree=tree,
            option=option,
            discount_factor=discount_factor,
        )

        down_value = self._price_node(
            node=node.next_down,
            remaining_steps=remaining_steps - 1,
            tree=tree,
            option=option,
            discount_factor=discount_factor,
        )

        # Compute the discounted risk-neutral continuation value.
        hold_value = discount_factor * (
            tree.up_probability * up_value
            + tree.down_probability * down_value
        )

        # American options may exercise early at the current node.
        node.option_value = option.value_at_node(
            spot=node.price,
            hold_value=hold_value,
        )

        return node.option_value