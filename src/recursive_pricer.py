import math

from src.binomial_tree import BinomialTree
from src.node import Node
from src.option import Option


class RecursivePricer:
    """Prices European options recursively on a binomial lattice."""

    def price(
        self,
        tree: BinomialTree,
        option: Option,
    ) -> float:
        """Build the lattice and return the option price."""
        tree.root.option_value = None
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
        self,
        node: Node,
        remaining_steps: int,
        tree: BinomialTree,
        option: Option,
        discount_factor: float,
    ) -> float:
        """Recursively price the option from one node."""
        if node.option_value is not None:
            return node.option_value

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

        node.option_value = discount_factor * (
            tree.up_probability * up_value
            + tree.down_probability * down_value
        )

        return node.option_value