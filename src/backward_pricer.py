import math
from typing import Self

from src.binomial_tree import BinomialTree
from src.node import Node
from src.option import Option


class BackwardPricer:
    """Prices vanilla options backward on a binomial lattice."""

    def price(
        self: Self,
        tree: BinomialTree,
        option: Option,
    ) -> float:
        """Price an option by backward induction."""
        last_trunk = tree.build_trunk()

        # Start from the complete maturity column.
        next_top = tree.build_column_backward(
            trunk_node=last_trunk,
            step=tree.nb_steps,
        )

        self._set_terminal_payoffs(
            next_top,
            option,
        )

        discount_factor = math.exp(
            -tree.rate * tree.dt
        )

        current_trunk = last_trunk

        # Rebuild and price one column at a time toward the root.
        for step in range(
            tree.nb_steps - 1,
            -1,
            -1,
        ):
            previous_trunk = (
                current_trunk.previous_trunk
            )

            if previous_trunk is None:
                raise RuntimeError(
                    "Incomplete trunk."
                )

            current_top = tree.build_column_backward(
                trunk_node=previous_trunk,
                step=step,
                next_top=next_top,
            )

            self._price_column(
                current_top,
                tree,
                option,
                discount_factor,
            )

            # The priced column becomes the reference for the next step.
            next_top = current_top
            current_trunk = previous_trunk

        if tree.root.option_value is None:
            raise RuntimeError(
                "Root option value was not calculated."
            )

        return tree.root.option_value

    @staticmethod
    def _set_terminal_payoffs(
        top_node: Node,
        option: Option,
    ) -> None:
        """Set option payoffs on the maturity column."""
        current_node: Node | None = top_node

        # Traverse the terminal column through vertical links.
        while current_node is not None:
            current_node.option_value = (
                option.payoff(
                    current_node.price
                )
            )

            current_node = (
                current_node.lower_neighbor
            )

    @staticmethod
    def _price_column(
        top_node: Node,
        tree: BinomialTree,
        option: Option,
        discount_factor: float,
    ) -> None:
        """Price all nodes of one column."""
        current_node: Node | None = top_node

        while current_node is not None:
            next_up = current_node.next_up
            next_down = current_node.next_down

            # Both successors must belong to the already-priced next column.
            if next_up is None or next_down is None:
                raise RuntimeError(
                    "Incomplete lattice connections."
                )

            if (
                next_up.option_value is None
                or next_down.option_value is None
            ):
                raise RuntimeError(
                    "Next column has not been priced."
                )

            # Risk-neutral continuation value discounted over one time step.
            hold_value = discount_factor * (
                tree.up_probability
                * next_up.option_value
                + tree.down_probability
                * next_down.option_value
            )

            current_node.option_value = (
                option.value_at_node(
                    spot=current_node.price,
                    hold_value=hold_value,
                )
            )

            current_node = (
                current_node.lower_neighbor
            )