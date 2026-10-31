
from typing import Self

from src.contracts.dividend import CashDividend
from src.lattices.node import Node, TrunkNode
from src.lattices.trinomial_tree import TrinomialTree


class AllocationPrunedTrinomialTree(TrinomialTree):
    """Trinomial lattice with probability-based allocation pruning."""

    def __init__(
        self: Self,
        spot: float,
        rate: float,
        volatility: float,
        maturity: float,
        nb_steps: int,
        dividends: list[CashDividend] | None = None,
        pruning_threshold: float = 1e-18
    ) -> None:
        super().__init__(
            spot=spot,
            rate=rate,
            volatility=volatility,
            maturity=maturity,
            nb_steps=nb_steps,
            dividends=dividends,
            pruning_threshold=pruning_threshold,
        )

        self.allocated_node_count: int = 1
        self.terminal_top: Node = self.root

    def build(
        self: Self
    ) -> None:
        """Build the lattice and count allocated nodes."""
        self.allocated_node_count = 1
        self.terminal_top = self.root

        super().build()

    def _build_column(
        self: Self,
        trunk_price: float,
        top_exponent: int,
        bottom_exponent: int,
        previous_trunk: TrunkNode
    ) -> tuple[Node, TrunkNode]:
        """Allocate one column and update allocation statistics."""
        next_top, next_trunk = super()._build_column(
            trunk_price,
            top_exponent,
            bottom_exponent,
            previous_trunk,
        )

        self.allocated_node_count += (
            top_exponent
            - bottom_exponent
            + 1
        )

        self.terminal_top = next_top

        return next_top, next_trunk

    def _active_extremes(
        self: Self,
        current_top: Node
    ) -> tuple[Node, Node]:
        """Find the highest and lowest significant nodes."""
        active_top: Node | None = None
        active_bottom: Node | None = None

        most_likely_node = current_top
        maximum_probability = (
            current_top.reach_probability
        )

        current_node: Node | None = current_top

        while current_node is not None:
            probability = current_node.reach_probability

            if probability > maximum_probability:
                maximum_probability = probability
                most_likely_node = current_node

            if probability >= self.pruning_threshold:
                if active_top is None:
                    active_top = current_node

                active_bottom = current_node

            current_node = current_node.lower_neighbor

        # Always retain at least the most likely node.
        if active_top is None or active_bottom is None:
            return most_likely_node, most_likely_node

        return active_top, active_bottom

    def _column_bounds(
        self: Self,
        current_top: Node,
        next_trunk_price: float,
        step: int,
        dividend: CashDividend | None
    ) -> tuple[int, int]:
        """Allocate only the levels required by significant nodes."""
        if (
            self.pruning_threshold == 0.0
            or self.volatility == 0.0
        ):
            return super()._column_bounds(
                current_top,
                next_trunk_price,
                step,
                dividend,
            )

        active_top, active_bottom = self._active_extremes(
            current_top
        )

        top_expected, _ = self._transition_moments(
            active_top,
            step,
            dividend,
        )

        bottom_expected, _ = self._transition_moments(
            active_bottom,
            step,
            dividend,
        )

        # Project the significant extremes onto the next-column grid.
        top_middle_exponent = self._closest_exponent(
            top_expected,
            next_trunk_price,
        )

        bottom_middle_exponent = self._closest_exponent(
            bottom_expected,
            next_trunk_price,
        )

        # Each significant node needs an upper and lower successor.
        # Keep exponent zero so the trunk remains in the column.
        top_exponent = max(
            top_middle_exponent + 1,
            0,
        )

        bottom_exponent = min(
            bottom_middle_exponent - 1,
            0,
        )

        return top_exponent, bottom_exponent

    def _connect_significant_node(
        self: Self,
        current_node: Node,
        upper_candidate: Node | None,
        middle_candidate: Node,
        expected_value: float,
        variance: float,
        step: int,
        dividend: CashDividend | None
    ) -> None:
        """Connect a significant node within the allocated grid."""
        next_down = middle_candidate.lower_neighbor
        if upper_candidate is None or next_down is None:
            raise RuntimeError(
                "Allocation-pruned grid is too narrow "
                f"at step {step}."
            )

        self._link_full_transition(
            current_node,
            upper_candidate,
            middle_candidate,
            next_down,
            expected_value,
            variance,
            dividend,
        )

    def _connect_columns(
        self: Self,
        current_top: Node,
        next_top: Node,
        step: int,
        dividend: CashDividend | None
    ) -> None:
        """Connect nodes while allowing truncated column boundaries."""
        if (
            self.pruning_threshold == 0.0
            or self.volatility == 0.0
        ):
            super()._connect_columns(
                current_top,
                next_top,
                step,
                dividend,
            )
            return

        current_node: Node | None = current_top

        # Move down the next column as expected prices decrease.
        middle_candidate: Node = next_top
        upper_candidate: Node | None = None

        while current_node is not None:
            expected_value, variance = (
                self._transition_moments(
                    current_node,
                    step,
                    dividend,
                )
            )

            # Move the candidate monotonically toward the local expectation.
            while middle_candidate.lower_neighbor is not None:
                lower_candidate = (
                    middle_candidate.lower_neighbor
                )

                if (
                    abs(lower_candidate.price - expected_value)
                    >= abs(middle_candidate.price - expected_value)
                ):
                    break

                upper_candidate = middle_candidate
                middle_candidate = lower_candidate

            if self._should_prune(current_node):
                # Negligible nodes can be redirected to a boundary.
                self._set_monomial_transition(
                    current_node,
                    middle_candidate,
                )

            else:
                self._connect_significant_node(
                    current_node,
                    upper_candidate,
                    middle_candidate,
                    expected_value,
                    variance,
                    step,
                    dividend,
                )

            # Keep the reachability mass consistent with the selected branches.
            self._propagate_reach_probability(
                current_node
            )

            current_node = current_node.lower_neighbor
