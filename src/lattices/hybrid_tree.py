import math
from typing import Self

from src.contracts.dividend import CashDividend
from src.contracts.dividend_timing import dividend_for_lattice_step
from src.lattices.node import (
    Node, TrunkNode, bottom_node, middle_successors,
    nearest_geometric_exponent,
)
from src.lattices.transition_moments import TransitionMoments
from src.lattices.trinomial_probability import TrinomialProbabilitySolver


class HybridTree:
    """Represents a binomial lattice with trinomial dividend steps."""

    def __init__(
        self: Self,
        spot: float,
        rate: float,
        volatility: float,
        maturity: float,
        nb_steps: int,
        dividends: list[CashDividend] | None = None
    ) -> None:
        self._validate_inputs(
            spot,
            volatility,
            maturity,
            nb_steps,
            dividends,
        )

        # Preserve contract inputs before deriving the one-step lattice geometry.
        self.spot: float = spot
        self.rate: float = rate
        self.volatility: float = volatility
        self.maturity: float = maturity
        self.nb_steps: int = nb_steps

        self.dividends: list[CashDividend] = (
            []
            if dividends is None
            else dividends
        )

        self.dt: float = (
            maturity
            / nb_steps
        )

        # Normal steps use the exact binomial spacing.
        self.binomial_alpha: float = (
            self._compute_binomial_alpha()
        )

        # Adjacent nodes in a binomial column differ by alpha squared.
        self.level_ratio: float = (
            self.binomial_alpha ** 2
        )

        self.up_probability: float = (
            1.0
            / (
                self.binomial_alpha
                + 1.0
            )
        )

        self.down_probability: float = (
            1.0
            - self.up_probability
        )

        self.root: TrunkNode = TrunkNode(
            spot
        )

    @staticmethod
    def _validate_inputs(
        spot: float,
        volatility: float,
        maturity: float,
        nb_steps: int,
        dividends: list[CashDividend] | None
    ) -> None:
        """Validate the hybrid lattice parameters."""
        if spot <= 0.0:
            raise ValueError(
                "Spot must be strictly positive."
            )

        if volatility < 0.0:
            raise ValueError(
                "Volatility cannot be negative."
            )

        if maturity <= 0.0:
            raise ValueError(
                "Maturity must be strictly positive."
            )

        if type(nb_steps) is not int or nb_steps < 1:
            raise ValueError(
                "Number of steps must be a positive integer."
            )

        # Event checks apply only when cash dividends are scheduled.
        if dividends is None:
            return

        for dividend in dividends:
            if dividend.time <= 0.0:
                raise ValueError(
                    "Dividend time must be strictly positive."
                )

            if dividend.time > maturity:
                raise ValueError(
                    "Dividend must occur before option maturity."
                )

    def _compute_binomial_alpha(
        self: Self
    ) -> float:
        """Return the exact binomial spacing factor."""
        variance_factor = math.exp(
            self.volatility ** 2
            * self.dt
        )

        middle_term = (
            1.0
            + variance_factor
        ) / 2.0

        return (
            middle_term
            + math.sqrt(
                middle_term ** 2
                - 1.0
            )
        )

    def build(
        self: Self
    ) -> None:
        """Build the complete hybrid lattice."""
        current_top: Node = self.root

        # Track the theoretical forward independently of lattice nodes.
        current_forward = self.spot

        for step in range(
            1,
            self.nb_steps + 1,
        ):
            dividend = self._dividend_for_step(
                step
            )

            next_forward = self._next_forward(
                current_forward,
                step,
                dividend,
            )

            # Ordinary dates retain the cheaper recombining binomial transition.
            if dividend is None:
                current_top = (
                    self._build_binomial_column(
                        current_top
                    )
                )

            else:
                current_top = (
                    self._build_dividend_column(
                        current_top,
                        next_forward,
                        step,
                        dividend,
                    )
                )

            current_forward = next_forward

    def _dividend_for_step(
        self: Self,
        step: int
    ) -> CashDividend | None:
        """Find the single event for one right-closed lattice interval."""
        return dividend_for_lattice_step(
            self.dividends, step, self.maturity, self.nb_steps,
        )

    def _transition_moments(
        self: Self,
        node: Node,
        step: int,
        dividend: CashDividend | None
    ) -> tuple[float, float]:
        """Return moments on the same exact lattice time interval."""
        return TransitionMoments.at_lattice_step(
            spot=node.price, rate=self.rate, volatility=self.volatility,
            step=step, dt=self.dt, dividend=dividend,
        )

    def _next_forward(
        self: Self,
        current_forward: float,
        step: int,
        dividend: CashDividend | None
    ) -> float:
        """Return the same forward-adjusted central price."""
        expected_value, _ = TransitionMoments.at_lattice_step(
            spot=current_forward, rate=self.rate, volatility=self.volatility,
            step=step, dt=self.dt, dividend=dividend,
        )
        return expected_value

    def _set_binomial_probabilities(
        self: Self,
        node: Node
    ) -> None:
        """Store normal binomial probabilities on one node."""
        node.up_probability = (
            self.up_probability
        )

        node.mid_probability = None

        node.down_probability = (
            self.down_probability
        )

    def _build_binomial_column(
        self: Self,
        current_top: Node
    ) -> Node:
        """Build one normal recombining binomial step."""
        growth_factor = math.exp(
            self.rate
            * self.dt
        )

        next_top = Node(
            current_top.price
            * growth_factor
            * self.binomial_alpha
        )

        current_top.next_up = (
            next_top
        )

        current_top.next_mid = None

        current_node: Node | None = current_top
        next_node = next_top

        # Build both successors and reuse shared nodes to preserve recombination.
        while current_node is not None:
            down_node = Node(
                current_node.price
                * growth_factor
                / self.binomial_alpha
            )

            current_node.next_down = (
                down_node
            )

            current_node.next_mid = None

            next_node.lower_neighbor = (
                down_node
            )

            self._set_binomial_probabilities(
                current_node
            )

            # Adjacent paths recombine on the same next-column node.
            current_node = (
                current_node.lower_neighbor
            )

            if current_node is not None:
                current_node.next_up = (
                    down_node
                )

            next_node = down_node

        return next_top

    @staticmethod
    def _bottom_node(top_node: Node) -> Node:
        """Return the lowest node of a column."""
        return bottom_node(top_node)

    def _closest_exponent(
        self: Self, expected_value: float, forward_price: float,
    ) -> int:
        """Return the nearest geometric bridge level."""
        return nearest_geometric_exponent(
            expected_value, forward_price, self.level_ratio,
        )

    def _dividend_column_bounds(
        self: Self,
        current_top: Node,
        next_forward: float,
        step: int,
        dividend: CashDividend
    ) -> tuple[int, int]:
        """Return dynamic bounds of a trinomial bridge column."""
        current_bottom = self._bottom_node(
            current_top
        )

        top_expected, _ = (
            self._transition_moments(
                current_top,
                step,
                dividend,
            )
        )

        bottom_expected, _ = (
            self._transition_moments(
                current_bottom,
                step,
                dividend,
            )
        )

        top_middle = self._closest_exponent(
            top_expected,
            next_forward,
        )

        bottom_middle = self._closest_exponent(
            bottom_expected,
            next_forward,
        )

        # Each middle branch requires one neighboring node on each side.
        top_exponent = max(
            top_middle + 1,
            0,
        )

        bottom_exponent = min(
            bottom_middle - 1,
            0,
        )

        return (
            top_exponent,
            bottom_exponent,
        )

    def _build_bridge_column(
        self: Self,
        forward_price: float,
        top_exponent: int,
        bottom_exponent: int
    ) -> Node:
        """Build one geometric column on the hybrid bridge grid."""
        top_node: Node | None = None
        current_node: Node | None = None

        for exponent in range(
            top_exponent,
            bottom_exponent - 1,
            -1,
        ):
            price = (
                forward_price
                * self.level_ratio ** exponent
            )

            # The zero exponent anchors each bridge column at the forward.
            if exponent == 0:
                next_node: Node = TrunkNode(
                    price
                )
            else:
                next_node = Node(
                    price
                )

            if top_node is None:
                top_node = next_node

            if current_node is not None:
                current_node.lower_neighbor = (
                    next_node
                )

            current_node = next_node

        # An empty bridge is invalid and cannot support transition calibration.
        if top_node is None:
            raise RuntimeError(
                "Hybrid bridge column was not created."
            )

        return top_node

    def _build_dividend_column(
        self: Self,
        current_top: Node,
        next_forward: float,
        step: int,
        dividend: CashDividend
    ) -> Node:
        """Build one trinomial transition for a dividend step."""
        if self.volatility == 0.0:
            return self._build_zero_volatility_dividend_step(
                current_top,
                next_forward,
            )

        top_exponent, bottom_exponent = (
            self._dividend_column_bounds(
                current_top,
                next_forward,
                step,
                dividend,
            )
        )

        # Allocate the bridge grid before connecting its conditional transitions.
        next_top = self._build_bridge_column(
            next_forward,
            top_exponent,
            bottom_exponent,
        )

        self._connect_dividend_column(
            current_top,
            next_top,
            step,
            dividend,
        )

        return next_top

    @staticmethod
    def _find_middle_successor(
        expected_value: float, upper_candidate: Node, middle_candidate: Node,
    ) -> tuple[Node, Node, Node]:
        """Find the nearest middle successor and its two neighbors."""
        return middle_successors(
            expected_value, upper_candidate, middle_candidate,
            grid_name="Hybrid",
        )

    def _connect_dividend_column(
        self: Self,
        current_top: Node,
        next_top: Node,
        step: int,
        dividend: CashDividend
    ) -> None:
        """Connect one dividend step with three local branches."""
        current_node: Node | None = current_top

        upper_candidate: Node = next_top

        middle_candidate = (
            next_top.lower_neighbor
        )

        if middle_candidate is None:
            raise RuntimeError(
                "Incomplete hybrid bridge column."
            )

        while current_node is not None:
            expected_value, variance = (
                self._transition_moments(
                    current_node,
                    step,
                    dividend,
                )
            )

            # The upper candidate must bracket the next conditional expectation.
            if (
                abs(
                    upper_candidate.price
                    - expected_value
                )
                < abs(
                    middle_candidate.price
                    - expected_value
                )
            ):
                raise RuntimeError(
                    "Hybrid grid is too narrow above."
                )

            next_up, next_mid, next_down = (
                self._find_middle_successor(
                    expected_value,
                    upper_candidate,
                    middle_candidate,
                )
            )

            # Bind the three bridge successors before solving their probabilities.
            current_node.next_up = (
                next_up
            )

            current_node.next_mid = (
                next_mid
            )

            current_node.next_down = (
                next_down
            )

            self._set_trinomial_probabilities(
                current_node,
                next_up,
                next_mid,
                next_down,
                expected_value,
                variance,
            )

            # Expected values decrease as we move down the current column.
            upper_candidate = next_up
            middle_candidate = next_mid

            current_node = (
                current_node.lower_neighbor
            )

    @staticmethod
    def _set_trinomial_probabilities(
        current_node: Node,
        next_up: Node,
        next_mid: Node,
        next_down: Node,
        expected_value: float,
        variance: float
    ) -> None:
        """Store local probabilities of one dividend transition."""
        probabilities = TrinomialProbabilitySolver.solve(
            expected_value=expected_value,
            variance=variance,
            up_price=next_up.price,
            mid_price=next_mid.price,
            down_price=next_down.price,
        )

        if not probabilities.is_admissible():
            raise RuntimeError(
                "Invalid hybrid trinomial probabilities."
            )

        current_node.up_probability = (
            probabilities.up
        )

        current_node.mid_probability = (
            probabilities.mid
        )

        current_node.down_probability = (
            probabilities.down
        )

    @staticmethod
    def _build_zero_volatility_dividend_step(
        current_top: Node,
        next_forward: float
    ) -> Node:
        """Build the deterministic dividend transition."""
        next_node = TrunkNode(
            next_forward
        )

        current_node: Node | None = current_top

        while current_node is not None:
            current_node.next_up = next_node
            current_node.next_mid = next_node
            current_node.next_down = next_node

            current_node.up_probability = 0.0
            current_node.mid_probability = 1.0
            current_node.down_probability = 0.0

            current_node = (
                current_node.lower_neighbor
            )

        return next_node