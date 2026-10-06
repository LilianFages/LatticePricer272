import math
from typing import Self

from src.dividend import CashDividend
from src.node import Node, TrunkNode
from src.transition_moments import TransitionMoments
from src.trinomial_probability import TrinomialProbabilitySolver


class TrinomialTree:
    """Represents a recombining trinomial lattice."""

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
        self._validate_inputs(
            spot,
            volatility,
            maturity,
            nb_steps,
            dividends,
            pruning_threshold,
        )

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

        self.pruning_threshold: float = (
            pruning_threshold
        )

        self.pruned_node_count: int = 0

        self.dt: float = (
            maturity
            / nb_steps
        )

        # Standard trinomial spacing used by the course.
        self.alpha: float = math.exp(
            volatility
            * math.sqrt(
                3.0 * self.dt
            )
        )

        (
            self.no_dividend_up_probability,
            self.no_dividend_mid_probability,
            self.no_dividend_down_probability,
        ) = self._compute_no_dividend_probabilities()

        self.root: TrunkNode = TrunkNode(
            spot
        )

        # The root is reached with certainty.
        self.root.reach_probability = 1.0

    @staticmethod
    def _validate_inputs(
        spot: float,
        volatility: float,
        maturity: float,
        nb_steps: int,
        dividends: list[CashDividend] | None,
        pruning_threshold: float
    ) -> None:
        """Validate the main lattice parameters."""
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

        if not (
            0.0
            <= pruning_threshold
            < 1.0
        ):
            raise ValueError(
                "Pruning threshold must belong to [0, 1)."
            )

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

    def _compute_no_dividend_probabilities(
        self: Self
    ) -> tuple[float, float, float]:
        """Return stable probabilities for a no-dividend transition."""
        if self.volatility == 0.0:
            return (
                0.0,
                1.0,
                0.0,
            )

        variance_ratio = math.expm1(
            self.volatility ** 2
            * self.dt
        )

        down_probability = (
            variance_ratio
            / (
                (1.0 - self.alpha)
                * (
                    1.0 / self.alpha ** 2
                    - 1.0
                )
            )
        )

        up_probability = (
            down_probability
            / self.alpha
        )

        mid_probability = (
            1.0
            - up_probability
            - down_probability
        )

        return (
            up_probability,
            mid_probability,
            down_probability,
        )

    def build(
        self: Self
    ) -> None:
        """Build the complete recombining trinomial lattice."""
        self.root.reach_probability = 1.0
        self.pruned_node_count = 0

        current_top: Node = self.root
        current_trunk: TrunkNode = self.root

        for step in range(
            1,
            self.nb_steps + 1,
        ):
            dividend = self._dividend_for_step(
                step
            )

            next_trunk_price = self._next_trunk_price(
                current_trunk,
                step,
                dividend,
            )

            top_exponent, bottom_exponent = (
                self._column_bounds(
                    current_top,
                    next_trunk_price,
                    step,
                    dividend,
                )
            )

            next_top, next_trunk = self._build_column(
                next_trunk_price,
                top_exponent,
                bottom_exponent,
                current_trunk,
            )

            self._connect_columns(
                current_top,
                next_top,
                step,
                dividend,
            )

            current_top = next_top
            current_trunk = next_trunk

    def _dividend_for_step(
        self: Self,
        step: int
    ) -> CashDividend | None:
        """Return the dividend occurring during one time step."""
        step_start = (
            step - 1
        ) * self.dt

        step_end = (
            step
            * self.dt
        )

        dividend_in_step: CashDividend | None = None

        for dividend in self.dividends:
            if (
                step_start
                < dividend.time
                <= step_end
            ):
                if dividend_in_step is not None:
                    raise ValueError(
                        "Only one dividend per time step is supported."
                    )

                dividend_in_step = dividend

        return dividend_in_step

    def _transition_moments(
        self: Self,
        node: Node,
        step: int,
        dividend: CashDividend | None
    ) -> tuple[float, float]:
        """Return local expected value and variance."""
        step_start = (
            step - 1
        ) * self.dt

        step_end = (
            step
            * self.dt
        )

        return TransitionMoments.compute(
            spot=node.price,
            rate=self.rate,
            volatility=self.volatility,
            step_start=step_start,
            step_end=step_end,
            dividend=dividend,
        )

    def _next_trunk_price(
        self: Self,
        current_trunk: TrunkNode,
        step: int,
        dividend: CashDividend | None
    ) -> float:
        """Return the next forward-adjusted trunk price."""
        expected_value, _ = self._transition_moments(
            current_trunk,
            step,
            dividend,
        )

        return expected_value

    @staticmethod
    def _bottom_node(
        top_node: Node
    ) -> Node:
        """Return the lowest node of a column."""
        current_node = top_node

        while current_node.lower_neighbor is not None:
            current_node = (
                current_node.lower_neighbor
            )

        return current_node

    def _closest_exponent(
        self: Self,
        expected_value: float,
        trunk_price: float
    ) -> int:
        """Return the lattice exponent closest to an expected value."""
        if expected_value <= 0.0:
            raise RuntimeError(
                "Expected stock price must remain positive."
            )

        if self.volatility == 0.0:
            return 0

        raw_exponent = math.log(
            expected_value
            / trunk_price
        ) / math.log(
            self.alpha
        )

        lower_exponent = math.floor(
            raw_exponent
        )

        upper_exponent = (
            lower_exponent
            + 1
        )

        lower_price = (
            trunk_price
            * self.alpha ** lower_exponent
        )

        upper_price = (
            trunk_price
            * self.alpha ** upper_exponent
        )

        if (
            abs(
                upper_price
                - expected_value
            )
            < abs(
                lower_price
                - expected_value
            )
        ):
            return upper_exponent

        return lower_exponent

    def _column_bounds(
        self: Self,
        current_top: Node,
        next_trunk_price: float,
        step: int,
        dividend: CashDividend | None
    ) -> tuple[int, int]:
        """Return dynamic upper and lower exponents for the next column."""
        if self.volatility == 0.0:
            return (
                step,
                -step,
            )

        current_bottom = self._bottom_node(
            current_top
        )

        top_expected, _ = self._transition_moments(
            current_top,
            step,
            dividend,
        )

        bottom_expected, _ = self._transition_moments(
            current_bottom,
            step,
            dividend,
        )

        top_middle_exponent = self._closest_exponent(
            top_expected,
            next_trunk_price,
        )

        bottom_middle_exponent = self._closest_exponent(
            bottom_expected,
            next_trunk_price,
        )

        # Each middle successor needs one upper and one lower neighbor.
        top_exponent = (
            top_middle_exponent
            + 1
        )

        bottom_exponent = (
            bottom_middle_exponent
            - 1
        )

        return (
            top_exponent,
            bottom_exponent,
        )

    def _node_price(
        self: Self,
        trunk_price: float,
        exponent: int
    ) -> float:
        """Return one geometric node price."""
        return (
            trunk_price
            * self.alpha ** exponent
        )

    def _build_column(
        self: Self,
        trunk_price: float,
        top_exponent: int,
        bottom_exponent: int,
        previous_trunk: TrunkNode
    ) -> tuple[Node, TrunkNode]:
        """Build one complete dynamically-sized column."""
        top_node: Node | None = None
        current_node: Node | None = None
        trunk_node: TrunkNode | None = None

        for exponent in range(
            top_exponent,
            bottom_exponent - 1,
            -1,
        ):
            if exponent == 0:
                trunk_candidate = TrunkNode(
                    trunk_price
                )

                trunk_candidate.previous_trunk = (
                    previous_trunk
                )

                next_node: Node = (
                    trunk_candidate
                )

                trunk_node = (
                    trunk_candidate
                )

            else:
                next_node = Node(
                    self._node_price(
                        trunk_price,
                        exponent,
                    )
                )

            if top_node is None:
                top_node = next_node

            if current_node is not None:
                current_node.lower_neighbor = (
                    next_node
                )

            current_node = next_node

        if top_node is None:
            raise RuntimeError(
                "Trinomial column was not created."
            )

        if trunk_node is None:
            raise RuntimeError(
                "Trunk node was not created."
            )

        return (
            top_node,
            trunk_node,
        )

    def _should_prune(
        self: Self,
        node: Node
    ) -> bool:
        """Return whether a node should use monomial branching."""
        return (
            self.pruning_threshold > 0.0
            and node.reach_probability
            < self.pruning_threshold
        )

    def _set_monomial_transition(
        self: Self,
        node: Node,
        next_mid: Node
    ) -> None:
        """Connect a negligible node only to its middle successor."""
        node.next_up = (
            next_mid
        )

        node.next_mid = (
            next_mid
        )

        node.next_down = (
            next_mid
        )

        node.up_probability = 0.0
        node.mid_probability = 1.0
        node.down_probability = 0.0

        self.pruned_node_count += 1

    @staticmethod
    def _add_reach_probability(
        starting_node: Node,
        ending_node: Node | None,
        transition_probability: float | None
    ) -> None:
        """Add one transition contribution to a successor probability."""
        if ending_node is None:
            return

        if transition_probability is None:
            return

        ending_node.reach_probability += (
            starting_node.reach_probability
            * transition_probability
        )

    def _propagate_reach_probability(
        self: Self,
        node: Node
    ) -> None:
        """Propagate root-to-node probability to all successors."""
        self._add_reach_probability(
            node,
            node.next_up,
            node.up_probability,
        )

        self._add_reach_probability(
            node,
            node.next_mid,
            node.mid_probability,
        )

        self._add_reach_probability(
            node,
            node.next_down,
            node.down_probability,
        )

    def _connect_columns(
        self: Self,
        current_top: Node,
        next_top: Node,
        step: int,
        dividend: CashDividend | None
    ) -> None:
        """Connect adjacent columns with local trinomial transitions."""
        current_node: Node | None = (
            current_top
        )

        upper_candidate: Node = (
            next_top
        )

        middle_candidate = (
            next_top.lower_neighbor
        )

        if middle_candidate is None:
            raise RuntimeError(
                "Incomplete next column."
            )

        while current_node is not None:
            expected_value, variance = (
                self._transition_moments(
                    current_node,
                    step,
                    dividend,
                )
            )

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
                    "Trinomial grid is too narrow above."
                )

            next_up, next_mid, next_down = (
                self._find_middle_successor(
                    expected_value,
                    upper_candidate,
                    middle_candidate,
                )
            )

            if self._should_prune(
                current_node
            ):
                self._set_monomial_transition(
                    current_node,
                    next_mid,
                )

            else:
                current_node.next_up = (
                    next_up
                )

                current_node.next_mid = (
                    next_mid
                )

                current_node.next_down = (
                    next_down
                )

                self._set_probabilities(
                    current_node,
                    next_up,
                    next_mid,
                    next_down,
                    expected_value,
                    variance,
                    dividend,
                )

            self._propagate_reach_probability(
                current_node
            )

            # Candidate movement must remain based on the geometric grid,
            # independently of whether the current node was pruned.
            upper_candidate = (
                next_up
            )

            middle_candidate = (
                next_mid
            )

            current_node = (
                current_node.lower_neighbor
            )

    @staticmethod
    def _find_middle_successor(
        expected_value: float,
        upper_candidate: Node,
        middle_candidate: Node
    ) -> tuple[Node, Node, Node]:
        """Return the closest middle node and its two neighbors."""
        upper_node = (
            upper_candidate
        )

        middle_node = (
            middle_candidate
        )

        while True:
            lower_node = (
                middle_node.lower_neighbor
            )

            if lower_node is None:
                raise RuntimeError(
                    "Trinomial grid is too narrow below."
                )

            if (
                abs(
                    lower_node.price
                    - expected_value
                )
                < abs(
                    middle_node.price
                    - expected_value
                )
            ):
                if lower_node.lower_neighbor is None:
                    raise RuntimeError(
                        "Trinomial grid is too narrow below."
                    )

                upper_node = (
                    middle_node
                )

                middle_node = (
                    lower_node
                )

                continue

            return (
                upper_node,
                middle_node,
                lower_node,
            )

    def _set_probabilities(
        self: Self,
        current_node: Node,
        next_up: Node,
        next_mid: Node,
        next_down: Node,
        expected_value: float,
        variance: float,
        dividend: CashDividend | None
    ) -> None:
        """Set local probabilities matching the target moments."""
        if self.volatility == 0.0:
            current_node.up_probability = 0.0
            current_node.mid_probability = 1.0
            current_node.down_probability = 0.0

            return

        # Without a dividend, the grid is aligned with the forward.
        # Stable closed-form probabilities avoid loss of precision
        # when the time step becomes very small.
        if dividend is None:
            current_node.up_probability = (
                self.no_dividend_up_probability
            )

            current_node.mid_probability = (
                self.no_dividend_mid_probability
            )

            current_node.down_probability = (
                self.no_dividend_down_probability
            )

            return

        # Dividend transitions require node-dependent probabilities.
        probabilities = TrinomialProbabilitySolver.solve(
            expected_value=expected_value,
            variance=variance,
            up_price=next_up.price,
            mid_price=next_mid.price,
            down_price=next_down.price,
        )

        if not probabilities.is_admissible():
            raise RuntimeError(
                "Invalid trinomial probabilities."
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