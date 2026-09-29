import math

from src.node import Node, TrunkNode


class BinomialTree:
    """Represents a recombining binomial lattice."""

    def __init__(
        self,
        spot: float,
        rate: float,
        volatility: float,
        maturity: float,
        nb_steps: int,
    ) -> None:
        self._validate_inputs(
            spot,
            volatility,
            maturity,
            nb_steps,
        )

        self.spot: float = spot
        self.rate: float = rate
        self.volatility: float = volatility
        self.maturity: float = maturity
        self.nb_steps: int = nb_steps

        self.dt: float = maturity / nb_steps

        # Controls the spacing between price levels
        self.alpha: float = self._compute_alpha()

        self.up_probability: float = 1.0 / (
            self.alpha + 1.0
        )
        self.down_probability: float = (
            1.0 - self.up_probability
        )

        self.root: TrunkNode = TrunkNode(spot)

    @staticmethod
    def _validate_inputs(
        spot: float,
        volatility: float,
        maturity: float,
        nb_steps: int,
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

    def _compute_alpha(self) -> float:
        """Compute the lattice spacing factor alpha."""
        variance_factor = math.exp(
            self.volatility ** 2 * self.dt
        )

        middle_term = (
            1.0 + variance_factor
        ) / 2.0

        return middle_term + math.sqrt(
            middle_term ** 2 - 1.0
        )

    def _node_price(
        self,
        step: int,
        position: int,
    ) -> float:
        """Compute the price of one node in a column."""
        growth_factor = math.exp(
            self.rate * self.dt * step
        )

        alpha_power = (
            step - 2 * position
        )

        return (
            self.spot
            * growth_factor
            * self.alpha ** alpha_power
        )

    def build(self) -> None:
        """Build the complete recombining binomial lattice."""
        current_top: Node = self.root

        for _ in range(self.nb_steps):
            current_top = self._build_next_column(
                current_top
            )

    def build_trunk(self) -> TrunkNode:
        """Build the binomial trunk and return its last node."""
        current_trunk = self.root

        growth_factor = math.exp(
            self.rate * self.dt
        )

        for step in range(self.nb_steps):
            if step % 2 == 0:
                next_price = (
                    current_trunk.price
                    * growth_factor
                    * self.alpha
                )

                next_trunk = TrunkNode(
                    next_price
                )

                current_trunk.next_up = (
                    next_trunk
                )

            else:
                next_price = (
                    current_trunk.price
                    * growth_factor
                    / self.alpha
                )

                next_trunk = TrunkNode(
                    next_price
                )

                current_trunk.next_down = (
                    next_trunk
                )

            next_trunk.previous_trunk = (
                current_trunk
            )

            current_trunk = next_trunk

        return current_trunk

    def build_column_backward(
        self,
        trunk_node: TrunkNode,
        step: int,
        next_top: Node | None = None,
    ) -> Node:
        """Build one column around its trunk node."""
        trunk_position = step // 2

        if trunk_position == 0:
            top_node: Node = trunk_node
        else:
            top_node = Node(
                self._node_price(
                    step,
                    0,
                )
            )

        current_node = top_node
        next_node = next_top

        for position in range(step + 1):
            if position > 0:
                if position == trunk_position:
                    lower_node: Node = trunk_node
                else:
                    lower_node = Node(
                        self._node_price(
                            step,
                            position,
                        )
                    )

                current_node.lower_neighbor = (
                    lower_node
                )

                current_node = lower_node

            if next_node is not None:
                lower_next = (
                    next_node.lower_neighbor
                )

                if lower_next is None:
                    raise RuntimeError(
                        "Incomplete next column."
                    )

                current_node.next_up = next_node
                current_node.next_down = lower_next

                next_node = lower_next

        return top_node

    def _build_next_column(
        self,
        current_top: Node,
    ) -> Node:
        """Build one new column from the current one."""
        growth_factor = math.exp(
            self.rate * self.dt
        )

        next_top = Node(
            current_top.price
            * growth_factor
            * self.alpha
        )

        current_top.next_up = next_top

        current_node = current_top
        next_node = next_top

        while current_node is not None:
            down_node = Node(
                current_node.price
                * growth_factor
                / self.alpha
            )

            current_node.next_down = down_node
            next_node.lower_neighbor = down_node

            current_node = (
                current_node.lower_neighbor
            )

            if current_node is not None:
                current_node.next_up = down_node

            next_node = down_node

        return next_top