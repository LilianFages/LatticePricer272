import math

from src.node import Node


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

        self.root: Node = Node(spot)

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

    def build(self) -> None:
        """Build the complete recombining binomial lattice."""
        current_top = self.root

        for _ in range(self.nb_steps):
            current_top = self._build_next_column(
                current_top
            )

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