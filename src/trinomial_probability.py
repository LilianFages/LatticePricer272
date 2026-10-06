class TransitionProbabilities:
    """Stores the three probabilities of a trinomial transition."""

    def __init__(
        self,
        up: float,
        mid: float,
        down: float
    ) -> None:
        self.up: float = up
        self.mid: float = mid
        self.down: float = down

    def total(self) -> float:
        """Return the sum of the three probabilities."""
        return (
            self.up
            + self.mid
            + self.down
        )

    def is_admissible(
        self,
        tolerance: float = 1e-12
    ) -> bool:
        """Check whether all probabilities belong to [0, 1]."""
        return (
            -tolerance <= self.up <= 1.0 + tolerance
            and -tolerance <= self.mid <= 1.0 + tolerance
            and -tolerance <= self.down <= 1.0 + tolerance
        )


class TrinomialProbabilitySolver:
    """Solves trinomial probabilities from two target moments."""

    @staticmethod
    def solve(
        expected_value: float,
        variance: float,
        up_price: float,
        mid_price: float,
        down_price: float
    ) -> TransitionProbabilities:
        """Return probabilities matching mean and variance."""
        TrinomialProbabilitySolver._validate_inputs(
            expected_value,
            variance,
            up_price,
            mid_price,
            down_price,
        )

        # Subtract the middle-node contribution from both moment equations.
        first_moment_target = (
            expected_value
            - mid_price
        )

        second_moment_target = (
            variance
            + expected_value ** 2
            - mid_price ** 2
        )

        up_first_shift = (
            up_price
            - mid_price
        )
        down_first_shift = (
            down_price
            - mid_price
        )

        up_second_shift = (
            up_price ** 2
            - mid_price ** 2
        )
        down_second_shift = (
            down_price ** 2
            - mid_price ** 2
        )

        determinant = (
            up_first_shift
            * down_second_shift
            - down_first_shift
            * up_second_shift
        )

        up_probability = (
            first_moment_target
            * down_second_shift
            - down_first_shift
            * second_moment_target
        ) / determinant

        down_probability = (
            up_first_shift
            * second_moment_target
            - first_moment_target
            * up_second_shift
        ) / determinant

        mid_probability = (
            1.0
            - up_probability
            - down_probability
        )

        return TransitionProbabilities(
            up=up_probability,
            mid=mid_probability,
            down=down_probability,
        )

    @staticmethod
    def _validate_inputs(
        expected_value: float,
        variance: float,
        up_price: float,
        mid_price: float,
        down_price: float
    ) -> None:
        """Validate the inputs of the trinomial system."""
        if expected_value <= 0.0:
            raise ValueError(
                "Expected value must be strictly positive."
            )

        if variance < 0.0:
            raise ValueError(
                "Variance cannot be negative."
            )

        if down_price <= 0.0:
            raise ValueError(
                "Node prices must be strictly positive."
            )

        if not (
            up_price
            > mid_price
            > down_price
        ):
            raise ValueError(
                "Node prices must satisfy up > mid > down."
            )