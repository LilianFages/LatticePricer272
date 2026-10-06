import math
from typing import Self


class BinomialErrorEstimator:
    """Estimates the theoretical binomial pricing gap."""

    @staticmethod
    def _validate_market_inputs(
        volatility: float,
        maturity: float,
    ) -> None:
        """Validate inputs used by the error formula."""
        if volatility < 0.0:
            raise ValueError(
                "Volatility cannot be negative."
            )

        if maturity <= 0.0:
            raise ValueError(
                "Maturity must be strictly positive."
            )

    @staticmethod
    def _relative_gap_constant(
        rate: float,
        volatility: float,
        maturity: float,
    ) -> float:
        """Return the constant in relative_gap = C / N."""
        # With no volatility, the theoretical discretization gap vanishes.
        if volatility == 0.0:
            return 0.0

        # The numerator combines growth and variance over maturity.
        numerator = (
            math.exp(
                rate * maturity
            )
            * volatility ** 2
            * maturity
        )

        # The denominator normalizes by the dispersion of the diffusion.
        denominator = (
            2.0
            * math.sqrt(
                2.0 * math.pi
            )
            * math.sqrt(
                math.exp(
                    volatility ** 2
                    * maturity
                )
                - 1.0
            )
        )

        # This constant determines the asymptotic size of the pricing gap.
        return (
            numerator
            / denominator
        )

    def relative_gap(
        self: Self,
        nb_steps: int,
        rate: float,
        volatility: float,
        maturity: float,
    ) -> float:
        """Estimate Gap / Spot for a given number of steps."""
        self._validate_market_inputs(
            volatility,
            maturity,
        )

        # A binomial lattice requires a strictly positive integer step count.
        if (
            type(nb_steps) is not int
            or nb_steps < 1
        ):
            raise ValueError(
                "Number of steps must be a positive integer."
            )

        constant = self._relative_gap_constant(
            rate,
            volatility,
            maturity,
        )

        # The theoretical relative gap decreases at first order in 1 / N.
        return (
            constant
            / nb_steps
        )

    def steps_for_relative_precision(
        self: Self,
        target_precision: float,
        rate: float,
        volatility: float,
        maturity: float,
    ) -> int:
        """Return the steps required for a target relative gap."""
        self._validate_market_inputs(
            volatility,
            maturity,
        )

        if target_precision <= 0.0:
            raise ValueError(
                "Target precision must be strictly positive."
            )

        constant = self._relative_gap_constant(
            rate,
            volatility,
            maturity,
        )

        # With zero volatility, one step is sufficient for this criterion.
        if constant == 0.0:
            return 1

        required_steps = math.ceil(
            constant
            / target_precision
        )

        return max(
            required_steps,
            1,
        )