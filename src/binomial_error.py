import math


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
        if volatility == 0.0:
            return 0.0

        numerator = (
            math.exp(
                rate * maturity
            )
            * volatility ** 2
            * maturity
        )

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

        return (
            numerator
            / denominator
        )

    def relative_gap(
        self,
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

        return (
            constant
            / nb_steps
        )

    def steps_for_relative_precision(
        self,
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