import math

from src.option import CallOption, Option, PutOption


class BlackScholesPricer:
    """Prices European vanilla options with Black-Scholes."""

    @staticmethod
    def _normal_cdf(
        value: float,
    ) -> float:
        """Return the standard normal cumulative distribution."""
        return 0.5 * (
            1.0
            + math.erf(
                value / math.sqrt(2.0)
            )
        )

    @staticmethod
    def _validate_inputs(
        spot: float,
        volatility: float,
        maturity: float,
    ) -> None:
        """Validate Black-Scholes market inputs."""
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

    @staticmethod
    def _compute_d1_d2(
        spot: float,
        strike: float,
        rate: float,
        volatility: float,
        maturity: float,
    ) -> tuple[float, float]:
        """Compute Black-Scholes d1 and d2."""
        volatility_sqrt_time = (
            volatility
            * math.sqrt(maturity)
        )

        d1 = (
            math.log(
                spot / strike
            )
            + (
                rate
                + 0.5 * volatility ** 2
            )
            * maturity
        ) / volatility_sqrt_time

        d2 = (
            d1
            - volatility_sqrt_time
        )

        return d1, d2

    def price(
        self,
        option: Option,
        spot: float,
        rate: float,
        volatility: float,
        maturity: float,
    ) -> float:
        """Return the Black-Scholes European option price."""
        self._validate_inputs(
            spot,
            volatility,
            maturity,
        )

        if option.is_american:
            raise ValueError(
                "Black-Scholes only prices European options."
            )

        if option.strike == 0.0:
            if isinstance(
                option,
                CallOption,
            ):
                return spot

            if isinstance(
                option,
                PutOption,
            ):
                return 0.0

        if volatility == 0.0:
            discounted_strike = (
                option.strike
                * math.exp(
                    -rate * maturity
                )
            )

            if isinstance(
                option,
                CallOption,
            ):
                return max(
                    spot - discounted_strike,
                    0.0,
                )

            if isinstance(
                option,
                PutOption,
            ):
                return max(
                    discounted_strike - spot,
                    0.0,
                )

        d1, d2 = self._compute_d1_d2(
            spot,
            option.strike,
            rate,
            volatility,
            maturity,
        )

        discounted_strike = (
            option.strike
            * math.exp(
                -rate * maturity
            )
        )

        if isinstance(
            option,
            CallOption,
        ):
            return (
                spot
                * self._normal_cdf(d1)
                - discounted_strike
                * self._normal_cdf(d2)
            )

        if isinstance(
            option,
            PutOption,
        ):
            return (
                discounted_strike
                * self._normal_cdf(-d2)
                - spot
                * self._normal_cdf(-d1)
            )

        raise TypeError(
            "Unsupported option type."
        )