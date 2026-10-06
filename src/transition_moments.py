import math

from src.dividend import CashDividend


class TransitionMoments:
    """Computes one-step stock moments for the lattice."""

    @staticmethod
    def compute(
        spot: float,
        rate: float,
        volatility: float,
        step_start: float,
        step_end: float,
        dividend: CashDividend | None = None
    ) -> tuple[float, float]:
        """Return expected value and variance over one time step."""
        TransitionMoments._validate_inputs(
            spot,
            volatility,
            step_start,
            step_end,
            dividend,
        )

        dt = step_end - step_start

        if dividend is None:
            return TransitionMoments._without_dividend(
                spot,
                rate,
                volatility,
                dt,
            )

        return TransitionMoments._with_dividend(
            spot,
            rate,
            volatility,
            step_start,
            step_end,
            dividend,
        )

    @staticmethod
    def _without_dividend(
        spot: float,
        rate: float,
        volatility: float,
        dt: float
    ) -> tuple[float, float]:
        """Return moments when no dividend is paid."""
        expected_value = (
            spot
            * math.exp(rate * dt)
        )

        variance = (
            spot ** 2
            * math.exp(2.0 * rate * dt)
            * (
                math.exp(volatility ** 2 * dt)
                - 1.0
            )
        )

        return expected_value, variance

    @staticmethod
    def _with_dividend(
        spot: float,
        rate: float,
        volatility: float,
        step_start: float,
        step_end: float,
        dividend: CashDividend
    ) -> tuple[float, float]:
        """Return moments when a cash dividend occurs in the step."""
        dt = step_end - step_start

        remaining_time = (
            step_end
            - dividend.time
        )

        # The dividend is carried from ex-date to the end of the step.
        expected_value = (
            spot
            * math.exp(rate * dt)
            - dividend.amount
            * math.exp(rate * remaining_time)
        )

        # The course interpolates the dividend impact on step variance.
        dividend_weight = (
            remaining_time
            / dt
        )

        adjusted_spot = (
            spot
            - dividend_weight
            * dividend.amount
        )

        variance = (
            adjusted_spot ** 2
            * math.exp(2.0 * rate * dt)
            * (
                math.exp(volatility ** 2 * dt)
                - 1.0
            )
        )

        return expected_value, variance

    @staticmethod
    def _validate_inputs(
        spot: float,
        volatility: float,
        step_start: float,
        step_end: float,
        dividend: CashDividend | None
    ) -> None:
        """Validate parameters required by one transition."""
        if spot <= 0.0:
            raise ValueError(
                "Spot must be strictly positive."
            )

        if volatility < 0.0:
            raise ValueError(
                "Volatility cannot be negative."
            )

        if step_end <= step_start:
            raise ValueError(
                "Step end must be greater than step start."
            )

        if dividend is None:
            return

        # Convention: a dividend belongs to the step ending on its ex-date.
        if not (
            step_start
            < dividend.time
            <= step_end
        ):
            raise ValueError(
                "Dividend must occur inside the time step."
            )