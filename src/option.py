class Option:
    """Base class for vanilla options."""

    def __init__(
        self,
        strike: float,
        is_american: bool = False,
    ) -> None:
        if strike < 0.0:
            raise ValueError(
                "Strike cannot be negative."
            )

        self.strike: float = strike
        self.is_american: bool = is_american

    def payoff(self, spot: float) -> float:
        """Return the option payoff."""
        raise NotImplementedError

    def value_at_node(
        self,
        spot: float,
        hold_value: float,
    ) -> float:
        """Return the option value at an interim node."""
        if not self.is_american:
            return hold_value

        exercise_value = self.payoff(
            spot
        )

        return max(
            hold_value,
            exercise_value,
        )


class CallOption(Option):
    """Represents a vanilla call option."""

    def payoff(self, spot: float) -> float:
        return max(
            spot - self.strike,
            0.0,
        )


class PutOption(Option):
    """Represents a vanilla put option."""

    def payoff(self, spot: float) -> float:
        return max(
            self.strike - spot,
            0.0,
        )