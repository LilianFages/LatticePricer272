class Option:
    """Base class for vanilla options."""

    def __init__(self, strike: float) -> None:
        if strike <= 0.0:
            raise ValueError(
                "Strike must be strictly positive."
            )

        self.strike: float = strike

    def payoff(self, spot: float) -> float:
        """Return the option payoff."""
        raise NotImplementedError


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