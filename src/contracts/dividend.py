class CashDividend:
    """Represents a known discrete cash dividend."""

    def __init__(
        self,
        time: float,
        amount: float
    ) -> None:
        if time < 0.0:
            raise ValueError(
                "Dividend time cannot be negative."
            )

        if amount < 0.0:
            raise ValueError(
                "Dividend amount cannot be negative."
            )

        self.time: float = time
        self.amount: float = amount