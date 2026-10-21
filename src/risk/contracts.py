"""Typed structural contracts for financial risk calculations.

Risk engines depend on protocols, never on the Streamlit pricing adapter.
The concrete PricingRequest remains a frozen dataclass in the pricing API.
"""

from __future__ import annotations

from typing import Callable, Protocol


class RiskDividend(Protocol):
    @property
    def time(self) -> float:
        ...

    @property
    def amount(self) -> float:
        ...


class RiskRequest(Protocol):
    @property
    def spot(self) -> float:
        ...

    @property
    def strike(self) -> float:
        ...

    @property
    def rate(self) -> float:
        ...

    @property
    def volatility(self) -> float:
        ...

    @property
    def maturity(self) -> float:
        ...

    @property
    def steps(self) -> int:
        ...

    @property
    def model(self) -> str:
        ...

    @property
    def option_type(self) -> str:
        ...

    @property
    def exercise(self) -> str:
        ...

    @property
    def dividends(self) -> tuple[RiskDividend, ...]:
        ...

    @property
    def pruning_threshold(self) -> float:
        ...


class RiskPricingResult(Protocol):
    @property
    def price(self) -> float:
        ...

    @property
    def elapsed_ms(self) -> float:
        ...


PricingFunction = Callable[[RiskRequest], RiskPricingResult]
RiskValidator = Callable[[RiskRequest], None]
