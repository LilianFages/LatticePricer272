"""Typed structural contracts for financial risk calculations.

Risk engines depend on protocols, never on the Streamlit pricing adapter.
The concrete PricingRequest remains a frozen dataclass in the pricing API.
"""

from __future__ import annotations

from typing import Callable, Protocol


class RiskDividend(Protocol):
    """Minimal cash-event interface required by the risk routines."""

    @property
    def time(self) -> float:
        ...

    @property
    def amount(self) -> float:
        ...


class RiskRequest(Protocol):
    """Read-only view of market, contract, and lattice assumptions."""

    # Continuous market and contract parameters.
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

    # Discretization and contract conventions.
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

    # Dated cashflows and optional numerical pruning configuration.
    @property
    def dividends(self) -> tuple[RiskDividend, ...]:
        ...

    @property
    def pruning_threshold(self) -> float:
        ...


class RiskPricingResult(Protocol):
    """Numerical premium and elapsed-time fields returned by a pricer."""

    @property
    def price(self) -> float:
        ...

    @property
    def elapsed_ms(self) -> float:
        ...


PricingFunction = Callable[[RiskRequest], RiskPricingResult]
RiskValidator = Callable[[RiskRequest], None]
