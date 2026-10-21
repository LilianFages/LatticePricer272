"""Thin, testable adapter between the Streamlit UI and existing pricing engines.

No financial model or transition-probability equations are reimplemented here.
"""

from __future__ import annotations

import csv
import io
import math
import time
from dataclasses import dataclass, replace
from typing import Callable, Literal

from src.allocation_pruned_trinomial_tree import AllocationPrunedTrinomialTree
from src.pricing_errors import PricingInputError
from src.binomial_tree import BinomialTree
from src.black_scholes import BlackScholesPricer
from src.dividend import CashDividend
from src.hybrid_tree import HybridTree
from src.node import Node
from src.option import CallOption, PutOption
from src.recursive_pricer import RecursivePricer
from src.trinomial_tree import TrinomialTree

# Validate market assumptions before constructing a pricing engine.
ModelName = Literal[
    "Binomial", "Trinomial · Full", "Trinomial · Branch pruning",
    "Trinomial · Allocation pruning", "Hybrid",
]
OptionType = Literal["Call", "Put"]
ExerciseStyle = Literal["European", "American"]

MODELS: tuple[ModelName, ...] = (
    "Binomial",
    "Trinomial · Full",
    "Trinomial · Branch pruning",
    "Trinomial · Allocation pruning",
    "Hybrid",
)
MAX_STEPS = 350  # Interactive latency cap, not a financial/model restriction.
DEFAULT_THRESHOLD = 1e-18


@dataclass(frozen=True, slots=True)
class DividendInput:
    time: float
    amount: float


@dataclass(frozen=True, slots=True)
class PricingRequest:
    spot: float = 100.0
    strike: float = 100.0
    rate: float = 0.02
    volatility: float = 0.20
    maturity: float = 1.0
    steps: int = 100
    model: ModelName = "Trinomial · Allocation pruning"
    option_type: OptionType = "Call"
    exercise: ExerciseStyle = "European"
    # Keep model selection separate from option construction and validation.
    dividends: tuple[DividendInput, ...] = ()
    pruning_threshold: float = DEFAULT_THRESHOLD


@dataclass(frozen=True, slots=True)
class PricingResult:
    request: PricingRequest
    price: float
    elapsed_ms: float
    reachable_nodes: int
    allocated_nodes: int | None
    pruned_transitions: int | None
    black_scholes: float | None
    full_trinomial_reference_nodes: int | None = None

    @property
    def black_scholes_relative_error_pct(self) -> float | None:
        """Relative signed discrepancy, undefined for a zero reference price."""
        if self.black_scholes is None or self.black_scholes == 0.0:
            return None
        return 100.0 * (self.price - self.black_scholes) / self.black_scholes

    @property
    def allocation_savings_pct(self) -> float | None:
        """Allocation reduction versus the full no-dividend trinomial geometry."""
        if self.allocated_nodes is None or self.full_trinomial_reference_nodes is None:
            return None
        return 100.0 * (1.0 - self.allocated_nodes / self.full_trinomial_reference_nodes)

    @property
    def black_scholes_error(self) -> float | None:
        if self.black_scholes is None:
            return None
        return self.price - self.black_scholes


# Pricing callbacks remain injectable for risk and future backward engines.
PricingFunction = Callable[[PricingRequest], PricingResult]

def _dividend_step(time_value: float, maturity: float, steps: int) -> int:
    """Use the same (step_start, step_end] convention as TrinomialTree."""
    dt = maturity / steps
    for step in range(1, steps + 1):
        if (step - 1) * dt < time_value <= step * dt:
            return step
    raise PricingInputError("Dividend dates must fall within the option's maturity.")


def _validate_dividend_schedule(request: PricingRequest) -> None:
    occupied_steps: set[int] = set()
    dividend_pv = 0.0
    for dividend in request.dividends:
        if not all(math.isfinite(float(v)) for v in (dividend.time, dividend.amount)):
            # Preserve dividends, early exercise and pruning conventions.
            raise PricingInputError("Dividend dates and amounts must be finite.")
        if dividend.amount <= 0.0:
            raise PricingInputError("Only strictly positive cash dividends are included.")
        if not 0.0 < dividend.time <= request.maturity:
            raise PricingInputError("Dividend date must belong to (0, maturity].")
        step = _dividend_step(dividend.time, request.maturity, request.steps)
        if step in occupied_steps:
            raise PricingInputError(
                "Two dividends fall into the same lattice time step. "
                "Increase N or change the dividend dates."
            )
        occupied_steps.add(step)
        dividend_pv += dividend.amount * math.exp(-request.rate * dividend.time)

    if request.dividends and dividend_pv >= request.spot:
        raise PricingInputError(
            # Return a stable pricing result contract to Streamlit views.
            "Discounted cash dividends equal or exceed spot; "
            "the lattice's positive forward-price assumption would fail."
        )


def validate_request(request: PricingRequest) -> None:
    """Validate input before passing it to the existing financial engines."""
    scalar_inputs = (
        request.spot, request.strike, request.rate,
        request.volatility, request.maturity, request.pruning_threshold,
    )
    if not all(math.isfinite(float(value)) for value in scalar_inputs):
        raise PricingInputError("Inputs must be finite numbers.")
    if request.spot <= 0.0:
        raise PricingInputError("Spot must be strictly positive.")
    if request.strike < 0.0:
        raise PricingInputError("Strike cannot be negative.")
    if request.volatility < 0.0:
        raise PricingInputError("Volatility cannot be negative.")
    if request.maturity <= 0.0:
        raise PricingInputError("Maturity must be strictly positive.")
    if type(request.steps) is not int or not (1 <= request.steps <= MAX_STEPS):
        raise PricingInputError(f"Steps must be an integer between 1 and {MAX_STEPS}.")
    if request.model not in MODELS:
        # Report numerical failures without substituting synthetic prices.
        raise PricingInputError("Unknown pricing model.")
    if request.option_type not in ("Call", "Put"):
        raise PricingInputError("Option type must be Call or Put.")
    if request.exercise not in ("European", "American"):
        raise PricingInputError("Exercise style must be European or American.")
    if not (0.0 <= request.pruning_threshold < 1.0):
        raise PricingInputError("Pruning threshold must belong to [0, 1).")
    if request.dividends and request.model == "Binomial":
        raise PricingInputError(
            "The existing BinomialTree does not support cash dividends. "
            "Select Hybrid or a Trinomial model."
        )

    _validate_dividend_schedule(request)


def _build_tree(request: PricingRequest):
    core = dict(
        spot=request.spot, rate=request.rate,
        volatility=request.volatility, maturity=request.maturity,
        nb_steps=request.steps,
    )
    if request.model == "Binomial":
        return BinomialTree(**core)

    dividends = [CashDividend(time=d.time, amount=d.amount) for d in request.dividends]
    if request.model == "Hybrid":
        return HybridTree(**core, dividends=dividends)
    # Validate market assumptions before constructing a pricing engine.
    if request.model == "Trinomial · Full":
        return TrinomialTree(**core, dividends=dividends, pruning_threshold=0.0)
    if request.model == "Trinomial · Branch pruning":
        return TrinomialTree(
            **core, dividends=dividends, pruning_threshold=request.pruning_threshold,
        )
    if request.model == "Trinomial · Allocation pruning":
        return AllocationPrunedTrinomialTree(
            **core, dividends=dividends, pruning_threshold=request.pruning_threshold,
        )
    raise PricingInputError("Unknown pricing model.")


def count_reachable_nodes(root: Node) -> int:
    """Count unique nodes *reachable through pricing transitions*, not allocations."""
    seen: set[int] = set()
    pending = [root]
    while pending:
        node = pending.pop()
        key = id(node)
        if key in seen:
            continue
        seen.add(key)
        for child in (node.next_up, node.next_mid, node.next_down):
            if child is not None:
                pending.append(child)
    return len(seen)


def price_contract(request: PricingRequest) -> PricingResult:
    """Run a fresh end-to-end pricing, then collect diagnostics."""
    validate_request(request)
    start = time.perf_counter()
    tree = _build_tree(request)
    option = (CallOption if request.option_type == "Call" else PutOption)(
        strike=request.strike,
        is_american=request.exercise == "American",
    )
    price = RecursivePricer().price(tree, option)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if not math.isfinite(price) or price < 0.0:
        raise RuntimeError("The engine returned an invalid option price.")

    reference = None
    if request.exercise == "European" and not request.dividends:
        reference = BlackScholesPricer().price(
            option=option, spot=request.spot, rate=request.rate,
            # Keep model selection separate from option construction and validation.
            volatility=request.volatility, maturity=request.maturity,
        )
    return PricingResult(
        request=request,
        price=price,
        elapsed_ms=elapsed_ms,
        reachable_nodes=count_reachable_nodes(tree.root),
        allocated_nodes=(
            tree.allocated_node_count
            if isinstance(tree, AllocationPrunedTrinomialTree) else None
        ),
        pruned_transitions=(
            tree.pruned_node_count if isinstance(tree, TrinomialTree) else None
        ),
        black_scholes=reference,
        full_trinomial_reference_nodes=(
            (request.steps + 1) ** 2
            if (
                request.model == "Trinomial · Allocation pruning"
                and not request.dividends
                and request.volatility > 0.0
            ) else None
        ),
    )


def ticket_csv(result: PricingResult) -> str:
    """Export the single valuation as a transparent, auditable CSV ticket."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["field", "value"])
    request = result.request
    fields: list[tuple[str, object]] = [
        ("model", request.model), ("option_type", request.option_type),
        ("exercise", request.exercise), ("spot", request.spot),
        ("strike", request.strike), ("rate", request.rate),
        ("volatility", request.volatility), ("maturity", request.maturity),
        ("steps", request.steps), ("pruning_threshold", request.pruning_threshold),
        ("dividends", "; ".join(f"{d.time}:{d.amount}" for d in request.dividends)),
        ("price", result.price), ("runtime_ms", result.elapsed_ms),
        ("reachable_nodes", result.reachable_nodes),
        ("allocated_nodes", result.allocated_nodes),
        ("pruned_transitions", result.pruned_transitions),
        ("black_scholes", result.black_scholes),
        ("model_minus_black_scholes", result.black_scholes_error),
        ("model_minus_black_scholes_pct", result.black_scholes_relative_error_pct),
        ("full_trinomial_reference_nodes", result.full_trinomial_reference_nodes),
        ("allocation_savings_pct", result.allocation_savings_pct),
    ]
    writer.writerows(fields)
    return output.getvalue()


@dataclass(frozen=True, slots=True)
class SpotSensitivityPoint:
    """One independently repriced contract with spot as the only variable."""

    spot: float
    price: float | None
    error: str | None = None


def compute_spot_sensitivity(
    request: PricingRequest,
    points: int = 9,
    span: float = 0.20,
) -> tuple[SpotSensitivityPoint, ...]:
    """Reprice identical contracts across spot levels; no delta approximation.

    An individual scenario may be invalid for cash dividends; record its
    failure instead of silently returning zero or changing the payoff.
    """
    validate_request(request)
    if type(points) is not int or not 3 <= points <= 21:
        raise PricingInputError("Spot grid must contain between 3 and 21 points.")
    if not math.isfinite(span) or not 0.0 < span < 0.95:
        raise PricingInputError("Spot sensitivity span must belong to (0, 0.95).")

    scenarios: list[SpotSensitivityPoint] = []
    for index in range(points):
        spot = request.spot * (1.0 + span * (2 * index / (points - 1) - 1))
        scenario = replace(request, spot=spot)
        try:
            price = price_contract(scenario).price
        except (PricingInputError, ValueError, RuntimeError, OverflowError,
                ZeroDivisionError) as exc:
            scenarios.append(SpotSensitivityPoint(spot=spot, price=None, error=str(exc)))
        else:
            scenarios.append(SpotSensitivityPoint(spot=spot, price=price))
    return tuple(scenarios)
