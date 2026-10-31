"""Thin, testable adapter between the Streamlit UI and existing pricing engines.

No financial model or transition-probability equations are reimplemented here.
"""

from __future__ import annotations

import csv
import io
import math
import sys
import time
from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Callable, Literal

from src.lattices.allocation_pruned_trinomial_tree import AllocationPrunedTrinomialTree
from src import PricingInputError
from src.lattices.binomial_tree import BinomialTree
from src.pricing.black_scholes import BlackScholesPricer
from src.contracts.dividend import CashDividend
from src.contracts.dividend_timing import dividend_grid_step
from src.lattices.hybrid_tree import HybridTree
from src.pricing.backward_pricer import BackwardPricer
from src.lattices.node import Node, count_reachable_nodes as _count_priced_nodes
from src.contracts.option import CallOption, PutOption
from src.pricing.recursive_pricer import RecursivePricer
from src.lattices.trinomial_tree import TrinomialTree

# Validate market assumptions before constructing a pricing engine.
ModelName = Literal[
    "Binomial", "Trinomial · Full", "Trinomial · Branch pruning",
    "Trinomial · Allocation pruning", "Hybrid",
]
OptionType = Literal["Call", "Put"]
ExerciseStyle = Literal["European", "American"]
PricingAlgorithm = Literal["Recursive", "Backward"]
PRICING_ALGORITHMS: tuple[PricingAlgorithm, ...] = ("Recursive", "Backward")

MODELS: tuple[ModelName, ...] = (
    "Binomial",
    "Trinomial · Full",
    "Trinomial · Branch pruning",
    "Trinomial · Allocation pruning",
    "Hybrid",
)
MAX_STEPS = 350  # Preserve the limits of research and comparison workspaces.
TERMINAL_STEP_LIMITS: dict[PricingAlgorithm, int] = {
    "Recursive": 700,
    "Backward": 1000,
}
MAX_TERMINAL_STEPS = max(TERMINAL_STEP_LIMITS.values())
DEFAULT_THRESHOLD = 1e-18


@dataclass(frozen=True, slots=True)
class DividendInput:
    # Use the validated contract and preserve the engine's pruning rules.
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
    algorithm: PricingAlgorithm = "Recursive"

    @property
    def black_scholes_relative_error_pct(self) -> float | None:
        """Relative signed discrepancy, undefined for a zero reference price."""
        if self.black_scholes is None or self.black_scholes == 0.0:
            return None
        return 100.0 * (self.price - self.black_scholes) / self.black_scholes

    @property
    def allocation_savings_pct(self) -> float | None:
        """Allocation reduction versus the full no-dividend trinomial geometry."""
        # Use the validated contract and preserve the engine's pruning rules.
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
    """Share the lattice's stable, right-closed ex-date convention."""
    try:
        return dividend_grid_step(time_value, maturity, steps)
    except ValueError as exc:
        raise PricingInputError(
            "Dividend dates must fall within the option's maturity."
        ) from exc


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


def validate_request(
    request: PricingRequest, *, max_steps: int = MAX_STEPS,
) -> None:
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
        # Use the validated contract and preserve the engine's pruning rules.
        raise PricingInputError("Maturity must be strictly positive.")
    if type(request.steps) is not int or not (1 <= request.steps <= max_steps):
        raise PricingInputError(f"Steps must be an integer between 1 and {max_steps}.")
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


def _build_tree(
    request: PricingRequest,
) -> BinomialTree | HybridTree | TrinomialTree | AllocationPrunedTrinomialTree:
    core = dict(
        # Use the validated contract and preserve the engine's pruning rules.
        spot=request.spot, rate=request.rate,
        volatility=request.volatility, maturity=request.maturity,
        nb_steps=request.steps,
    )
    # Model-specific trees retain distinct cash-dividend support.
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
    """Count unique nodes through pricing transitions, excluding allocations."""
    return _count_priced_nodes(root)

def _full_trinomial_reference_nodes(request: PricingRequest) -> int | None:
    """Report theoretical full-tree size only for comparable nondividend cases."""
    if (
        # Use the validated contract and preserve the engine's pruning rules.
        request.model == "Trinomial · Allocation pruning"
        and not request.dividends
        and request.volatility > 0.0
    ):
        return (request.steps + 1) ** 2
    return None


def terminal_step_limit(algorithm: PricingAlgorithm) -> int:
    """Return the Pricing Terminal depth limit for the selected algorithm."""
    try:
        return TERMINAL_STEP_LIMITS[algorithm]
    except (KeyError, TypeError) as exc:
        raise PricingInputError("Unknown pricing algorithm.") from exc


def validate_terminal_request(request: PricingRequest) -> None:
    """Validate a ticket or saved reference within the terminal's widest cap."""
    validate_request(request, max_steps=MAX_TERMINAL_STEPS)


@contextmanager
def _recursive_pricing_capacity(steps: int) -> Iterator[None]:
    """Allow deep memoized traversal and restore the interpreter limit afterward."""
    previous = sys.getrecursionlimit()
    # A recursive step may pass through node, expectation and child methods.
    required = max(previous, 4 * steps + 200)
    changed = required > previous
    if changed:
        sys.setrecursionlimit(required)
    try:
        yield
    finally:
        if changed:
            sys.setrecursionlimit(previous)


def price_contract(request: PricingRequest, *, algorithm: PricingAlgorithm = "Recursive",
) -> PricingResult:
    """Price a contract using the explicitly selected numerical algorithm."""
    validate_request(request, max_steps=terminal_step_limit(algorithm))
    start = time.perf_counter()
    tree = _build_tree(request)
    option = (CallOption if request.option_type == "Call" else PutOption)(
        strike=request.strike,
        is_american=request.exercise == "American",
    )
    # Use the validated contract and preserve the engine's pruning rules.
    if algorithm == "Recursive":
        with _recursive_pricing_capacity(request.steps):
            price = RecursivePricer().price(tree, option)
    else:
        price = BackwardPricer().price(tree, option)
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
        algorithm=algorithm,
        full_trinomial_reference_nodes=_full_trinomial_reference_nodes(request),
    )


# Export observed valuation data without triggering another pricing run.
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
        ("pricing_algorithm", result.algorithm),
        ("reachable_nodes", result.reachable_nodes),
        ("allocated_nodes", result.allocated_nodes),
        ("pruned_transitions", result.pruned_transitions),
        ("black_scholes", result.black_scholes),
        ("model_minus_black_scholes", result.black_scholes_error),
        # Use the validated contract and preserve the engine's pruning rules.
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
    *,
    algorithm: PricingAlgorithm = "Recursive",
) -> tuple[SpotSensitivityPoint, ...]:
    """Reprice identical contracts across spot levels; no delta approximation.

    An individual scenario may be invalid for cash dividends; record its
    failure instead of silently returning zero or changing the payoff.
    """
    validate_request(request, max_steps=terminal_step_limit(algorithm))
    if type(points) is not int or not 3 <= points <= 21:
        raise PricingInputError("Spot grid must contain between 3 and 21 points.")
    if not math.isfinite(span) or not 0.0 < span < 0.95:
        raise PricingInputError("Spot sensitivity span must belong to (0, 0.95).")

    scenarios: list[SpotSensitivityPoint] = []
    for index in range(points):
        spot = request.spot * (1.0 + span * (2 * index / (points - 1) - 1))
        scenario = replace(request, spot=spot)
        try:
            price = price_contract(scenario, algorithm=algorithm).price
        except (PricingInputError, ValueError, RuntimeError, OverflowError,
                # Use the validated contract and preserve the engine's pruning rules.
                ZeroDivisionError) as exc:
            scenarios.append(SpotSensitivityPoint(spot=spot, price=None, error=str(exc)))
        else:
            scenarios.append(SpotSensitivityPoint(spot=spot, price=price))
    return tuple(scenarios)
