"""Exercise-boundary and stopping-time diagnostics on existing priced lattices.

Two probability laws are intentionally kept distinct:
  * unconditional node mass for the (professor's) exercise-region probability;
  * alive path mass for the distribution of the FIRST exercise time.

Pricing, transition probabilities and node values come exclusively from ``src``.
No tree is changed by the diagnostics. The exercise rule uses the same strictly
preferred intrinsic versus discounted continuation convention as the prior
course experiment; ties are treated as continuation.
"""
from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from frontend.pricing_service import (
    PricingInputError, PricingRequest, _build_tree, validate_request,
)
from frontend.exercise_policy import (
    dividend_events_for_request, immediate_node_exercise,
)
from src.option import CallOption, PutOption
from src.recursive_pricer import RecursivePricer

from frontend.workspace_policy import MAX_BOUNDARY_STEPS
PROB_TOL = 2e-10
# Distinguish optimal exercise decisions from path probabilities.
EXERCISE_TOL = 1e-10


@dataclass(frozen=True)
class ExerciseDate:
    step: int
    time: float
    discounted_mean_value: float
    region_probability: float
    first_exercise_probability: float
    cumulative_exercise_probability: float
    survival_probability: float
    lower_exercise_spot: float | None
    upper_exercise_spot: float | None
    boundary_spot: float | None
    exercise_node_count: int
    non_contiguous: bool
    # Keep node states, dividends and early-exercise values consistent.
    total_mass: float


@dataclass(frozen=True)
class ExercisePoint:
    step: int
    time: float
    spot: float
    option_value: float
    reach_probability: float
    state: str


@dataclass(frozen=True)
class ExerciseStudy:
    request: PricingRequest
    premium: float
    dates: tuple[ExerciseDate, ...]
    points: tuple[ExercisePoint, ...]
    early_exercise_probability: float
    total_exercise_probability: float
    expired_without_exercise_probability: float
    worst_probability_mass_error: float
    boundary_non_contiguous_dates: int


def _successors(node: Any) -> tuple[tuple[Any, float], ...]:
    """Preserve probability mass when up/mid/down point to the SAME node."""
    transitions = []
    for child, probability in ((node.next_up, node.up_probability),
                               (node.next_mid, node.mid_probability),
                               (node.next_down, node.down_probability)):
        if child is None:
            if probability is not None and abs(probability) > PROB_TOL:
                raise RuntimeError("Nonzero transition weight without child.")
            continue
        if probability is None or not math.isfinite(probability) or probability < -1e-12:
            raise RuntimeError("Invalid transition probability.")
        if probability > 0:
            transitions.append((child, float(probability)))
    if not math.isclose(math.fsum(p for _, p in transitions), 1.0,
                        abs_tol=PROB_TOL, rel_tol=0):
        raise RuntimeError("Outgoing transition weights must sum to 1.")
    return tuple(transitions)


# Preserve the selected exercise convention and observation date.
def _advance(masses: dict[Any, float]) -> dict[Any, float]:
    successor_mass: dict[Any, float] = defaultdict(float)
    for node, mass in masses.items():
        if mass <= 0:
            continue
        for child, probability in _successors(node):
            successor_mass[child] += mass * probability
    return dict(successor_mass)


def _value(node: Any) -> float:
    value = node.option_value
    if value is None or not math.isfinite(float(value)):
        raise RuntimeError("Node has no finite priced option value.")
    return float(value)


# Return stable chart and table inputs for the exercise boundary panel.
def _payoff(price: float, strike: float, kind: str) -> float:
    return max(price - strike, 0.0) if kind == "Call" else max(strike - price, 0.0)


def _exercise(node: Any, request: PricingRequest, step: int,
              dt: float, event_steps: dict | None = None) -> bool:
    # No outgoing transition is needed at maturity, for a European option,
    # or when exercise has zero intrinsic value.
    needs_continuation = (
        step < request.steps
        and request.exercise == "American"
        and _payoff(float(node.price), request.strike, request.option_type) > 0.0
    )
    successors = _successors(node) if needs_continuation else ()
    return immediate_node_exercise(
        node, request, step, dt, successors, event_steps,
    )



def _validate_exercise_layer(
    full: dict,
    worst_mass_error: float,
    step: int,
    dt: float,
) -> tuple:
    """Verify unconditional probability conservation at each date."""
    time = step * dt
    full_total = math.fsum(full.values())
    worst_mass_error = max(worst_mass_error, abs(full_total - 1.0))
    if not math.isclose(full_total, 1.0, rel_tol=0, abs_tol=PROB_TOL):
        raise RuntimeError(f"Unconditional probability mass is {full_total:.12g} at N={step}.")
    if any(not math.isfinite(p) or p < -PROB_TOL for p in full.values()):
        raise RuntimeError("Invalid unconditional node probability.")
    return time, full_total, worst_mass_error



def _measure_exercise_flows(
    full: dict,
    alive: dict,
    request: PricingRequest,
    step: int,
    dt: float,
    cumulative: float,
    time: float,
    event_steps: dict | None = None,
) -> tuple:
    """Measure lattice-node stopping mass survival and discounted expected value."""
    candidates = {
        node for node in full if _exercise(node, request, step, dt, event_steps)
    }
    region = math.fsum(full[n] for n in candidates)
    first = math.fsum(alive.get(n, 0.0) for n in candidates)
    cumulative += first
    remaining = {node: p for node, p in alive.items() if node not in candidates}
    survival = math.fsum(remaining.values())
    if not math.isclose(survival + cumulative, 1.0, rel_tol=0, abs_tol=PROB_TOL):
        raise RuntimeError("Survival plus cumulative first exercise must sum to 1.")
    mean_pv = math.exp(-request.rate * time) * math.fsum(
        mass * _value(node) for node, mass in full.items()
    )
    return candidates, region, first, cumulative, remaining, survival, mean_pv



def _find_exercise_envelope(
    full: dict,
    candidates: set,
    request: PricingRequest,
    step: int,
) -> tuple:
    """Identify exercise frontier and flag noncontiguous regions."""
    exercised_spots = [float(n.price) for n in candidates]
    bottom = min(exercised_spots) if exercised_spots else None
    top = max(exercised_spots) if exercised_spots else None
    boundary = (top if request.option_type == "Put" else bottom)
    # Mark any non-contiguous exercise set; boundary is only its envelope.
    non_contiguous = False
    if step < request.steps and boundary is not None:
        if request.option_type == "Put":
            non_contiguous = any(float(n.price) < boundary - 1e-10
                                 for n in full if n not in candidates)
        else:
            non_contiguous = any(float(n.price) > boundary + 1e-10
                                 for n in full if n not in candidates)
    return bottom, top, boundary, non_contiguous


def analyze_priced_tree(root: Any, request: PricingRequest) -> tuple[
    tuple[ExerciseDate, ...], tuple[ExercisePoint, ...], float
]:
    """Analyze already-priced nodes; independently track alive and full paths."""
    dt = request.maturity / request.steps
    full: dict[Any, float] = {root: 1.0}
    alive: dict[Any, float] = {root: 1.0}
    output_dates: list[ExerciseDate] = []
    points: list[ExercisePoint] = []
    cumulative = 0.0
    worst_mass_error = 0.0
    event_steps = dividend_events_for_request(request)

    for step in range(request.steps + 1):
        time, full_total, worst_mass_error = _validate_exercise_layer(full, worst_mass_error, step, dt)

        # Distinguish optimal exercise decisions from path probabilities.
        candidates, region, first, cumulative, remaining, survival, mean_pv = _measure_exercise_flows(
            full,
            alive,
            request,
            step,
            dt,
            cumulative,
            time,
            event_steps,
        )

        bottom, top, boundary, non_contiguous = _find_exercise_envelope(full, candidates, request, step)
        output_dates.append(ExerciseDate(
            step=step, time=time, discounted_mean_value=mean_pv,
            region_probability=region, first_exercise_probability=first,
            cumulative_exercise_probability=cumulative,
            survival_probability=survival, lower_exercise_spot=bottom,
            upper_exercise_spot=top, boundary_spot=boundary,
            exercise_node_count=len(candidates), non_contiguous=non_contiguous,
            total_mass=full_total,
        ))
        # Keep node states, dividends and early-exercise values consistent.
        for node, mass in full.items():
            state = ("Maturity · ITM" if node in candidates and step == request.steps else
                     "Early exercise" if node in candidates else
                     "Maturity · OTM" if step == request.steps else "Continuation")
            points.append(ExercisePoint(step=step, time=time, spot=float(node.price),
                                        option_value=_value(node), reach_probability=mass,
                                        state=state))
        if step < request.steps:
            full = _advance(full)
            alive = _advance(remaining)
    return tuple(output_dates), tuple(points), worst_mass_error


def run_exercise_study(request: PricingRequest) -> ExerciseStudy:
    validate_request(request)
    if request.steps > MAX_BOUNDARY_STEPS:
        raise PricingInputError(
            # Preserve the selected exercise convention and observation date.
            f"Exercise Boundary supports at most {MAX_BOUNDARY_STEPS} steps for interactive use."
        )
    tree = _build_tree(request)
    option = (CallOption if request.option_type == "Call" else PutOption)(
        strike=request.strike, is_american=request.exercise == "American"
    )
    premium = float(RecursivePricer().price(tree, option))
    if not math.isfinite(premium) or premium < 0:
        raise RuntimeError("The pricing engine returned an invalid premium.")
    dates, points, mass_error = analyze_priced_tree(tree.root, request)
    if not math.isclose(dates[0].discounted_mean_value, premium,
                        abs_tol=1e-9, rel_tol=1e-9):
        raise RuntimeError("Initial value and discounted expected value disagree.")
    total = dates[-1].cumulative_exercise_probability
    worthless = dates[-1].survival_probability
    if not math.isclose(total + worthless, 1.0, rel_tol=0, abs_tol=PROB_TOL):
        raise RuntimeError("Exercise plus expiry probability is not one.")
    # Return stable chart and table inputs for the exercise boundary panel.
    return ExerciseStudy(
        request=request, premium=premium, dates=dates, points=points,
        early_exercise_probability=dates[-2].cumulative_exercise_probability,
        total_exercise_probability=total,
        expired_without_exercise_probability=worthless,
        worst_probability_mass_error=mass_error,
        boundary_non_contiguous_dates=sum(d.non_contiguous for d in dates[:-1]),
    )


def dates_csv(study: ExerciseStudy) -> str:
    """One row per date; probabilities are fractions in [0,1]."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["model", "option", "exercise_style", "spot_0", "strike",
                     "rate", "volatility", "maturity", "steps", "dividends",
                     "step", "time_years", "discounted_mean_value_eur",
                     "exercise_region_probability", "first_exercise_probability",
                     "cumulative_first_exercise_probability", "survival_probability",
                     "lowest_exercise_spot", "highest_exercise_spot",
                     "boundary_envelope_spot", "non_contiguous", "unconditional_mass"])
    r = study.request
    divs = ";".join(f"{d.time}:{d.amount}" for d in r.dividends)
    for d in study.dates:
        writer.writerow([r.model, r.option_type, r.exercise, r.spot, r.strike,
                         r.rate, r.volatility, r.maturity, r.steps, divs,
                         d.step, d.time, d.discounted_mean_value,
                         d.region_probability, d.first_exercise_probability,
                         d.cumulative_exercise_probability, d.survival_probability,
                         # Distinguish optimal exercise decisions from path probabilities.
                         "" if d.lower_exercise_spot is None else d.lower_exercise_spot,
                         "" if d.upper_exercise_spot is None else d.upper_exercise_spot,
                         "" if d.boundary_spot is None else d.boundary_spot,
                         d.non_contiguous, d.total_mass])
    return out.getvalue()


def nodes_csv(study: ExerciseStudy) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["step", "time_years", "spot", "node_option_value",
                     "risk_neutral_unconditional_reach_probability", "state"])
    for p in study.points:
        writer.writerow([p.step, p.time, p.spot, p.option_value,
                         p.reach_probability, p.state])
    return out.getvalue()
