"""Pure, numerical summaries of already computed Greek-stability scenarios.

No pricing is triggered. The bump sweep and N sweep have different Gamma
stencils; an adaptive Gamma width changing with N must not be described as
isolating discretization error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence

from frontend.analytics_service import StabilityObservation

GAMMA_N_MODES = (
    "Grid-aware (adaptive)",
    "Fixed ±1%", "Fixed ±2%", "Fixed ±3.5%", "Fixed ±5%",
)


def n_sweep_gamma_bump(mode: str) -> float | None:
    """Translate the user's explicit stencil choice; None uses the grid."""
    if mode == GAMMA_N_MODES[0]:
        return None
    values = dict(zip(GAMMA_N_MODES[1:], (1.0, 2.0, 3.5, 5.0)))
    if mode not in values:
        raise ValueError(f"Unknown Gamma bump mode: {mode!r}")
    return values[mode]


@dataclass(frozen=True, slots=True)
class RiskMeasure:
    dimension: str
    coordinate: float
    steps: int
    greek: str
    measured: float
    plotted: float
    gamma_bump_pct: float
    analytical: float | None


@dataclass(frozen=True, slots=True)
class StabilitySummary:
    successful: int
    total: int
    n_delta_span: float | None
    n_gamma_span: float | None
    n_delta_max_bs_gap: float | None
    n_gamma_max_bs_gap: float | None
    near_zero_n_gamma: int


def observations(
    rows: Sequence[StabilityObservation],
    *, dimension: str, greek: str,
    analytical: float | None = None, deviation: bool = False,
) -> tuple[RiskMeasure, ...]:
    """Retain only successful finite observations and never invent zeroes.

    A 'deviation' is legitimate only when an analytical comparator exists.
    """
    if dimension not in ("spot bump", "steps") or greek not in ("Delta", "Gamma"):
        raise ValueError("Unsupported risk dimension or Greek.")
    if deviation and (analytical is None or not math.isfinite(analytical)):
        raise ValueError("A finite analytical benchmark is required for deviations.")
    field = greek.lower()
    result: list[RiskMeasure] = []
    for point in rows:
        value = getattr(point, field)
        if point.dimension != dimension or point.status != "ok" or value is None:
            continue
        if not all(math.isfinite(float(v)) for v in (
            value, point.coordinate, point.gamma_bump_pct
        )):
            continue
        measured = float(value)
        result.append(RiskMeasure(
            dimension=dimension, coordinate=float(point.coordinate),
            steps=point.steps, greek=greek, measured=measured,
            plotted=measured - analytical if deviation else measured,
            gamma_bump_pct=float(point.gamma_bump_pct),
            analytical=analytical,
        ))
    return tuple(sorted(result, key=lambda x: x.coordinate))


def _span(values: Sequence[RiskMeasure]) -> float | None:
    return max(x.measured for x in values) - min(x.measured for x in values) if values else None


def _max_gap(values: Sequence[RiskMeasure], ref: float | None) -> float | None:
    if ref is None or not math.isfinite(ref) or not values:
        return None
    return max(abs(x.measured-ref) for x in values)


def summarize_stability(
    rows: Sequence[StabilityObservation], analytical: dict[str, float] | None,
) -> StabilitySummary:
    """Observed spans, not confidence intervals or convergence guarantees."""
    valid = sum(point.status == "ok" and point.delta is not None
                and point.gamma is not None
                and math.isfinite(point.delta) and math.isfinite(point.gamma)
                for point in rows)
    delta = observations(rows, dimension="steps", greek="Delta")
    gamma = observations(rows, dimension="steps", greek="Gamma")
    ref_d = analytical.get("Delta") if analytical else None
    ref_g = analytical.get("Gamma") if analytical else None
    near_zero = sum(
        abs(x.measured) < .05*abs(ref_g)
        for x in gamma
    ) if ref_g is not None and math.isfinite(ref_g) and abs(ref_g) > 1e-5 else 0
    return StabilitySummary(
        successful=valid, total=len(rows),
        n_delta_span=_span(delta), n_gamma_span=_span(gamma),
        n_delta_max_bs_gap=_max_gap(delta, ref_d),
        n_gamma_max_bs_gap=_max_gap(gamma, ref_g),
        near_zero_n_gamma=near_zero,
    )
