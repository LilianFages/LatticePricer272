"""Risk Numerics: unchanged price engines, explicit N-dependent stencils."""
from __future__ import annotations

import math

import pytest

from frontend.analytics_service import StabilityObservation
from frontend.research_risk_numerics import (
    GAMMA_N_MODES, n_sweep_gamma_bump, observations, summarize_stability,
)


def _obs(
    dim: str, x: float, delta: float = .5,
    gamma: float | None = .02, status: str = "ok",
    gamma_bump: float = 3.5,
) -> StabilityObservation:
    """Create a deterministic observation without running the pricer."""
    return StabilityObservation(
        dim, x, int(x) if dim == "steps" else 100,
        1.0, gamma_bump, delta, gamma, status,
    )


def test_adaptive_n_stencil_is_explicit() -> None:
    assert n_sweep_gamma_bump("Grid-aware (adaptive)") is None


@pytest.mark.parametrize("mode,expected", [
    ("Fixed ±1%", 1.), ("Fixed ±2%", 2.),
    ("Fixed ±3.5%", 3.5), ("Fixed ±5%", 5.),
])
def test_fixed_n_stencil_choices(mode: str, expected: float) -> None:
    assert n_sweep_gamma_bump(mode) == expected


def test_unknown_n_stencil_is_rejected() -> None:
    with pytest.raises(ValueError):
        n_sweep_gamma_bump("fixed unknown")


# Explicitly omit unavailable observations instead of plotting zeros.
def test_failed_and_missing_points_are_omitted_not_zero_filled() -> None:
    rows = (_obs("steps", 40, status="error"),
            _obs("steps", 80, gamma=None),
            _obs("steps", 100, gamma=.01))
    assert [p.coordinate for p in observations(rows, dimension="steps", greek="Gamma")] == [100.]


def test_deviation_requires_analytical_comparator() -> None:
    with pytest.raises(ValueError, match="analytical"):
        observations((_obs("steps", 40),), dimension="steps", greek="Delta", deviation=True)


def test_absolute_and_deviation_are_distinct() -> None:
    rows = (_obs("steps", 40, delta=.4), _obs("steps", 80, delta=.45))
    out = observations(rows, dimension="steps", greek="Delta", analytical=.5, deviation=True)
    assert [p.plotted for p in out] == pytest.approx([-.1, -.05])
    assert [p.measured for p in out] == pytest.approx([.4, .45])


def test_spot_and_n_dimensions_remain_separate() -> None:
    rows = (_obs("spot bump", .5, gamma=.01), _obs("steps", 80, gamma=.02))
    out = observations(rows, dimension="spot bump", greek="Gamma")
    assert len(out) == 1 and out[0].coordinate == .5


# These summaries describe numerical stability rather than accuracy.
def test_summaries_are_spreads_not_accuracy_claims() -> None:
    rows = (_obs("steps", 40, .41, .01), _obs("steps", 80, .44, .03))
    result = summarize_stability(rows, {"Delta": .5, "Gamma": .02})
    assert result.successful == result.total == 2
    assert result.n_delta_span == pytest.approx(.03)
    assert result.n_gamma_span == pytest.approx(.02)
    assert result.n_delta_max_bs_gap == pytest.approx(.09)
    assert result.n_gamma_max_bs_gap == pytest.approx(.01)


def test_near_zero_gamma_flag_only_for_valid_bs_reference() -> None:
    rows = (_obs("steps", 40, gamma=0.), _obs("steps", 80, gamma=.03))
    assert summarize_stability(rows, {"Delta": .5, "Gamma": .02}).near_zero_n_gamma == 1
    assert summarize_stability(rows, None).near_zero_n_gamma == 0


# Invalid plotted Greeks must never appear as valid measurements.
def test_nonfinite_risk_estimate_is_not_plotted() -> None:
    rows = (_obs("steps", 40, gamma=float("nan")), _obs("steps", 80, gamma=.03))
    result = observations(rows, dimension="steps", greek="Gamma")
    assert len(result) == 1 and result[0].coordinate == 80.


def test_invalid_greek_dimension_rejected() -> None:
    with pytest.raises(ValueError):
        observations((), dimension="wrong", greek="Vega")


def test_mode_list_is_stable() -> None:
    assert len(GAMMA_N_MODES) == len(set(GAMMA_N_MODES)) == 5
