"""Numerically stable placement of cash ex-dates on a uniform lattice.

An ex-date belongs to (t[k-1], t[k]]. Floating-point equivalents of
an exact grid boundary are treated as that boundary, but genuine interior
ex-dates keep their original timing for dividend transition moments.
"""

from __future__ import annotations

import math

BOUNDARY_ULPS = 8


def _boundary_tolerance(left: float, right: float) -> float:
    """Allow a handful of binary-rounding units, never a fraction of a step."""
    return BOUNDARY_ULPS * max(math.ulp(left), math.ulp(right))


def dividend_grid_step(time: float, maturity: float, steps: int) -> int:
    """Return the 1-based lattice step containing an ex-date.

    The accepted interval is (t[k-1], t[k]]. An approximately identical
    representation of an endpoint is assigned to the step ending there.
    """
    if not (math.isfinite(time) and math.isfinite(maturity)):
        raise ValueError("Dividend and maturity times must be finite.")
    if maturity <= 0.0 or type(steps) is not int or steps < 1:
        raise ValueError("Maturity and step count must be positive.")
    if not 0.0 < time <= maturity:
        raise ValueError("Dividend must occur inside (0, maturity].")

    dt = maturity / steps
    ratio = time / dt
    nearest = round(ratio)
    if 1 <= nearest <= steps:
        endpoint = nearest * dt
        if abs(time - endpoint) <= _boundary_tolerance(time, endpoint):
            return nearest

    assigned = math.ceil(ratio)
    if not 1 <= assigned <= steps:
        raise ValueError("Dividend cannot be assigned to a lattice step.")
    return assigned


def dividend_time_in_step(time: float, start: float, end: float) -> float:
    """Return event time, clamping only a rounding error at the right edge.

    A genuine off-grid event retains its actual ex-date. Events at the left
    edge belong to the previous step, not this one.
    """
    if not all(map(math.isfinite, (time, start, end))) or end <= start:
        raise ValueError("Invalid dividend transition interval.")
    if not start < time:
        raise ValueError("Dividend must occur inside the time step.")
    if abs(time - end) <= _boundary_tolerance(time, end):
        return end
    if time < end:
        return time
    raise ValueError("Dividend must occur inside the time step.")
