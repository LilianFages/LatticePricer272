"""Pure empirical two-objective Pareto dominance shared by live and archive.

A point is dominated iff another observation is no worse on either axis,
and strictly better on at least one. Identical coordinates both remain.
Only observed points are used; no interpolation, smoothing or model ranking.
Callers retain responsibility for numeric validity, filtering and ordering.
"""
from __future__ import annotations

from collections.abc import Sequence


def nondominated_indices(points: Sequence[tuple[float, float]]) -> tuple[int, ...]:
    """Positions of nondominated (cost, error) points in original order."""
    keep: list[int] = []
    for i, (cost, error) in enumerate(points):
        dominated = any(
            j != i
            and other_cost <= cost
            and other_error <= error
            and (other_cost < cost or other_error < error)
            for j, (other_cost, other_error) in enumerate(points)
        )
        if not dominated:
            keep.append(i)
    return tuple(keep)
