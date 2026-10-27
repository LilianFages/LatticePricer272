"""Pure chart-display selection; full priced lattice data is never discarded."""
from __future__ import annotations

import math

import pandas as pd


def _representative_layer(group: pd.DataFrame, per_step: int) -> pd.DataFrame:
    """Keep extremes and exercise states when downsampling one layer."""
    group = group.sort_values("stock", ascending=False).reset_index(drop=True)
    n = len(group)
    if n <= per_step:
        return group
    anchors = {0, n - 1}
    if "state" in group:
        for _, same_state in group.groupby("state", sort=False):
            anchors.add(int(same_state.index[0]))
            anchors.add(int(same_state.index[-1]))
    if len(anchors) > per_step:
        # Engine states are at most four; this protects against new labels.
        anchors = set(sorted(anchors)[:per_step - 1]) | {n - 1}
    for i in range(per_step):
        if len(anchors) >= per_step:
            break
        anchors.add(round(i * (n - 1) / (per_step - 1)))
    if len(anchors) < per_step:
        for i in range(n):
            if len(anchors) >= per_step:
                break
            anchors.add(i)
    return group.iloc[sorted(anchors)]


def select_chart_rows(
    frame: pd.DataFrame,
    *,
    step_window: tuple[int, int] | None = None,
    overview: bool = False,
    per_step: int = 10,
) -> pd.DataFrame:
    """Select a readable chart, preserving exercise-state boundaries per step.

    Overview: at most ``per_step`` representative points per time layer.
    Detailed window: ALL points in the selected layers, with no sampling.
    The source frame, exercise prices and probability values remain untouched.
    """
    if step_window is not None:
        lower, upper = step_window
        if lower > upper:
            raise ValueError("Invalid time window.")
        frame = frame.loc[frame["step"].between(lower, upper)]
    if not overview or len(frame) == 0:
        return frame.copy()
    if per_step < 2:
        raise ValueError("At least two representative points per step are required.")
    chunks: list[pd.DataFrame] = []
    for _, group in frame.groupby("step", sort=True):
        chunks.append(_representative_layer(group, per_step))
    return pd.concat(chunks, ignore_index=True)


def overview_step_indices(
    final_step: int, *, max_layers: int = 40,
    event_times: tuple[float, ...] = (), maturity: float = 1.0,
) -> frozenset[int]:
    """Keep a sparse global timeline, including maturity and dividend dates.

    The right/left neighboring time steps of each ex-date are retained, so
    overview sampling never hides an ex-date between two lattice layers.
    This changes display only; neither prices nor full snapshots are modified.
    """
    if final_step < 0 or max_layers < 2:
        raise ValueError("Invalid overview time-grid controls.")
    if maturity <= 0 or not math.isfinite(maturity):
        raise ValueError("Maturity must be strictly positive and finite.")
    if final_step + 1 <= max_layers:
        return frozenset(range(final_step + 1))
    stride = math.ceil(final_step / (max_layers - 1))
    selected = set(range(0, final_step + 1, stride))
    selected.add(final_step)
    for event in event_times:
        if not math.isfinite(event) or not 0 <= event <= maturity:
            raise ValueError("Invalid ex-dividend date for overview.")
        coordinate = final_step * event / maturity
        selected.add(max(0, min(final_step, math.floor(coordinate))))
        selected.add(max(0, min(final_step, math.ceil(coordinate))))
    return frozenset(selected)


def representative_transitions(
    edges: tuple, *, step_window: tuple[int, int] | None = None,
    max_links: int = 240,
) -> tuple:
    """Bound the *drawn* real edges, retaining broad time/spot coverage.

    The full immutable snapshot.edges is untouched. The selected lines are
    genuine engine edges, never interpolations, hypothetical links or a new
    transition kernel. Every displayed time interval gets its own budget;
    within each interval edges are sampled across the source-price range.
    """
    if type(max_links) is not int or max_links < 2:
        raise ValueError("max_links must be an integer >= 2")
    if step_window is not None and step_window[0] >= step_window[1]:
        raise ValueError("Link window must contain at least two time layers")
    groups: dict[int, list] = {}
    for edge in edges:
        if step_window is not None and not (
            step_window[0] <= edge.step < step_window[1]
        ):
            continue
        groups.setdefault(edge.step, []).append(edge)
    count = sum(map(len, groups.values()))
    if count <= max_links:
        return tuple(edge for layer in groups.values() for edge in layer)

    if len(groups) > max_links:
        raise ValueError("Link budget smaller than the number of time intervals")
    ordered_steps = sorted(groups)
    base, extra = divmod(max_links, len(ordered_steps))
    chosen = []
    for rank, step in enumerate(ordered_steps):
        layer = sorted(
            groups[step],
            key=lambda edge: (edge.from_price, edge.to_price, edge.from_id,
                              edge.to_id, edge.kind),
        )
        budget = min(len(layer), base + (rank < extra))
        if budget >= len(layer):
            chosen.extend(layer)
            continue
        # Spread lines across the whole underlying-price range, instead of
        # keeping only at-the-money transitions (which would hide tail paths).
        indices = {round(i * (len(layer) - 1) / (budget - 1))
                   for i in range(budget)}
        # Include a genuine simplified/pruned transition where one exists.
        special = next((i for i, edge in enumerate(layer) if edge.simplified), None)
        if special is not None and not any(layer[i].simplified for i in indices):
            indices.remove(sorted(indices)[len(indices) // 2])
            indices.add(special)
        chosen.extend(layer[i] for i in sorted(indices))
    return tuple(chosen)

# LOT_5C4_GUARDED_REFACTOR
