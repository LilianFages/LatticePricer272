from __future__ import annotations

import math


class Node:
    """Represents a node in a recombining lattice."""

    def __init__(
        self,
        price: float
    ) -> None:
        self.price: float = price
        self.option_value: float | None = None

        self.next_up: Node | None = None
        self.next_mid: Node | None = None
        self.next_down: Node | None = None

        self.lower_neighbor: Node | None = None

        # Transition probabilities belong to the starting node.
        self.up_probability: float | None = None
        self.mid_probability: float | None = None
        self.down_probability: float | None = None

        # Probability of reaching this node from the root.
        self.reach_probability: float = 0.0


class TrunkNode(Node):
    """Represents a trunk node with a backward link."""

    def __init__(
        self,
        price: float
    ) -> None:
        super().__init__(
            price
        )

        self.previous_trunk: TrunkNode | None = None


def bottom_node(top_node: Node) -> Node:
    """Return the lowest linked node in a lattice column."""
    current = top_node
    while current.lower_neighbor is not None:
        current = current.lower_neighbor
    return current


def nearest_geometric_exponent(
    expected_value: float,
    reference_price: float,
    level_ratio: float,
    *,
    zero_volatility: bool = False
) -> int:
    """Select the closest geometric price level, preferring the lower on ties.

    A zero-volatility trinomial grid has only the central price level.
    """
    if expected_value <= 0.0:
        raise RuntimeError("Expected stock price must remain positive.")
    if zero_volatility:
        return 0

    raw = math.log(expected_value / reference_price) / math.log(level_ratio)
    lower = math.floor(raw)
    upper = lower + 1
    lower_price = reference_price * level_ratio ** lower
    upper_price = reference_price * level_ratio ** upper
    if abs(upper_price - expected_value) < abs(lower_price - expected_value):
        return upper
    return lower


def middle_successors(
    expected_value: float,
    upper_candidate: Node,
    middle_candidate: Node,
    *,
    grid_name: str
) -> tuple[Node, Node, Node]:
    """Slide a three-node successor stencil down to the closest middle level."""
    upper = upper_candidate
    middle = middle_candidate
    while True:
        lower = middle.lower_neighbor
        if lower is None:
            raise RuntimeError(f"{grid_name} grid is too narrow below.")
        if abs(lower.price - expected_value) < abs(middle.price - expected_value):
            if lower.lower_neighbor is None:
                raise RuntimeError(f"{grid_name} grid is too narrow below.")
            upper, middle = middle, lower
            continue
        return upper, middle, lower


def count_reachable_nodes(root: Node) -> int:
    """Count distinct nodes linked by up/mid/down pricing transitions.

    Vertical lower_neighbor links describe allocated geometry, not pricing
    reachability, and must not be traversed here.
    """
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
