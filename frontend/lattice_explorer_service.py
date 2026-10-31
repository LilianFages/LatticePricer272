"""Read-only snapshots of pricing-engine lattice graphs for small-step inspection.

The engines construct and value the nodes. This adapter only aggregates forward
probabilities and extracts geometry; it never changes transition probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.node import Node
from frontend.lattice_allocation import capture_allocated_columns
from frontend.exercise_policy import (
    dividend_events_for_request, immediate_node_exercise,
)
from src.option import CallOption, PutOption
from src.recursive_pricer import RecursivePricer
from frontend.pricing_service import (
    PricingInputError, PricingRequest, _build_tree, validate_request,
)

from frontend.workspace_policy import MAX_VISUAL_STEPS
EPS = 1e-13


# Traverse the reachable graph without double-counting recombining nodes.
@dataclass(frozen=True)
class VisualNode:
    node_id: str
    step: int
    price: float
    option_value: float
    reach_probability: float
    state: str
    up_probability: float | None
    mid_probability: float | None
    down_probability: float | None
    simplified: bool


@dataclass(frozen=True)
class VisualEdge:
    edge_id: str
    from_id: str
    to_id: str
    step: int
    from_price: float
    to_price: float
    probability: float
    kind: str
    simplified: bool = False


@dataclass(frozen=True)
class AllocatedVisualNode:
    """A *real* allocated node, including nodes without a positive-mass path."""

    node_id: str
    step: int
    price: float
    connected: bool
    option_value: float | None
    reach_probability: float | None


@dataclass(frozen=True)
class LatticeSnapshot:
    request: PricingRequest
    model_price: float
    nodes: tuple[VisualNode, ...]
    edges: tuple[VisualEdge, ...]
    column_masses: tuple[float, ...]
    simplified_nodes: int
    engine_allocated_nodes: int | None
    # Keep generated node labels aligned with their step and state.
    engine_pruned_transitions: int | None
    allocated_geometry: tuple[AllocatedVisualNode, ...] = ()

    @property
    def measured_allocations(self) -> int:
        return len(self.allocated_geometry)

    @property
    def unreached_allocations(self) -> int:
        return sum(not item.connected for item in self.allocated_geometry)

    @property
    def displayed_nodes(self) -> int:
        return len(self.nodes)


def _successors(node: Node) -> list[tuple[str, Node, float]]:
    """Group repeated successor objects; ignore zero-probability branches."""
    grouped: dict[int, list[Any]] = {}
    for name, target, probability in (
        ("Up", node.next_up, node.up_probability),
        ("Mid", node.next_mid, node.mid_probability),
        ("Down", node.next_down, node.down_probability),
    ):
        if target is None or probability is None:
            continue
        if probability < -EPS or probability > 1 + EPS:
            raise RuntimeError("Invalid lattice transition probability.")
        if probability <= 0:
            continue
        key = id(target)
        if key not in grouped:
            grouped[key] = [target, 0.0, []]
        grouped[key][1] += float(probability)
        grouped[key][2].append(name)
    # Recombined successors share one identity and their probability weights add.
    out: list[tuple[str, Node, float]] = []
    for target, p, names in grouped.values():
        out.append(("/".join(names), target, p))
    if out and abs(sum(p for _, _, p in out) - 1.0) > 1e-9:
        raise RuntimeError("Outgoing probability mass is not one.")
    return out


def _exercise_state(
    node: Node, request: PricingRequest, step: int,
    event_steps: dict | None = None,
) -> str:
    """Classify immediate node exercise, separate from dividend-event exercise."""
    if step == request.steps:
        return "Maturity"
    if immediate_node_exercise(
        node, request, step, request.maturity / request.steps,
        ((child, probability) for _, child, probability in _successors(node)),
        event_steps,
    ):
        return "Early exercise"
    return "Continuation"




def _advance_reachable_frontier(
    layer: list,
    mass: dict,
    step: int,
    forward_edges: list,
) -> tuple:
    """Follow positive probability successors without duplicating nodes."""
    next_nodes: dict[int, Node] = {}
    # Preserve probability mass and discounted values from the model.
    next_mass: dict[int, float] = {}
    for node in layer:
        for kind, child, probability in _successors(node):
            key = id(child)
            next_nodes[key] = child
            next_mass[key] = next_mass.get(key, 0.0) + mass[id(node)] * probability
            forward_edges.append((step, node, child, probability, kind))
    frontier = next_nodes
    mass = next_mass
    return frontier, mass

def _trace_reachable_layers(tree: object, request: PricingRequest) -> tuple:
    """Advance actual lattice probability mass without inference."""
    frontier: dict[int, Node] = {id(tree.root): tree.root}
    mass: dict[int, float] = {id(tree.root): 1.0}
    layers: list[list[Node]] = []
    masses: list[dict[int, float]] = []
    forward_edges: list[tuple[int, Node, Node, float, str]] = []
    # Advance the true probability frontier without creating inferred paths.
    for step in range(request.steps + 1):
        layer = sorted(frontier.values(), key=lambda node: (-node.price, id(node)))
        if not layer:
            raise RuntimeError("An empty probability layer was encountered.")
        total = math.fsum(mass[id(node)] for node in layer)
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise RuntimeError(f"Probability mass is {total:.12g} at step {step}.")
        layers.append(layer)
        masses.append(dict(mass))
        if step == request.steps:
            break
        frontier, mass = _advance_reachable_frontier(layer, mass, step, forward_edges)
    return layers, masses, forward_edges



def _make_priced_visual_nodes(layers: list, masses: list, ids: dict, request: PricingRequest) -> object:
    """Describe priced nodes with their measured financial states."""
    nodes_out: list[VisualNode] = []
    event_steps = dividend_events_for_request(request)
    for step, layer in enumerate(layers):
        # Return deterministic tabular coordinates for visualization.
        for node in layer:
            is_simple = (
                node.next_mid is not None
                and node.next_up is node.next_mid is node.next_down
                and node.mid_probability is not None
                and math.isclose(node.mid_probability, 1.0, abs_tol=EPS)
            )
            nodes_out.append(VisualNode(
                node_id=ids[id(node)], step=step, price=float(node.price),
                option_value=float(node.option_value),
                reach_probability=masses[step][id(node)],
                state=_exercise_state(node, request, step, event_steps),
                up_probability=node.up_probability,
                mid_probability=node.mid_probability,
                down_probability=node.down_probability,
                simplified=is_simple,
            ))
    return nodes_out



def _make_allocated_visual_nodes(allocated_columns: list, ids: dict, masses: list) -> object:
    """Describe both connected nodes and allocated unreachable tails."""
    allocated_out: list[AllocatedVisualNode] = []
    for step, column in enumerate(allocated_columns):
        for rank, node in enumerate(column):
            node_ident = id(node)
            connected = node_ident in ids
            allocated_out.append(AllocatedVisualNode(
                node_id=(ids[node_ident] if connected else f"L{step:02d}-A{rank:03d}"),
                step=step,
                price=float(node.price),
                connected=connected,
                option_value=(float(node.option_value) if node.option_value is not None else None),
                reach_probability=(masses[step][node_ident] if connected else None),
            ))
    return allocated_out



def _make_visual_edges(ids: dict, forward_edges: list) -> object:
    """Assign graph identifiers to real positive mass transitions."""
    edge_out = tuple(
        VisualEdge(
            edge_id=f"E{index:04d}", from_id=ids[id(node)], to_id=ids[id(child)],
            step=step, from_price=float(node.price), to_price=float(child.price),
            probability=probability, kind=kind,
            simplified=(
                node.next_mid is not None
                and node.next_up is node.next_mid is node.next_down
                and node.mid_probability is not None
                and math.isclose(node.mid_probability, 1.0, abs_tol=EPS)
            ),
        )
        for index, (step, node, child, probability, kind) in enumerate(forward_edges)
    )
    return edge_out


def snapshot_priced_tree(request: PricingRequest) -> LatticeSnapshot:
    """Build and price a NEW tree; expose positive-probability geometry.

    Not all allocated nodes belong to this geometry: for branch pruning,
    allocated but disconnected tail nodes remain intentionally excluded.
    """
    validate_request(request)
    if request.steps > MAX_VISUAL_STEPS:
        raise PricingInputError(
            f"Lattice Explorer supports at most {MAX_VISUAL_STEPS} steps. "
            "Use Model Comparison for larger trees."
        )

    tree = _build_tree(request)
    option = (CallOption if request.option_type == "Call" else PutOption)(
        request.strike, is_american=request.exercise == "American"
    )
    # Observe actual columns at construction time (including unconnected tails),
    # without rewriting the model or making theoretical geometry assumptions.
    with capture_allocated_columns(tree) as allocated_columns:
        model_price = RecursivePricer().price(tree, option)
    if len(allocated_columns) != request.steps + 1:
        raise RuntimeError(
            "The allocation recorder did not observe all lattice columns."
        )

    # Forward propagation independent of Node.reach_probability: BinomialTree
    # and HybridTree do not populate that field themselves.
    layers, masses, forward_edges = _trace_reachable_layers(tree, request)

    ids = {
        id(node): f"L{step:02d}-N{rank:03d}"
        for step, nodes in enumerate(layers)
        for rank, node in enumerate(nodes)
    }
    nodes_out = _make_priced_visual_nodes(layers, masses, ids, request)
    # Traverse the reachable graph without double-counting recombining nodes.
    allocated_out = _make_allocated_visual_nodes(allocated_columns, ids, masses)
    # The two geometric sets must refer to the same Node identities.
    if sum(item.connected for item in allocated_out) != len(nodes_out):
        raise RuntimeError("Captured columns do not match the priced, connected geometry.")
    reported = getattr(tree, "allocated_node_count", None)
    if reported is not None and reported != len(allocated_out):
        raise RuntimeError("Measured allocations disagree with the engine counter.")

    edge_out = _make_visual_edges(ids, forward_edges)
    # Keep generated node labels aligned with their step and state.
    return LatticeSnapshot(
        request=request, model_price=float(model_price), nodes=tuple(nodes_out),
        edges=edge_out,
        column_masses=tuple(math.fsum(d.values()) for d in masses),
        simplified_nodes=sum(node.simplified for node in nodes_out),
        engine_allocated_nodes=getattr(tree, "allocated_node_count", None),
        engine_pruned_transitions=getattr(tree, "pruned_node_count", None),
        allocated_geometry=tuple(allocated_out),
    )
