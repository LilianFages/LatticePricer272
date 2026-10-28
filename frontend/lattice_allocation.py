"""Observe allocated lattice columns without changing any pricing equations.

This module wraps *instance methods only* during the build, then restores them.
It records the exact Python Node objects the existing engine creates, including
columns' disconnected tails. It does not synthesize missing/avoided nodes.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any

from src.binomial_tree import BinomialTree
from src.hybrid_tree import HybridTree
from src.node import Node
from src.trinomial_tree import TrinomialTree


def _column_nodes(top: Node) -> tuple[Node, ...]:
    """Read all objects in a built column via the engine's lower-neighbor links."""
    column: list[Node] = []
    seen: set[int] = set()
    current: Node | None = top
    while current is not None:
        identifier = id(current)
        if identifier in seen:
            raise RuntimeError("Cycle detected in a lattice column.")
        seen.add(identifier)
        column.append(current)
        if len(column) > 100_000:
            raise RuntimeError("Lattice column exceeds the visualizer safety limit.")
        current = current.lower_neighbor
    return tuple(column)


def _allocation_methods(tree: Any) -> tuple[str, ...]:
    if isinstance(tree, TrinomialTree):
        return ("_build_column",)
    if isinstance(tree, HybridTree):
        return ("_build_binomial_column", "_build_dividend_column")
    if isinstance(tree, BinomialTree):
        return ("_build_next_column",)
    raise TypeError("Unsupported lattice architecture for allocation tracing.")


@contextmanager
def capture_allocated_columns(tree: Any) -> Iterator[list[tuple[Node, ...]]]:
    """Record actual allocated columns while the original tree is being built.

    No financial method is reimplemented. Each wrapped call invokes the original
    method exactly once and returns its unchanged result. The instrumentation
    is per tree instance, and always restored in ``finally``.
    """
    columns: list[tuple[Node, ...]] = [(tree.root,)]
    previous: dict[str, tuple[bool, Any]] = {}

    def wrap(name: str) -> None:
        original = getattr(tree, name)
        in_instance = name in vars(tree)
        previous[name] = (in_instance, vars(tree).get(name))

        def record(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            top = result[0] if isinstance(result, tuple) else result
            if not isinstance(top, Node):
                raise RuntimeError(f"Unexpected {name} return type for a lattice column.")
            columns.append(_column_nodes(top))
            return result

        setattr(tree, name, record)

    try:
        for method in _allocation_methods(tree):
            wrap(method)
        yield columns
    finally:
        for name, (existed, old_value) in previous.items():
            if existed:
                setattr(tree, name, old_value)
            else:
                delattr(tree, name)
