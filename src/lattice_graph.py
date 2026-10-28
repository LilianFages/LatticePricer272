"""Identity-based traversal of actual pricing transitions.

Never follows ``lower_neighbor``: that link describes allocated geometry,
which can include nodes unreachable by a priced path. Probability filtering
belongs to callers measuring probability-weighted reachability, not here.
"""
from __future__ import annotations

from src.node import Node


def count_reachable_nodes(root: Node) -> int:
    """Count distinct Node objects reachable through up/mid/down pointers."""
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
