from __future__ import annotations


class Node:
    """Represents a node in a recombining lattice."""

    def __init__(self, price: float) -> None:
        self.price: float = price

        self.next_up: Node | None = None
        self.next_mid: Node | None = None
        self.next_down: Node | None = None

        self.lower_neighbor: Node | None = None