from __future__ import annotations


class Node:
    """Represents a node in a recombining lattice."""

    def __init__(self, price: float) -> None:
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


class TrunkNode(Node):
    """Represents a trunk node with a backward link."""

    def __init__(self, price: float) -> None:
        super().__init__(price)

        self.previous_trunk: TrunkNode | None = None