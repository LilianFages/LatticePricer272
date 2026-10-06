from src.node import Node, TrunkNode


def test_node_creation() -> None:
    node = Node(100.0)

    assert node.price == 100.0
    assert node.option_value is None

    # A new node must not have any lattice connection yet.
    assert node.next_up is None
    assert node.next_mid is None
    assert node.next_down is None
    assert node.lower_neighbor is None


def test_node_direct_connections() -> None:
    root = Node(100.0)
    up_node = Node(110.0)
    down_node = Node(90.0)

    root.next_up = up_node
    root.next_down = down_node
    up_node.lower_neighbor = down_node

    # Direct references must preserve the recombining lattice structure.
    assert root.next_up is up_node
    assert root.next_down is down_node
    assert root.next_mid is None
    assert up_node.lower_neighbor is down_node


def test_trunk_node_creation() -> None:
    node = TrunkNode(100.0)

    # A trunk node extends Node with a backward trunk reference.
    assert isinstance(node, Node)
    assert node.price == 100.0
    assert node.option_value is None
    assert node.previous_trunk is None


def test_trunk_node_backward_link() -> None:
    previous_node = TrunkNode(100.0)
    current_node = TrunkNode(102.0)

    current_node.previous_trunk = previous_node

    assert current_node.previous_trunk is previous_node