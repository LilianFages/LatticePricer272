from src.node import Node


def test_node_creation() -> None:
    node = Node(100.0)

    assert node.price == 100.0
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

    assert root.next_up is up_node
    assert root.next_down is down_node
    assert root.next_mid is None
    assert up_node.lower_neighbor is down_node