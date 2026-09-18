from routeapp.router import select_route


def test_op_17_route():
    assert select_route("op_17") == "lane_b"


def test_op_23_route():
    assert select_route("op_23") == "lane_a"


def test_op_41_route():
    assert select_route("op_41") == "lane_a"


def test_op_52_route():
    assert select_route("op_52") == "lane_b"
