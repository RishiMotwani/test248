from routeapp.router import choose_lane


def test_op_17():
    assert choose_lane("op-17") == "lane-b"


def test_op_23():
    assert choose_lane("op-23") == "lane-a"


def test_op_41():
    assert choose_lane("op-41") == "lane-a"


def test_op_52():
    assert choose_lane("op-52") == "lane-b"
