"""Routing policy helper."""
OPS = ("op_17", "op_23", "op_41", "op_52")
LANES = ("lane_a", "lane_b")

def select_route(operation):
    """Return the configured lane for one opaque operation code."""
    raise NotImplementedError
