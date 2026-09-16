"""Baselines for memory allocation and retrieval comparisons (task A).

Each baseline consumes the same pre-extracted fact stream as the adaptive system;
only the memory policy differs. See baseline_runner for the shared interface.
"""

from baselines.baseline_runner import (  # noqa: F401
    BaseBaseline,
    BaselineRunner,
    fact_matches,
    matched_gt_keys,
)

__all__ = ["BaseBaseline", "BaselineRunner", "fact_matches", "matched_gt_keys"]
