"""Phase 17: rerun E20 with the repaired counterfactual fixture."""

from __future__ import annotations

import sys
from pathlib import Path

from experiments import e20_counterfactual_history as e20


RESULTS_DIR = Path(__file__).resolve().parent / "results"
REPAIR_JSON = RESULTS_DIR / "e20_counterfactual_history_repair.json"
REPAIR_REPORT = RESULTS_DIR / "e20_counterfactual_history_repair_report.md"


def main(argv=None):
    old_json = e20.OUT_JSON
    old_report = e20.OUT_REPORT

    e20.OUT_JSON = REPAIR_JSON
    e20.OUT_REPORT = REPAIR_REPORT

    try:
        return e20.main(argv)
    finally:
        e20.OUT_JSON = old_json
        e20.OUT_REPORT = old_report


if __name__ == "__main__":
    main(sys.argv[1:])