"""E22 — thin full-grid runner for the Phase-18 aligned E19.

Rebinds E19's output artifacts to the ``*_full*`` result files and delegates to
``e19.main`` for the full 225-cell grid (135 primary + 90 negative-control),
then restores the original output paths in ``finally`` so importing this module
does not permanently redirect subsequent E19 writes.

Usage::

    python experiments/e22_e19_full_grid.py --full --force
    python experiments/e22_e19_full_grid.py --full --resume

Only delegation; no benchmark logic is duplicated here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import e19_coding_generalization as e19  # noqa: E402

FULL_JSON = Path(__file__).resolve().parent / "results" / "e19_coding_generalization_full.json"
FULL_REPORT = Path(__file__).resolve().parent / "results" / "e19_coding_generalization_full_report.md"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--full" not in argv and "--pilot" not in argv:
        argv.insert(0, "--full")
    if "--force" not in argv and "--resume" not in argv:
        argv.insert(0, "--resume")

    original_json = e19.OUT_JSON
    original_report = e19.OUT_REPORT
    try:
        e19.OUT_JSON = FULL_JSON
        e19.OUT_REPORT = FULL_REPORT
        e19.main(argv)
    finally:
        e19.OUT_JSON = original_json
        e19.OUT_REPORT = original_report
    return 0


if __name__ == "__main__":
    sys.exit(main())