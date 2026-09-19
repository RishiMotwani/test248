"""Phase 19 closure: identify and recover the single missing negative-control cell.

This is a narrow utility for experiment accounting closure only.
It does NOT modify the historical Phase-19 artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.counterfactual_task_suite import build_variant as build_counterfactual_variant  # noqa: E402
from data.coding_task_suite import build_task  # noqa: E402
from experiments import coding_benchmark as cb  # noqa: E402
from experiments import e19_coding_generalization as e19  # noqa: E402
from experiments.e17_coding_capability import check_models  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parent / "results"
REPAIRED_JSON = RESULTS_DIR / "e19_coding_generalization_full_repaired.json"
RECOVERY_JSON = RESULTS_DIR / "e24_missing_negative_control.json"
WORK_ROOT = RESULTS_DIR / "e19_work"

EXPECTED_MISSING = ("write_retry", 3, 256, "llm_summarization")


def _run_key(task_id: str, seed: int, budget: int, method: str) -> str:
    return f"{task_id}:{seed}:{budget}:{method}"


def _expected_grid() -> List[Tuple[str, int, int, str]]:
    """Enumerate the full 225-cell Phase-19 grid."""
    cells = []
    for task_id in e19.ALL_TASKS:
        for seed in e19.FULL_SEEDS:
            for budget in e19.FULL_BUDGETS:
                for method in e19.METHODS:
                    cells.append((task_id, seed, budget, method))
    return cells


def identify_missing() -> Tuple[str, int, int, str]:
    """Load the historical JSON and return the one missing cell key."""
    if not REPAIRED_JSON.exists():
        raise FileNotFoundError(f"Historical artifact not found: {REPAIRED_JSON}")

    data = json.loads(REPAIRED_JSON.read_text())
    records = data.get("records", [])

    actual_keys = set()
    for r in records:
        actual_keys.add((r["task_id"], r["seed"], r["historical_budget"], r["method"]))

    expected = set(_expected_grid())
    missing = expected - actual_keys

    if len(missing) == 0:
        raise RuntimeError("IDENTIFY FAIL: zero cells missing — historical JSON is already complete")
    if len(missing) > 1:
        raise RuntimeError(f"IDENTIFY FAIL: {len(missing)} cells missing — expected exactly one: {sorted(missing)}")

    missing_key = missing.pop()
    if missing_key != EXPECTED_MISSING:
        raise RuntimeError(
            f"IDENTIFY FAIL: missing cell differs from expected.\n"
            f"  Expected: {EXPECTED_MISSING}\n"
            f"  Found:    {missing_key}"
        )

    task_id, seed, budget, method = missing_key
    print(f"Missing cell identified: {task_id} / seed={seed} / budget={budget} / method={method}")
    return missing_key


def run_missing_cell(task_id: str, seed: int, budget: int, method: str) -> Dict:
    """Run exactly one cell using the Phase-19 benchmark infrastructure."""
    check_models(
        e19.DEFAULT_MODEL,
        e19.DEFAULT_EMBED_MODEL,
        e19.DEFAULT_ENDPOINT,
        True,
    )

    embed_fn = e19._resolve_embed(
        True,
        e19.DEFAULT_EMBED_MODEL,
        e19.DEFAULT_ENDPOINT,
    )

    coder = cb.OllamaCoder(
        model=e19.DEFAULT_MODEL,
        endpoint=e19.DEFAULT_ENDPOINT,
        max_output_tokens=e19.MAX_OUTPUT_TOKENS,
    )

    task = e19.build_e19_task(task_id, seed=seed)
    method_obj = cb.build_method(
        method,
        model=e19.DEFAULT_MODEL,
        endpoint=e19.DEFAULT_ENDPOINT,
        embed_fn=embed_fn,
        embedding_model=e19.DEFAULT_EMBED_MODEL,
        summarizer_generate_fn=None,
        task=task,
    )

    root = WORK_ROOT / f"{task_id}_{seed}_{budget}_{method}"
    rec = cb.run_method_run(
        task=task,
        seed=seed,
        historical_budget=budget,
        method=method_obj,
        coder=coder,
        work_root=root,
    )

    return e19.to_schema(rec)


def write_recovery_artifact(record: Dict) -> None:
    """Write the single-cell recovery artifact."""
    artifact = {
        "experiment_id": "E24_missing_negative_control",
        "source_experiment": "E19_coding_generalization_full_repaired",
        "recovered_at": time.time(),
        "record": record,
    }
    RECOVERY_JSON.write_text(json.dumps(artifact, indent=2, default=str))
    print(f"Wrote recovery artifact: {RECOVERY_JSON}")


def validate_recovery(record: Dict) -> None:
    """Validate the recovery record matches expected identity and schema."""
    assert record["task_id"] == "write_retry", f"task_id mismatch: {record['task_id']}"
    assert record["seed"] == 3, f"seed mismatch: {record['seed']}"
    assert record["historical_budget"] == 256, f"budget mismatch: {record['historical_budget']}"
    assert record["method"] == "llm_summarization", f"method mismatch: {record['method']}"

    required_fields = [
        "experiment", "task_id", "seed", "historical_budget", "method",
        "first_pass_success", "final_success", "model_calls",
        "failure_class",
    ]
    for field in required_fields:
        if field not in record:
            raise ValueError(f"Recovery record missing required field: {field}")

    print("Recovery record validation: PASS")


def verify_historical_unchanged() -> None:
    """Verify the historical Phase-19 JSON was not modified."""
    data = json.loads(REPAIRED_JSON.read_text())
    records = data.get("records", [])

    actual_keys = set()
    for r in records:
        actual_keys.add((r["task_id"], r["seed"], r["historical_budget"], r["method"]))

    # Primary grid should be 135 cells
    primary_keys = [k for k in actual_keys if k[0] in e19.PRIMARY_TASKS]
    assert len(primary_keys) == 135, f"Primary grid not 135/135: {len(primary_keys)}"

    # Total should still be 224 (one negative-control missing)
    assert len(actual_keys) == 224, f"Historical artifact changed: now {len(actual_keys)} records"

    print("Historical artifact verification: PASS (224 records, 135 primary)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 19 missing negative-control cell closure utility"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--identify",
        action="store_true",
        help="Print the one missing cell from the historical JSON",
    )
    group.add_argument(
        "--run",
        action="store_true",
        help="Run the missing cell and write recovery artifact",
    )
    args = parser.parse_args(argv)

    if args.identify:
        missing = identify_missing()
        print(f"{missing[0]} / seed={missing[1]} / budget={missing[2]} / method={missing[3]}")
        return 0

    if args.run:
        missing = identify_missing()
        task_id, seed, budget, method = missing
        print(f"Running missing cell: {task_id} / seed={seed} / budget={budget} / method={method}")
        record = run_missing_cell(task_id, seed, budget, method)
        validate_recovery(record)
        write_recovery_artifact(record)
        verify_historical_unchanged()
        print("Phase 19 closure: COMPLETE")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())