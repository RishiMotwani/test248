"""Phase-19 closure tests: verify the missing-cell recovery and GSD state."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


RESULTS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_phase19_expected_grid_is_225_cells():
    """The Phase-19 grid is 5 tasks x 3 seeds x 3 budgets x 5 methods = 225."""
    tasks = ["routing_policy", "retry_policy", "serialization_policy",
             "cache_readonly", "write_retry"]
    seeds = [1, 2, 3]
    budgets = [256, 512, 1024]
    methods = ["raw_clipped", "sliding_window", "llm_summarization",
               "vanilla_rag", "adaptive"]
    expected = len(tasks) * len(seeds) * len(budgets) * len(methods)
    assert expected == 225


def test_exactly_one_cell_missing_from_historical_json():
    """The historical repaired JSON has exactly one missing cell."""
    data = load_json(RESULTS_DIR / "e19_coding_generalization_full_repaired.json")
    recs = data.get("records", [])

    tasks = ["routing_policy", "retry_policy", "serialization_policy",
             "cache_readonly", "write_retry"]
    seeds = [1, 2, 3]
    budgets = [256, 512, 1024]
    methods = ["raw_clipped", "sliding_window", "llm_summarization",
               "vanilla_rag", "adaptive"]

    expected = {
        (t, s, b, m)
        for t in tasks
        for s in seeds
        for b in budgets
        for m in methods
    }

    actual = {
        (r["task_id"], r["seed"], r["historical_budget"], r["method"])
        for r in recs
    }

    missing = expected - actual
    assert len(missing) == 1, f"Expected 1 missing cell, found {len(missing)}: {missing}"


def test_missing_cell_is_exactly_write_retry_seed3_budget256_llm_summarization():
    """The missing cell is exactly the known timed-out negative-control cell."""
    data = load_json(RESULTS_DIR / "e19_coding_generalization_full_repaired.json")
    recs = data.get("records", [])

    tasks = ["routing_policy", "retry_policy", "serialization_policy",
             "cache_readonly", "write_retry"]
    seeds = [1, 2, 3]
    budgets = [256, 512, 1024]
    methods = ["raw_clipped", "sliding_window", "llm_summarization",
               "vanilla_rag", "adaptive"]

    expected = {
        (t, s, b, m)
        for t in tasks
        for s in seeds
        for b in budgets
        for m in methods
    }

    actual = {
        (r["task_id"], r["seed"], r["historical_budget"], r["method"])
        for r in recs
    }

    missing = expected - actual
    assert len(missing) == 1
    task_id, seed, budget, method = missing.pop()
    assert task_id == "write_retry"
    assert seed == 3
    assert budget == 256
    assert method == "llm_summarization"


def test_recovery_artifact_has_correct_key():
    """The new recovery JSON has the correct task/seed/budget/method."""
    recovery = load_json(Path(__file__).resolve().parent.parent / "experiments" / "results" / "e24_missing_negative_control.json")

    assert recovery["task_id"] == "write_retry"
    assert recovery["seed"] == 3
    assert recovery["budget"] == 256
    assert recovery["method"] == "llm_summarization"
    assert recovery["source_experiment"] == "e19_coding_generalization_full_repaired"


def test_historical_json_not_modified_by_closure():
    """The historical Phase-19 repaired JSON is not modified by the closure utility."""
    # We verify this by checking the hashes match the known Phase-19 values
    import hashlib

    json_path = Path(__file__).resolve().parent.parent / "experiments" / "results" / "e19_coding_generalization_full_repaired.json"
    report_path = Path(__file__).resolve().parent.parent / "experiments" / "results" / "e19_coding_generalization_full_repaired_report.md"

    json_hash = hashlib.sha256(json_path.read_bytes()).hexdigest()
    report_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()

    assert json_hash == "5d1f58c27c88679c01a1c0ae93ed9b3b5fd8bcf89cc9bbd25cd152a110ac3962"
    assert report_hash == "03ddf8a04277d56bcb9eed6ec8bec218d34b1de982f908dfd66ef2b374d59792"


def test_primary_grid_remains_135():
    """Primary record count remains 135 in the historical artifact."""
    data = load_json(RESULTS_DIR / "e19_coding_generalization_full_repaired.json")
    recs = data.get("records", [])

    primary_tasks = ["routing_policy", "retry_policy", "serialization_policy"]
    primary = sum(1 for r in recs if r["task_id"] in primary_tasks)
    assert primary == 135


def test_recovery_schema_compatible_with_existing_e19_negative_control():
    """The new recovery artifact has the same essential schema as an existing E19 negative-control result."""
    recovery = load_json(Path(__file__).resolve().parent.parent / "experiments" / "results" / "e24_missing_negative_control.json")
    historical = load_json(RESULTS_DIR / "e19_coding_generalization_full_repaired.json")

    # Find an existing negative-control record (cache_readonly or write_retry)
    neg_recs = [r for r in historical.get("records", []) if r["task_id"] in ("cache_readonly", "write_retry")]
    assert neg_recs, "No negative-control records in historical data"
    existing = neg_recs[0]

    # Essential schema fields that must be present in both recovery and existing E19 records
    essential_fields = [
        "task_id", "seed", "historical_budget", "method",
        "first_pass_success", "final_success",
        "failure_class", "model_calls", "uses_production_pipeline",
        "leakage_detected", "context_sha", "patch_sha"
    ]

    for field in essential_fields:
        assert field in recovery, f"Recovery missing field: {field}"
        assert field in existing, f"Existing missing field: {field}"


def test_gsd_handoff_contains_updated_phase19_verdict():
    """The GSD handoff contains the updated Phase-19 verdict."""
    handoff = Path(__file__).resolve().parent.parent / ".planning" / "gsd_handoff.md"
    content = handoff.read_text()

    # Should mention Phase 19 repair completion
    assert "Phase 19" in content
    assert "correction_identity" in content
    assert "27/27" in content
    assert "adaptive" in content


def test_checkpoint_exists_with_recovered_cell_identity():
    """The checkpoint exists and contains the recovered-cell identity."""
    checkpoint = Path(__file__).resolve().parent.parent / ".planning" / "checkpoints" / "phase-19-closed.md"
    assert checkpoint.exists(), "Checkpoint file does not exist"

    content = checkpoint.read_text()
    assert "write_retry" in content
    assert "seed=3" in content or "seed=3" in content
    assert "budget=256" in content
    assert "llm_summarization" in content
    assert "e24_missing_negative_control.json" in content


def test_no_production_memory_files_modified():
    """No production memory files are modified by the closure code."""
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD~1"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent
    )
    changed_files = result.stdout.strip().split()
    for f in changed_files:
        assert not f.startswith("memory_optimizer/"), f"memory_optimizer modified: {f}"
        assert f != "config.yaml", "config.yaml modified"
        assert f != "server.py", "server.py modified"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])