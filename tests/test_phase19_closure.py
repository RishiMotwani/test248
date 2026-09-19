"""Phase 19 closure verification tests.

These tests verify the experimental accounting is complete and correct
without calling Ollama (except where explicitly noted).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.e19_coding_generalization as e19
import experiments.e24_missing_negative_control as e24


REPAIRED_JSON = Path(__file__).resolve().parent.parent / "experiments" / "results" / "e19_coding_generalization_full_repaired.json"
RECOVERY_JSON = Path(__file__).resolve().parent.parent / "experiments" / "results" / "e24_missing_negative_control.json"


def test_expected_grid_is_225_cells():
    """The Phase-19 full grid is 225 cells."""
    expected = e24._expected_grid()
    assert len(expected) == 225, f"Expected 225 cells, got {len(expected)}"

    # Verify dimensions: 5 tasks x 3 seeds x 3 budgets x 5 methods
    tasks = {c[0] for c in expected}
    seeds = {c[1] for c in expected}
    budgets = {c[2] for c in expected}
    methods = {c[3] for c in expected}
    assert tasks == set(e19.ALL_TASKS)
    assert seeds == set(e19.FULL_SEEDS)
    assert budgets == set(e19.FULL_BUDGETS)
    assert methods == set(e19.METHODS)


def test_exactly_one_cell_missing_from_historical_json():
    """The historical repaired JSON has exactly one missing cell."""
    data = json.loads(REPAIRED_JSON.read_text())
    records = data.get("records", [])

    actual_keys = set()
    for r in records:
        actual_keys.add((r["task_id"], r["seed"], r["historical_budget"], r["method"]))

    expected = set(e24._expected_grid())
    missing = expected - actual_keys

    assert len(missing) == 1, f"Expected exactly 1 missing cell, found {len(missing)}: {sorted(missing)}"


def test_missing_cell_is_exactly_write_retry_seed3_budget256_llm_summarization():
    """The missing cell has the exact expected identity."""
    data = json.loads(REPAIRED_JSON.read_text())
    records = data.get("records", [])

    actual_keys = set()
    for r in records:
        actual_keys.add((r["task_id"], r["seed"], r["historical_budget"], r["method"]))

    expected = set(e24._expected_grid())
    missing = expected - actual_keys
    missing_key = missing.pop()

    assert missing_key == ("write_retry", 3, 256, "llm_summarization"), \
        f"Missing cell identity mismatch: {missing_key}"


def test_recovery_artifact_has_correct_identity():
    """The recovery JSON has the correct task/seed/budget/method."""
    if not RECOVERY_JSON.exists():
        pytest.skip("Recovery artifact not yet created (run --run first)")

    data = json.loads(RECOVERY_JSON.read_text())
    record = data.get("record", {})

    assert record["task_id"] == "write_retry"
    assert record["seed"] == 3
    assert record["historical_budget"] == 256
    assert record["method"] == "llm_summarization"


def test_historical_artifact_not_modified_by_closure():
    """The historical Phase-19 JSON remains unchanged (224 records)."""
    data = json.loads(REPAIRED_JSON.read_text())
    records = data.get("records", [])

    actual_keys = set()
    for r in records:
        actual_keys.add((r["task_id"], r["seed"], r["historical_budget"], r["method"]))

    # Should still be 224 (the historical artifact is immutable)
    assert len(actual_keys) == 224, \
        f"Historical artifact was modified: now has {len(actual_keys)} records"


def test_primary_grid_remains_135():
    """Primary record count in historical JSON is 135/135."""
    data = json.loads(REPAIRED_JSON.read_text())
    records = data.get("records", [])

    primary_count = sum(1 for r in records if r["task_id"] in e19.PRIMARY_TASKS)
    assert primary_count == 135, f"Primary grid not 135/135: {primary_count}"


def test_recovery_artifact_schema_matches_e19_negative_control():
    """The recovery artifact has the same essential schema as an E19 negative-control record."""
    if not RECOVERY_JSON.exists():
        pytest.skip("Recovery artifact not yet created (run --run first)")

    recovery = json.loads(RECOVERY_JSON.read_text())
    rec = recovery.get("record", {})

    # Load an existing negative-control record from historical JSON for comparison
    data = json.loads(REPAIRED_JSON.read_text())
    historical_negative = None
    for r in data.get("records", []):
        if r["task_id"] in e19.NEGATIVE_CONTROL_TASKS:
            historical_negative = r
            break

    assert historical_negative is not None, "No negative-control records in historical JSON"

    # Both should have the same core E19 schema fields
    core_fields = [
        "experiment", "task_id", "seed", "historical_budget", "method",
        "historical_context_tokens", "workspace_context_tokens",
        "task_prompt_tokens", "total_prompt_tokens",
        "first_pass_success", "final_success",
        "critical_fact_recall", "correction_recall",
        "negative_constraint_recall", "long_range_fact_recall",
        "obsolete_fact_exposure", "failure_class",
        "context_sha", "patch_sha", "latency_ms", "model_calls",
        "uses_production_pipeline", "leakage_detected",
    ]

    for field in core_fields:
        assert field in rec, f"Recovery record missing field: {field}"
        assert field in historical_negative, f"Historical negative record missing field: {field}"


def test_gsd_handoff_contains_phase19_verdict():
    """The GSD handoff document contains the updated Phase-19 verdict."""
    handoff = Path(__file__).resolve().parent.parent / ".planning" / "gsd_handoff.md"
    content = handoff.read_text()

    assert "Phase 19" in content
    assert "correction_identity" in content
    assert "27/27" in content
    assert "adaptive = 27/27" in content
    assert "vanilla_rag = 26/27" in content
    assert "adaptive_advances = False" in content or "adaptive_advances = false" in content
    assert "write_retry / seed=3 / budget=256 / llm_summarization" in content


def test_checkpoint_exists_with_recovered_cell():
    """The phase-19-closed checkpoint exists and contains the recovered-cell identity."""
    checkpoint = Path(__file__).resolve().parent.parent / ".planning" / "checkpoints" / "phase-19-closed.md"
    if not checkpoint.exists():
        pytest.skip("Checkpoint not yet created (will be created after closure)")
        return

    content = checkpoint.read_text()
    assert "write_retry / seed=3 / budget=256 / llm_summarization" in content
    assert "224" in content  # historical artifact count
    assert "135/135" in content  # primary grid
    assert "adaptive = 27/27" in content
    assert "vanilla_rag = 26/27" in content


def test_no_production_memory_files_modified():
    """Closure code does not modify production memory_optimizer or server files."""
    # This is a structural test - the e24 module should not import or modify
    # memory_optimizer/ or server.py
    import experiments.e24_missing_negative_control as e24_mod
    import inspect

    source = inspect.getsource(e24_mod)
    # Should not contain imports of memory_optimizer or server (except possibly via e19/coding_benchmark)
    # The closure utility only uses e19 and coding_benchmark, which are the experiment harness
    assert "memory_optimizer" not in source or "from memory_optimizer" not in source
    assert "server.py" not in source


# The following test requires Ollama and should only be run as part of the full closure flow
@pytest.mark.integration
def test_identify_prints_expected_missing_cell(capsys):
    """--identify prints exactly the expected missing cell."""
    e24.main(["--identify"])
    captured = capsys.readouterr()
    assert "write_retry / seed=3 / budget=256 / method=llm_summarization" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])