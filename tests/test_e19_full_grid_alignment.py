"""Phase-18 alignment tests: E19 is locked to the E20-validated benchmark.

All assertions run offline — no Ollama, no fixture materialisation, no grid
execution. They pin the E19 <-> E20 alignment contract so the full 225-cell
grid cannot silently drift away from the validated counterfactual benchmark.
"""

import pytest

from experiments import e22_e19_full_grid as e22
from experiments import e19_coding_generalization as e19


def test_e19_exact_primary_tasks():
    assert e19.PRIMARY_TASKS == [
        "routing_policy",
        "retry_policy",
        "serialization_policy",
    ]


def test_e19_counterfactual_variant_mapping():
    assert set(e19.E19_COUNTERFACTUAL_VARIANTS) == set(e19.PRIMARY_TASKS)
    for tid in e19.PRIMARY_TASKS:
        assert e19.E19_COUNTERFACTUAL_VARIANTS[tid] == "A"


def test_e20_calibration_is_valid():
    cal = e19.check_e20_calibration()
    assert cal["passed"] is True
    assert cal["groups"] == e19.PRIMARY_TASKS


def test_build_e19_task_counterfactual_primary():
    for tid in e19.PRIMARY_TASKS:
        t = e19.build_e19_task(tid, seed=1)
        assert t.task_id == tid
        assert t.metadata["counterfactual"] is True
        assert t.metadata["variant"] == "A"
        assert t.metadata["group"] == tid
        assert t.gold_patch_path.is_file()
        assert t.hidden_test_path.is_file()
        assert len(t.history) == 600


def test_build_e19_task_negative_controls_not_counterfactual():
    assert set(e19.NEGATIVE_CONTROL_TASKS) == {"cache_readonly", "write_retry"}
    for tid in e19.NEGATIVE_CONTROL_TASKS:
        t = e19.build_e19_task(tid, seed=1)
        assert t.metadata.get("counterfactual") is not True
        assert t.hidden_test_path.is_file()
        gold_path = getattr(t, "gold_patch_path", None)
        if gold_path is None:
            gold_path = t.workspace_path.parent / "gold.patch"
        assert gold_path.is_file()


def test_full_grid_artifact_paths_distinct_from_pilot():
    assert e22.FULL_JSON.name == "e19_coding_generalization_full.json"
    assert e22.FULL_REPORT.name == "e19_coding_generalization_full_report.md"
    assert e22.FULL_JSON != e19.OUT_JSON
    assert e22.FULL_REPORT != e19.OUT_REPORT


def test_full_grid_dimensions():
    assert e19.FULL_SEEDS == [1, 2, 3]
    assert e19.FULL_BUDGETS == [256, 512, 1024]
    assert e19.METHODS == [
        "raw_clipped",
        "sliding_window",
        "llm_summarization",
        "vanilla_rag",
        "adaptive",
    ]
    primary = (len(e19.PRIMARY_TASKS) * len(e19.FULL_SEEDS)
               * len(e19.FULL_BUDGETS) * len(e19.METHODS))
    negative = (len(e19.NEGATIVE_CONTROL_TASKS) * len(e19.FULL_SEEDS)
                * len(e19.FULL_BUDGETS) * len(e19.METHODS))
    assert primary == 135
    assert negative == 90
    assert primary + negative == 225


def test_gate_c_is_e20_certification():
    gates = e19.compute_gates(
        records=[],
        diagnostics=[],
        offline={},
        gold_checks=[],
        unit_tests={},
        task_ids_primary=e19.PRIMARY_TASKS,
        budgets_lo=e19.FULL_BUDGETS[0],
        budgets_hi=e19.FULL_BUDGETS[-1],
    )
    assert gates["history_dependence"]["source"] == "E20 counterfactual history calibration"
    assert gates["history_dependence"]["passed"] is e19.check_e20_calibration()["passed"]