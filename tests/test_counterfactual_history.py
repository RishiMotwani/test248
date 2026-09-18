"""Offline tests for Phase 16 E20 counterfactual-history benchmark calibration.

No Ollama / no network: gold-patch checks use an injected ``GoldCoder`` and the
``no_history`` method with embeddings disabled, exactly like the E17 suite. The
testable guarantees here are:

* the counterfactual fixtures are structurally sound and deterministic
  (workspace + prompt byte-identical across A/B; only history, hidden tests and
  gold patches differ),
* every gold patch applies cleanly and passes its variant's hidden tests
  (Hard Gate 1), offline,
* hard-gate-2 counting and the VALID / E19-full-grid-eligible verdict logic,
* no hidden-test logic leaks into prompts or histories,
* the E19 report renderer uses pilot headings and budget-derived columns
  (presentation-only fix from the smithy plan),
* the E20 report renderer emits its full section structure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.e20_counterfactual_history as e20
import experiments.e19_coding_generalization as e19
from data import counterfactual_task_suite as cf
from experiments import coding_benchmark as cb


# ---------------------------------------------------------------------------
# Fixture structure & determinism
# ---------------------------------------------------------------------------

def test_counterfactual_suite_self_test():
    cf.self_test()


def test_three_groups_two_variants_each():
    assert cf.list_group_ids() == ["routing_policy", "retry_policy",
                                   "serialization_policy"]
    for gid in cf.list_group_ids():
        assert cf.list_variants(gid) == ["A", "B"]


def test_workspace_byte_identical_across_variants():
    for gid in cf.list_group_ids():
        ws = cf.workspace_dir(gid)
        files = sorted(p.relative_to(ws) for p in ws.rglob("*") if p.is_file())
        assert len(files) >= 2
        # workspace is a single per-group fixture; nothing variant-specific
        assert not (cf.variant_dir(gid, "A") / "workspace").exists()
        # the effective context is therefore identical for both variants
        task_a = cf.build_variant(gid, "A", seed=1)
        task_b = cf.build_variant(gid, "B", seed=1)
        assert cb.workspace_context(task_a) == cb.workspace_context(task_b)
        assert cf.workspace_sha(gid) == cf.workspace_sha(gid)


def test_prompt_byte_identical_across_variants():
    for gid in cf.list_group_ids():
        task_a = cf.build_variant(gid, "A", seed=1)
        task_b = cf.build_variant(gid, "B", seed=1)
        assert task_a.task_prompt == task_b.task_prompt


def test_no_history_prompt_identical_across_variants():
    for gid in cf.list_group_ids():
        assert (cf.no_history_prompt_sha(gid, "A")
                == cf.no_history_prompt_sha(gid, "B"))


def test_histories_differ_between_variants():
    for gid in cf.list_group_ids():
        assert cf.history_sha(gid, "A") != cf.history_sha(gid, "B")


def test_final_fact_ids_differ_between_variants():
    for gid in cf.list_group_ids():
        a = e20._final_fact_id(gid, "A")
        b = e20._final_fact_id(gid, "B")
        assert a != b
        assert a.endswith(".corr.001") or a.endswith(".sup.001")


def test_hidden_tests_differ_between_variants():
    for gid in cf.list_group_ids():
        assert cf.hidden_test_sha(gid, "A") != cf.hidden_test_sha(gid, "B")


def test_gold_patches_differ_between_variants():
    for gid in cf.list_group_ids():
        assert cf.gold_patch_sha(gid, "A") != cf.gold_patch_sha(gid, "B")


def test_committed_history_matches_fresh_regeneration():
    for gid in cf.list_group_ids():
        for variant in cf.list_variants(gid):
            regenerated = cf.build_history_text(gid, variant)
            committed = cf.history_path(gid, variant).read_text().splitlines()
            assert regenerated == committed, f"{gid}/{variant} history drifted"


def test_history_dependence_shape():
    for gid in cf.list_group_ids():
        for variant in cf.list_variants(gid):
            task = cf.build_variant(gid, variant, seed=1)
            assert len(task.history) == cf.HISTORY_TURNS_DEFAULT
            assert len(task.gold_facts) >= 2
            assert any((task.history_turns - f.turn_index) > 300
                       for f in task.gold_facts), "decision must be >300 turns old"
            obsoleted = {c.obsolete_fact_id for c in task.corrections}
            assert obsoleted.issubset(set(task.obsolete_fact_ids))
            assert task.gold_patch_path is not None and task.gold_patch_path.is_file()


# ---------------------------------------------------------------------------
# Hard Gate 1: gold patches apply and pass (offline)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gid", cf.list_group_ids())
@pytest.mark.parametrize("variant", ["A", "B"])
def test_gold_patch_applies_and_passes(gid, variant, tmp_path):
    task = cf.build_variant(gid, variant, seed=1)
    gold = task.gold_patch_path.read_text()
    coder = e19.StaticCoder(gold)
    rec = cb.run_method_run(
        task=task, seed=1, historical_budget=e20.BUDGET,
        method=cb.build_method("no_history", model="static", endpoint="http://x",
                               embed_fn=None, embedding_model="", task=task),
        coder=coder, work_root=tmp_path / f"gold_{gid}_{variant}")
    assert rec["patch_applied"], rec["failure_class"]
    assert rec["final_success"], rec["failure_class"]


def test_gold_gate_is_satisfied_by_full_check_set():
    checks = e20.run_gold_checks()
    assert len(checks) == 6
    assert e20.gold_gate_passed(checks)


# ---------------------------------------------------------------------------
# Phase 17: strict offline base-workspace validity + routing-specific integrity
# ---------------------------------------------------------------------------

def test_unpatched_workspace_fails_hidden_tests(tmp_path):
    import subprocess
    import sys

    for gid in cf.list_group_ids():
        for variant in cf.list_variants(gid):
            task = cf.build_variant(gid, variant, seed=1)
            repo = cb.prepare_workspace(
                task,
                tmp_path / f"base_{gid}_{variant}",
            )
            proc = subprocess.run(
                task.hidden_test_command,
                cwd=str(repo),
                text=True,
                capture_output=True,
                check=False,
            )
            assert proc.returncode != 0, (
                f"{gid}/{variant} hidden tests already pass on the "
                "unmodified workspace"
            )


def test_routing_policy_workspace_contains_no_assignment_table():
    gid = "routing_policy"
    task = cf.build_variant(gid, "A", seed=1)
    workspace = cb.workspace_context(task)

    assert "op_17" in workspace
    assert "op_23" in workspace
    assert "op_41" in workspace
    assert "op_52" in workspace
    assert "lane_a" in workspace
    assert "lane_b" in workspace

    forbidden_pairs = [
        'op_17": "lane_a"',
        'op_17": "lane_b"',
        'op_23": "lane_a"',
        'op_23": "lane_b"',
        'op_41": "lane_a"',
        'op_41": "lane_b"',
        'op_52": "lane_a"',
        'op_52": "lane_b"',
    ]

    for pair in forbidden_pairs:
        assert pair not in workspace


def test_routing_policy_mapping_differs_only_by_history_contract():
    a = cf.build_variant("routing_policy", "A", seed=1)
    b = cf.build_variant("routing_policy", "B", seed=1)

    assert cb.workspace_context(a) == cb.workspace_context(b)
    assert a.task_prompt == b.task_prompt
    assert a.history_text != b.history_text
    assert a.hidden_test_path.read_text() != b.hidden_test_path.read_text()
    assert a.gold_patch_path.read_text() != b.gold_patch_path.read_text()


# ---------------------------------------------------------------------------
# No hidden-test leakage
# ---------------------------------------------------------------------------

def test_no_prompt_history_leak():
    for gid in cf.list_group_ids():
        assert e20._no_prompt_history_leak(gid)


def test_constructed_prompts_pass_harness_leakage_check():
    for gid in cf.list_group_ids():
        for variant in cf.list_variants(gid):
            task = cf.build_variant(gid, variant, seed=1)
            prompt = cb.build_coding_prompt(task.task_prompt,
                                            cb.workspace_context(task), "")
            assert not cb.leakage_check(task, prompt)["leakage_detected"]


# ---------------------------------------------------------------------------
# Hard Gate 2 counting + verdict
# ---------------------------------------------------------------------------

def test_gate_2_threshold_constants():
    assert e20.GROUP_DH_MIN == 5
    assert e20.GROUP_NH_MAX == 3
    assert e20.GROUP_SEP_MIN == 2


def _all_method_records(method_outcomes):
    out = []
    for gid in cf.list_group_ids():
        for variant in ("A", "B"):
            for seed in (1, 2, 3):
                out.append(dict(group_id=gid, variant=variant, seed=seed,
                                method="direct_history", final_success=True))
                out.append(dict(group_id=gid, variant=variant, seed=seed,
                                method="no_history",
                                final_success=method_outcomes.get(gid, False)))
                out.append(dict(group_id=gid, variant=variant, seed=seed,
                                method="full_context", final_success=False))
    return out


def test_history_dependence_gate_best_case_passes():
    hd = e20.compute_history_dependence(_all_method_records({}))
    assert hd["passed"]
    for gid in cf.list_group_ids():
        assert hd["groups"][gid]["direct_history"] == "6/6"
        assert hd["groups"][gid]["no_history"] == "0/6"
        assert hd["groups"][gid]["separation"] == 6


def test_history_dependence_gate_fails_when_no_history_solves():
    # one group solved from the workspace alone (no history): gate must fail
    hd = e20.compute_history_dependence(_all_method_records(
        {"retry_policy": True}))
    assert not hd["passed"]
    assert not hd["groups"]["retry_policy"]["passed"]


def test_history_dependence_gate_fails_when_oracle_fails():
    records = _all_method_records({})
    for r in records:
        if r["method"] == "direct_history" and r["seed"] != 1:
            r["final_success"] = False
    hd = e20.compute_history_dependence(records)
    assert not hd["passed"]


def test_verdict_tracks_gates_and_eligibility():
    def _result(passed):
        gates = {
            "gold_patch_validity": {"passed": passed},
            "pair_integrity": {"passed": passed},
            "history_dependence": {"passed": passed},
        }
        return {"gates": gates, "records": []}

    v = e20.classify_verdict(_result(True))
    assert v["history_dependence_benchmark"] == "VALID"
    assert v["e19_full_grid_eligible"] is True

    v = e20.classify_verdict(_result(False))
    assert v["history_dependence_benchmark"] == "INVALID"
    assert v["e19_full_grid_eligible"] is False


def test_verdict_names_failed_groups():
    records = _all_method_records({"routing_policy": True})
    gates = {
        "gold_patch_validity": {"passed": True},
        "pair_integrity": {"passed": True},
        "history_dependence": e20.compute_history_dependence(records),
    }
    v = e20.classify_verdict({"gates": gates, "records": records})
    assert v["history_dependence_benchmark"] == "INVALID"
    assert "routing_policy" in v["rationale"]
    assert "retry_policy" not in v["rationale"]


# ---------------------------------------------------------------------------
# E19 report presentation-only fix (pilot headings + dynamic budget columns)
# ---------------------------------------------------------------------------

def _e19_synthetic_result():
    records = []
    for m in ("adaptive", "raw_clipped"):
        for b in (256, 512):
            for tid in ("user_ids", "validation_pure"):
                records.append({
                    "task_id": tid, "seed": 1, "historical_budget": b,
                    "method": m, "final_success": m == "adaptive",
                    "first_pass_success": m == "adaptive",
                    "failure_class": None, "patch_valid": True,
                    "patch_applied": True,
                    "historical_context_tokens": b,
                    "workspace_context_tokens": 400,
                    "task_prompt_tokens": 200,
                    "total_prompt_tokens": b + 600,
                    "coding_attempts": 1,
                    "diagnostics": {"critical_fact_recall": 1.0,
                                    "correction_recall": 0.0,
                                    "negative_constraint_recall": 1.0,
                                    "long_range_fact_recall": 1.0,
                                    "obsolete_fact_exposure": 0.0},
                })
    cfg = {"mode": "pilot", "budgets": [256, 512], "seeds": [1, 2],
           "methods": ["adaptive", "raw_clipped"], "model": "fake",
           "endpoint": "http://x", "embedding_model": "",
           "use_embeddings": False, "max_output_tokens": 700,
           "temperature": 0.1, "tokenizer": cb.TOKENIZER_NAME,
           "generated_at": 0.0}
    return {"config": cfg, "records": records,
            "gold_checks": [{"task_id": "user_ids", "final_success": True}],
            "diagnostics": [],
            "offline": {"gate": {"passed": True, "detail": {}}},
            "aggregate": e19.aggregate(records), "gates": {"all_passed": False}}


def test_e19_report_pilot_headings_and_budget_columns():
    result = _e19_synthetic_result()
    report = e19.generate_report(result)
    assert "## 14. Pilot: Results by Method x Budget (primary)" in report
    assert "## 15. Pilot: First-Pass vs Final Success" in report
    assert "## 14. Full Grid:" not in report
    assert "## 15. Full Grid:" not in report
    assert "mean (256) | mean (512) |" in report
    assert "mean (1024)" not in report
    # section 18 header column count matches the pilot budgets
    diag_lines = [l for l in report.splitlines()
                  if l.startswith("| method | metric | mean (")]
    assert diag_lines
    header = diag_lines[0]
    assert header.count("mean (") == len([256, 512])


def test_e19_report_full_mode_headings_when_full():
    result = _e19_synthetic_result()
    result["config"]["mode"] = "full"
    report = e19.generate_report(result)
    assert "## 14. Full Grid: Results by Method x Budget (primary)" in report
    assert "## 15. Full Grid: First-Pass vs Final Success" in report


# ---------------------------------------------------------------------------
# E20 report renderer
# ---------------------------------------------------------------------------

def _e20_synthetic_result():
    records = []
    for gid in cf.list_group_ids():
        for variant in ("A", "B"):
            for seed in (1, 2, 3):
                for method in ("no_history", "direct_history", "full_context"):
                    records.append({
                        "group_id": gid, "variant": variant, "seed": seed,
                        "method": method,
                        "final_success": method == "direct_history",
                        "historical_context_tokens": e20.BUDGET,
                        "total_prompt_tokens": 6000,
                        "overflow": method == "full_context",
                        "failure_class": None,
                    })
    cfg = {"mode": "run", "groups": cf.list_group_ids(),
           "budget": e20.BUDGET, "seeds": [1, 2, 3],
           "methods": ["no_history", "direct_history", "full_context"],
           "model": "fake", "endpoint": "http://x",
           "use_embeddings": False, "max_output_tokens": 700,
           "temperature": 0.1, "tokenizer": cb.TOKENIZER_NAME,
           "generated_at": 0.0}
    result = {"config": cfg, "records": records,
              "gold_checks": [{"group_id": gid, "variant": v,
                               "patch_applied": True, "hidden_test_pass": True}
                              for gid in cf.list_group_ids() for v in ("A", "B")],
              "integrity_gates": e20.compute_integrity_gates()}
    result["gates"] = {
        "gold_patch_validity": {"passed": True},
        "pair_integrity": {"passed": True},
        "history_dependence": e20.compute_history_dependence(records),
    }
    result["verdict"] = e20.classify_verdict(result)
    return result


def test_e20_report_has_all_seven_sections():
    report = e20.generate_report(_e20_synthetic_result())
    for i in range(1, 8):
        assert f"## {i}." in report, f"missing section {i}"
    assert "history_dependence_benchmark" in report
    assert "e19_full_grid_eligible" in report


# ---------------------------------------------------------------------------
# Output paths under repo
# ---------------------------------------------------------------------------

def test_output_paths_are_under_repo():
    repo = Path(__file__).resolve().parent.parent
    assert str(e20.OUT_JSON).startswith(str(repo))
    assert str(e20.OUT_REPORT).startswith(str(repo))


def test_e20_and_e19_results_artifacts_compatible():
    # E19 grid is 60 pilot cells x a documented schema; E20 vertices are 54 cells
    assert e20.GRID_CELLS == 54
    assert (e20.BUDGET,) == (1024,)
    assert e20.SEEDS == [1, 2, 3]