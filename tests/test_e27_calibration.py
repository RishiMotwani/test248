"""Offline tests for the Phase 22 E27 calibration runner logic.

No Ollama / no network / no embeddings and no writes to immutable artifacts.
The gate logic (E27 thresholds), the fixed 48-cell grid geometry, pairing and
verdict assembly are exercised on synthetic records; the gold-patch path uses
the in-repo ``StaticCoder``. The runner must never claim an adaptive-vs-RAG
result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments import e27_calibration as e27


def _cell(group="route_contract", variant="A", seed=1, budget=96,
          method="adaptive", success=True, ctx_sha="aaaa",
          hist_tokens=None, corr_recall=1.0, obs_exposure=0.0,
          long_range=1.0, leakage=False, ratio=None) -> dict:
    if hist_tokens is None:
        hist_tokens = budget
    return {
        "task_id": f"{group}_{variant}",
        "group": group,
        "variant": variant,
        "seed": seed,
        "historical_budget": budget,
        "method": method,
        "historical_context_tokens": hist_tokens,
        "budget_ratio": ratio if ratio is not None
        else round(hist_tokens / budget, 3),
        "final_success": bool(success),
        "first_pass_success": bool(success),
        "correction_recall": corr_recall,
        "obsolete_fact_exposure": obs_exposure,
        "long_range_fact_recall": long_range,
        "context_sha": ctx_sha,
        "correction_state_counts": {"CLEAN_CURRENT": 4,
                                    "CURRENT_PLUS_OBSOLETE": 0,
                                    "OBSOLETE_ONLY": 0, "NEITHER": 0},
        "correction_current_ids": ["c1"],
        "correction_obsolete_ids": [] if obs_exposure == 0.0 else ["o1"],
        "leakage_detected": leakage,
        "failure_class": "PASS",
        "patch_sha": "x", "latency_ms": {}, "uses_production_pipeline": True,
    }


def _pilot_pass_records() -> list:
    records = []
    for g in e27.GROUPS:
        for v in e27.VARIANTS:
            for s in e27.SEEDS:
                for m in e27.METHODS:
                    if m == "direct_history":
                        success, exposure = True, 0.0
                    elif m == "no_history":
                        success, exposure = False, 0.0
                    elif m == "adaptive":
                        success, exposure = True, 0.0
                    else:
                        success, exposure = True, 1.0
                    records.append(_cell(
                        group=g, variant=v, seed=s, method=m,
                        success=success, obs_exposure=exposure,
                        ctx_sha=f"{g}:{v}:{s}:{m}"))
    return records


# ---------------------------------------------------------------------------
# Grid geometry (fixed 48-cell pilot only; no full grid)
# ---------------------------------------------------------------------------

def test_grid_dimensions():
    assert e27.GROUPS == ["route_contract", "serialization_contract",
                          "retry_contract"]
    assert e27.VARIANTS == ["A", "B"]
    assert e27.SEEDS == [1, 2]
    assert e27.BUDGETS == [96]
    assert e27.METHODS == ["no_history", "direct_history",
                           "vanilla_rag", "adaptive"]
    assert e27.RECALL_METHODS == ["vanilla_rag", "adaptive"]
    assert len(e27._expected_keys()) == 48
    assert not hasattr(e27, "FULL_VARIANTS"), "E27 must never define a full grid"


def test_grid_completeness_detects_missing():
    all_keys = e27._expected_keys()
    missing_key = all_keys[17]
    records = []
    for k in all_keys:
        if k == missing_key:
            continue
        g, v, s, b, m = k.split(":")
        records.append(_cell(group=g, variant=v, seed=int(s), budget=int(b),
                             method=m))
    grid = e27.grid_completeness(records)
    assert grid["complete"] is False
    assert grid["missing"] == [missing_key]
    g, v, s, b, m = missing_key.split(":")
    complete = e27.grid_completeness(
        records + [_cell(group=g, variant=v, seed=int(s), budget=int(b),
                         method=m)])
    assert complete["complete"] is True


# ---------------------------------------------------------------------------
# Pilot gates (synthetic)
# ---------------------------------------------------------------------------

def test_history_dependence_gate_passes_on_clean_pilot():
    g = e27.compute_pilot_gates(_pilot_pass_records())
    assert g["passed"] is True
    for gid in e27.GROUPS:
        hd = g["gates"]["history_dependence"]["per_group"][gid]
        assert hd["direct_history_success"] == 1.0
        assert hd["no_history_success"] == 0.0
        assert hd["diff_cells"] == 4


def test_history_dependence_gate_fails_when_direct_is_not_better():
    records = _pilot_pass_records()
    for r in records:
        if r["method"] == "direct_history":
            r["final_success"] = False
    g = e27.compute_pilot_gates(records)
    assert g["passed"] is False
    assert g["gates"]["history_dependence"]["passed"] is False


def test_contradiction_state_gate_synthetic():
    records = []
    i = 0
    for g in e27.GROUPS:
        for v in ("A", "B"):
            for s in (1, 2):
                i += 1
                a = _cell(group=g, variant=v, seed=s, method="adaptive",
                          success=True, obs_exposure=0.0, corr_recall=1.0,
                          ctx_sha=f"a{i}")
                b = _cell(group=g, variant=v, seed=s, method="vanilla_rag",
                          success=True, obs_exposure=1.0, corr_recall=0.0,
                          ctx_sha=f"b{i}")
                records += [a, b]
    g = e27._contradiction_state(records)
    assert g["passed"] is True
    assert g["methods"]["adaptive"]["obsolete_fact_exposure"] == 0.0
    assert g["methods"]["vanilla_rag"]["obsolete_fact_exposure"] == 1.0
    for r in records:
        if r["method"] == "adaptive":
            r["obsolete_fact_exposure"] = 0.5
    g2 = e27._contradiction_state(records)
    assert g2["passed"] is False


def test_budget_binding_gate_synthetic():
    def row(g, m, ratio):
        return _cell(group=g, variant="A", seed=1, method=m, success=True,
                     obs_exposure=0.0 if m == "adaptive" else 1.0,
                     hist_tokens=96, ratio=ratio)
    bound = []
    for g in e27.GROUPS:
        for m in e27.RECALL_METHODS:
            bound.append(row(g, m, 1.0))
    g = e27._budget_binding(bound, 96)
    assert g["passed"] is True
    assert g["methods"]["adaptive"]["fraction"] == 1.0
    unbound = []
    for g in e27.GROUPS:
        for m in e27.RECALL_METHODS:
            unbound.append(row(g, m, 0.1 if m == "adaptive" else 1.0))
    g_fail = e27._budget_binding(unbound, 96)
    assert g_fail["passed"] is False
    assert g_fail["methods"]["adaptive"]["fraction"] == 0.0


def test_context_difference_gate_synthetic():
    records = []
    i = 0
    for g in e27.GROUPS:
        for v in ("A", "B"):
            for s in (1, 2):
                i += 1
                records += [
                    _cell(group=g, variant=v, seed=s, method="adaptive",
                          success=True, obs_exposure=0.0, ctx_sha=f"a{i}"),
                    _cell(group=g, variant=v, seed=s, method="vanilla_rag",
                          success=True, obs_exposure=1.0, ctx_sha=f"b{i}"),
                ]
    assert e27._context_difference(records)["passed"] is True
    for r in records:
        r["context_sha"] = "same"
    assert e27._context_difference(records)["passed"] is False


def test_combined_pilot_gates_pass_and_leakage_fails():
    records = _pilot_pass_records()
    g = e27.compute_pilot_gates(records)
    assert g["passed"] is True
    records[0]["leakage_detected"] = True
    g2 = e27.compute_pilot_gates(records)
    assert g2["passed"] is False
    assert g2["gates"]["no_leakage"]["passed"] is False


# ---------------------------------------------------------------------------
# Correction-state classification
# ---------------------------------------------------------------------------

def test_correction_states_classify_pairs():
    class C:
        pass

    c1, c2 = C(), C()
    c1.obsolete_fact_id, c1.current_fact_id = "o1", "c1"
    c2.obsolete_fact_id, c2.current_fact_id = "o2", "c2"
    states = e27._correction_states([c1, c2], {"c1", "c2"})
    assert states == {"CLEAN_CURRENT": 2, "CURRENT_PLUS_OBSOLETE": 0,
                      "OBSOLETE_ONLY": 0, "NEITHER": 0}
    states = e27._correction_states([c1, c2], {"c1", "o1"})
    assert states == {"CLEAN_CURRENT": 0, "CURRENT_PLUS_OBSOLETE": 1,
                      "OBSOLETE_ONLY": 0, "NEITHER": 1}
    states = e27._correction_states([c1, c2], set())
    assert states == {"CLEAN_CURRENT": 0, "CURRENT_PLUS_OBSOLETE": 0,
                      "OBSOLETE_ONLY": 0, "NEITHER": 2}


# ---------------------------------------------------------------------------
# Verdict (calibration only; never an adaptive-vs-RAG claim)
# ---------------------------------------------------------------------------

def test_verdict_requires_fixtures_grid_and_gates():
    grid = {"expected": 48, "present": 48, "missing": [], "complete": True}
    g_pass = e27.compute_pilot_gates(_pilot_pass_records())
    verdict = e27.compute_verdict(pilot_gates=g_pass,
                                  fixture_gates={"passed": True}, grid=grid)
    assert verdict["calibration_valid"] is True
    assert verdict["full_head_to_head_eligible"] is True

    verdict = e27.compute_verdict(pilot_gates=g_pass,
                                  fixture_gates={"passed": False}, grid=grid)
    assert verdict["calibration_valid"] is False
    assert verdict["full_head_to_head_eligible"] is False

    verdict = e27.compute_verdict(pilot_gates=g_pass,
                                  fixture_gates={"passed": True},
                                  grid={"complete": False})
    assert verdict["calibration_valid"] is False

    verdict = e27.compute_verdict(pilot_gates={"passed": False},
                                  fixture_gates={"passed": True}, grid=grid)
    assert verdict["calibration_valid"] is False


def test_verdict_never_claims_adaptive_vs_vanilla():
    verdict = e27.compute_verdict(
        pilot_gates=e27.compute_pilot_gates(_pilot_pass_records()),
        fixture_gates={"passed": True},
        grid={"expected": 48, "present": 48, "missing": [], "complete": True})
    assert "adaptive_beats_vanilla" not in verdict
    assert "mean_diff" not in verdict


# ---------------------------------------------------------------------------
# Gold patches pass offline (StaticCoder)
# ---------------------------------------------------------------------------

def test_gold_checks_pass_for_all_six_variants():
    for g in e27.GROUPS:
        for v in e27.VARIANTS:
            assert e27.gold_check(g, v)["final_success"] is True


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def test_pilot_report_renders_all_sections():
    records = _pilot_pass_records()
    result = {
        "config": {"groups": e27.GROUPS, "variants": e27.VARIANTS,
                   "seeds": e27.SEEDS, "budgets": e27.BUDGETS,
                   "methods": e27.METHODS, "grid_cells": 48,
                   "model": "llama3.1:8b",
                   "embedding_model": "nomic-embed-text"},
        "records": records,
        "fixture_gates": {"passed": True},
        "pilot_gates": e27.compute_pilot_gates(records),
        "aggregate": e27.aggregate(records),
        "diagnostics": e27.compute_diagnostics(records=records,
                                               gold_checks=[]),
        "verdict": e27.compute_verdict(
            pilot_gates=e27.compute_pilot_gates(records),
            fixture_gates={"passed": True},
            grid=e27.grid_completeness(records)),
    }
    text = e27.generate_report(result)
    for n in range(1, 16):
        assert f"## {n}." in text
    for table in ("Table A", "Table B", "Table C", "Table D"):
        assert table in text
    assert "calibration_valid" in text
    assert "full_head_to_head_eligible" in text
    assert "declares no winner" in text
    assert "## 14. Stop rule" in text


def test_aggregate_rolls_up_states():
    records = [_cell(group=g, variant="A", seed=1, method="adaptive",
                     success=True, obs_exposure=0.0, ctx_sha="a")
               for g in e27.GROUPS]
    agg = e27.aggregate(records)
    assert agg["adaptive"]["runs"] == 3
    assert agg["adaptive"]["correction_state_counts"]["CLEAN_CURRENT"] == 12


# ---------------------------------------------------------------------------
# Offline discipline
# ---------------------------------------------------------------------------

def test_runner_has_no_requests_and_kept_results_paths():
    source = Path(e27.__file__).read_text(encoding="utf-8")
    assert "import requests" not in source
    assert "from requests" not in source
    assert "e26_discriminative_coding_benchmark" not in source
    assert "e19_coding_generalization.json" not in source
    assert "e20_counterfactual_history" not in source


def test_runner_imports_offline_successfully():
    assert e27.OUT_JSON.name == "e27_calibration.json"
    assert e27.OUT_REPORT.name == "e27_calibration_report.md"
    assert e27.WORK_ROOT.name == "e27_work"