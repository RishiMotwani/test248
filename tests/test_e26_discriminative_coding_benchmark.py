"""Offline tests for the Phase 21 E26 benchmark runner logic.

No Ollama / no network / no embeddings and no writes to immutable artifacts.
The gate logic, grid geometry, pairing and verdict assembly are exercised on
synthetic records; the gold-patch path uses the in-repo ``StaticCoder``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from experiments import e26_discriminative_coding_benchmark as e26


def _cell(group="release_adapter", variant="A", seed=1, budget=96,
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
    for g in e26.GROUPS:
        for v in e26.PILOT_VARIANTS:
            for s in e26.PILOT_SEEDS:
                for m in e26.PILOT_METHODS:
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
# Grid geometry
# ---------------------------------------------------------------------------

def test_grid_dimensions():
    assert e26.PILOT_VARIANTS == ["A", "B"]
    assert e26.PILOT_SEEDS == [1, 2]
    assert e26.PILOT_BUDGETS == [96]
    assert e26.PILOT_METHODS == ["no_history", "direct_history",
                                 "vanilla_rag", "adaptive"]
    assert e26.FULL_VARIANTS == ["A"]
    assert e26.FULL_SEEDS == [1, 2, 3]
    assert e26.FULL_BUDGETS == [64, 96, 128]
    assert e26.FULL_METHODS == ["vanilla_rag", "adaptive"]
    assert len(e26._expected_keys(mode="pilot")) == 48
    assert len(e26._expected_keys(mode="full")) == 54


def test_grid_completeness_detects_missing():
    all_keys = e26._expected_keys(mode="full")
    missing_key = all_keys[17]
    records = []
    for k in all_keys:
        if k == missing_key:
            continue
        g, v, s, b, m = k.split(":")
        records.append(_cell(group=g, variant=v, seed=int(s), budget=int(b),
                             method=m))
    full = e26.grid_completeness(records, mode="full")
    assert full["complete"] is False
    assert len(full["missing"]) == 1
    assert full["missing"] == [missing_key]
    g, v, s, b, m = missing_key.split(":")
    complete = e26.grid_completeness(
        records + [_cell(group=g, variant=v, seed=int(s), budget=int(b),
                         method=m)], mode="full")
    assert complete["complete"] is True


# ---------------------------------------------------------------------------
# Pilot gates (synthetic)
# ---------------------------------------------------------------------------

def test_history_dependence_gate_passes_on_clean_pilot():
    g = e26.compute_pilot_gates(_pilot_pass_records())
    assert g["passed"] is True
    for gid in e26.GROUPS:
        hd = g["gates"]["history_dependence"]["per_group"][gid]
        assert hd["direct_history_success"] == 1.0
        assert hd["no_history_success"] == 0.0
        assert hd["diff_cells"] == 4


def test_history_dependence_gate_fails_when_direct_is_not_better():
    records = _pilot_pass_records()
    for r in records:
        if r["method"] == "direct_history":
            r["final_success"] = False
    g = e26.compute_pilot_gates(records)
    assert g["passed"] is False
    assert g["gates"]["history_dependence"]["passed"] is False


def test_contradiction_state_gate_synthetic():
    records = []
    i = 0
    for g in e26.GROUPS:
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
    g = e26._contradiction_state(records)
    assert g["passed"] is True
    assert g["methods"]["adaptive"]["obsolete_fact_exposure"] == 0.0
    assert g["methods"]["vanilla_rag"]["obsolete_fact_exposure"] == 1.0
    for r in records:
        if r["method"] == "adaptive":
            r["obs_mut"] = 1.0
            r["obsolete_fact_exposure"] = 1.0
    g2 = e26._contradiction_state(records)
    assert g2["passed"] is False


def test_budget_binding_gate_synthetic():
    def row(g, m, ratio):
        return _cell(group=g, variant="A", seed=1, method=m, success=True,
                     obs_exposure=0.0 if m == "adaptive" else 1.0,
                     hist_tokens=96, ratio=ratio)
    bound = []
    for g in e26.GROUPS:
        for m in e26.RECALL_METHODS:
            bound.append(row(g, m, 1.0))
    g = e26._budget_binding(bound, 96)
    assert g["passed"] is True
    assert g["methods"]["adaptive"]["fraction"] == 1.0
    unbound = []
    for g in e26.GROUPS:
        for m in e26.RECALL_METHODS:
            unbound.append(row(g, m, 0.1 if m == "adaptive" else 1.0))
    g_fail = e26._budget_binding(unbound, 96)
    assert g_fail["passed"] is False
    assert g_fail["methods"]["adaptive"]["fraction"] == 0.0


def test_context_difference_gate_synthetic():
    records = []
    i = 0
    for g in e26.GROUPS:
        for v in ("A", "B"):
            for s in (1, 2):
                i += 1
                records += [
                    _cell(group=g, variant=v, seed=s, method="adaptive",
                          success=True, obs_exposure=0.0, ctx_sha=f"a{i}"),
                    _cell(group=g, variant=v, seed=s, method="vanilla_rag",
                          success=True, obs_exposure=1.0, ctx_sha=f"b{i}"),
                ]
    assert e26._context_difference(records)["passed"] is True
    for r in records:
        r["context_sha"] = "same"
    assert e26._context_difference(records)["passed"] is False


def test_combined_pilot_gates_pass_and_leakage_fails():
    records = _pilot_pass_records()
    g = e26.compute_pilot_gates(records)
    assert g["passed"] is True
    records[0]["leakage_detected"] = True
    g2 = e26.compute_pilot_gates(records)
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
    states = e26._correction_states([c1, c2], {"c1", "c2"})
    assert states == {"CLEAN_CURRENT": 2, "CURRENT_PLUS_OBSOLETE": 0,
                      "OBSOLETE_ONLY": 0, "NEITHER": 0}
    states = e26._correction_states([c1, c2], {"c1", "o1"})
    assert states == {"CLEAN_CURRENT": 0, "CURRENT_PLUS_OBSOLETE": 1,
                      "OBSOLETE_ONLY": 0, "NEITHER": 1}
    states = e26._correction_states([c1, c2], set())
    assert states == {"CLEAN_CURRENT": 0, "CURRENT_PLUS_OBSOLETE": 0,
                      "OBSOLETE_ONLY": 0, "NEITHER": 2}


# ---------------------------------------------------------------------------
# Paired comparison + verdict
# ---------------------------------------------------------------------------

def test_paired_comparison_success_and_ci_ordering():
    records = []
    for g in e26.GROUPS:
        for s in e26.FULL_SEEDS:
            for b in e26.FULL_BUDGETS:
                records.append(_cell(group=g, variant="A", seed=s, budget=b,
                                     method="adaptive", success=True,
                                     obs_exposure=0.0, ctx_sha="a"))
                records.append(_cell(group=g, variant="A", seed=s, budget=b,
                                     method="vanilla_rag", success=False,
                                     obs_exposure=1.0, ctx_sha="b"))
    paired = e26.paired_comparison(records)
    assert paired["pairs"] == 27
    assert paired["adaptive_success"] == 1.0
    assert paired["vanilla_success"] == 0.0
    assert paired["mean_diff_ci95"][0] > 0


def test_compute_verdict_requires_gates_grid_and_ci():
    paired = {"mean_diff_ci95": [0.1, 0.9], "pairs": 27,
              "adaptive_success": 0.8, "vanilla_success": 0.2}
    full = {"expected": 54, "present": 54, "missing": [], "complete": True}
    fixtures = {"passed": True}
    verdict = e26.compute_verdict(pilot_gates={"passed": True},
                                  full_grid=full, paired=paired,
                                  fixture_gates=fixtures)
    assert verdict["adaptive_beats_vanilla"] is True
    assert verdict["ci_lower_bound"] == 0.1

    verdict = e26.compute_verdict(pilot_gates={"passed": False},
                                  full_grid=full, paired=paired,
                                  fixture_gates=fixtures)
    assert verdict["adaptive_beats_vanilla"] is False

    verdict = e26.compute_verdict(pilot_gates={"passed": True},
                                  full_grid={"complete": False},
                                  paired=paired, fixture_gates=fixtures)
    assert verdict["adaptive_beats_vanilla"] is False

    verdict = e26.compute_verdict(pilot_gates={"passed": True},
                                  full_grid=full,
                                  paired={"mean_diff_ci95": [-0.1, 0.5]},
                                  fixture_gates=fixtures)
    assert verdict["adaptive_beats_vanilla"] is False


# ---------------------------------------------------------------------------
# Gold patches pass offline (StaticCoder)
# ---------------------------------------------------------------------------

def test_gold_checks_pass_for_all_six_variants():
    for g in e26.GROUPS:
        for v in e26.PILOT_VARIANTS:
            assert e26.gold_check(g, v)["final_success"] is True


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def test_pilot_report_renders_gate_section():
    result = {
        "config": {"groups": e26.GROUPS, "variants": e26.PILOT_VARIANTS,
                   "seeds": e26.PILOT_SEEDS, "budgets": e26.PILOT_BUDGETS,
                   "methods": e26.PILOT_METHODS, "grid_cells": 48,
                   "model": "llama3.1:8b",
                   "embedding_model": "nomic-embed-text"},
        "records": _pilot_pass_records(),
        "fixture_gates": {"passed": True},
        "pilot_gates": e26.compute_pilot_gates(_pilot_pass_records()),
        "aggregate": e26.aggregate(_pilot_pass_records()),
        "verdict": {"pilot_gates_passed": True, "note": "pilot"},
    }
    text = e26.generate_pilot_report(result)
    assert "# E26 Discriminative Benchmark - Pilot Report (Phase 21)" in text
    assert "## 4. Pilot gates" in text
    assert "adaptive" in text and "vanilla_rag" in text


def test_aggregate_rolls_up_states():
    records = [_cell(group=g, variant="A", seed=1, method="adaptive",
                     success=True, obs_exposure=0.0, ctx_sha="a")
               for g in e26.GROUPS]
    agg = e26.aggregate(records)
    assert agg["adaptive"]["runs"] == 3
    assert agg["adaptive"]["correction_state_counts"]["CLEAN_CURRENT"] == 12


# ---------------------------------------------------------------------------
# Offline discipline
# ---------------------------------------------------------------------------

def test_runner_has_no_requests_and_kept_results_paths():
    source = Path(e26.__file__).read_text(encoding="utf-8")
    assert "import requests" not in source
    assert "from requests" not in source
    assert "e19_coding_generalization.json" not in source
    assert "e20_counterfactual_history" not in source


def test_runner_imports_offline_successfully():
    assert e26.OUT_PILOT_JSON.name == "e26_discriminative_coding_benchmark_pilot.json"
    assert e26.OUT_JSON.name == "e26_discriminative_coding_benchmark.json"
    assert e26.WORK_ROOT.name == "e26_work"