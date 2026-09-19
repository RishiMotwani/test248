"""Offline tests for the Phase 23 E28 model-tier certification wrapper.

No Ollama / no network / no LLM / no embeddings and no writes to immutable
artifacts. The wrapper is a thin delegate to E27: the grid geometry, gate
thresholds and verdict assembly all come from ``e27_calibration`` unchanged.
These tests pin the delegation, the E28 result shape (``model``,
``source_calibration``, ``model_comparison``, ``verdict``), the model-comparison
gate table, the 15-section report with Tables A-E, dry-run context construction
(no coder), and the import-time side-effect discipline (E27 output globals must
not be permanently rebound).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments import e27_calibration as e27
from experiments import e28_model_tier_certification as e28


def _cell(group="route_contract", variant="A", seed=1, budget=96,
          method="adaptive", success=True, ctx_sha="aaaa",
          hist_tokens=None, corr_recall=1.0, obs_exposure=0.0,
          long_range=1.0, leakage=False) -> dict:
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
        "budget_ratio": round(hist_tokens / budget, 3)
        if hist_tokens else 0.0,
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


def _passing_payload(config_model="qwen2.5:7b") -> dict:
    records = []
    i = 0
    for g in e27.GROUPS:
        for v in e27.VARIANTS:
            for s in e27.SEEDS:
                i += 1
                for m in e27.METHODS:
                    if m == "direct_history":
                        records.append(_cell(group=g, variant=v, seed=s,
                                             method=m, success=True,
                                             ctx_sha=f"d:{g}:{v}:{s}"))
                    elif m == "no_history":
                        records.append(_cell(group=g, variant=v, seed=s,
                                             method=m, success=False,
                                             hist_tokens=0,
                                             ctx_sha=f"nh:{g}:{v}:{s}"))
                    elif m == "adaptive":
                        records.append(_cell(group=g, variant=v, seed=s,
                                             method=m, success=True,
                                             obs_exposure=0.0, corr_recall=1.0,
                                             ctx_sha=f"a:{g}:{v}:{s}"))
                    else:
                        records.append(_cell(group=g, variant=v, seed=s,
                                             method=m, success=True,
                                             obs_exposure=1.0, corr_recall=0.0,
                                             ctx_sha=f"v:{g}:{v}:{s}"))
    calls = e27.compute_pilot_gates(records)
    grid = e27.grid_completeness(records)
    return {
        "experiment": "e27_calibration",
        "partial": False,
        "config": {"mode": "pilot", "groups": e27.GROUPS,
                   "variants": e27.VARIANTS, "seeds": e27.SEEDS,
                   "budgets": e27.BUDGETS, "methods": e27.METHODS,
                   "grid_cells": 48, "model": config_model,
                   "embedding_model": "nomic-embed-text",
                   "temperature": 0.1, "tokenizer": "nomic",
                   "ingestion": "oracle_pre_extracted",
                   "adaptive": {"retention_mode": "dual_score"}},
        "fixture_gates": {"passed": True},
        "records": records,
        "pilot_gates": calls,
        "aggregate": e27.aggregate(records),
        "diagnostics": e27.compute_diagnostics(records=records,
                                               gold_checks=[]),
        "verdict": e27.compute_verdict(pilot_gates=calls,
                                       fixture_gates={"passed": True},
                                       grid=grid),
    }


def _wrapped() -> dict:
    return e28.wrap_payload(_passing_payload(),
                            {"pilot_gates": {"gates": {}}, "verdict": {}})


# ---------------------------------------------------------------------------
# Grid surface + output plumbing
# ---------------------------------------------------------------------------

def test_grid_surface_is_exactly_e27():
    assert e27.GROUPS == ["route_contract", "serialization_contract",
                          "retry_contract"]
    assert e27.VARIANTS == ["A", "B"]
    assert e27.SEEDS == [1, 2]
    assert e27.BUDGETS == [96]
    assert e27.METHODS == ["no_history", "direct_history",
                           "vanilla_rag", "adaptive"]
    assert e28.CANDIDATE_MODEL == "qwen2.5:7b"
    assert e28.SOURCE_MODEL == "llama3.1:8b"
    assert len(e27._expected_keys()) == 48


def test_output_paths_point_to_e28_and_source_is_e27():
    assert e28.OUT_JSON.name == "e28_model_tier_certification.json"
    assert e28.OUT_REPORT.name == "e28_model_tier_certification_report.md"
    assert e28.SOURCE_JSON == e27.OUT_JSON
    assert e27.OUT_JSON.name == "e27_calibration.json"
    assert e27.OUT_REPORT.name == "e27_calibration_report.md"
    assert e27.WORK_ROOT.name == "e27_work"


def test_import_does_not_permanently_rebind_e27_outputs():
    assert e27.OUT_JSON.name == "e27_calibration.json"
    assert e27.OUT_REPORT.name == "e27_calibration_report.md"
    assert e27.WORK_ROOT.name == "e27_work"


def test_wrapper_does_not_redefine_locked_thresholds():
    for name in ("DIRECT_MIN_SUCCESS", "NO_HISTORY_MAX_SUCCESS",
                 "HISTORY_DIFF_MIN", "ADAPTIVE_MIN_CORR_RECALL",
                 "ADAPTIVE_MAX_OBS_EXPOSURE", "VANILLA_MIN_OBS_EXPOSURE",
                 "BUDGET_BINDING_RATIO", "BUDGET_BINDING_MIN_FRACTION",
                 "CONTEXT_DIFF_MIN_FRACTION"):
        assert not hasattr(e28, name), f"E28 must not redefine {name}"
    assert e28.GATE_NAMES == ["history_dependence", "contradiction_state",
                              "budget_binding", "context_difference",
                              "no_leakage"]


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------

def test_delegation_passes_model_override_and_rebinds_outputs(tmp_path,
                                                              monkeypatch):
    seen = {}

    def fake_main(argv):
        seen["argv"] = list(argv)
        seen["json_bound"] = e27.OUT_JSON.name
        seen["report_bound"] = e27.OUT_REPORT.name
        seen["work_bound"] = e27.WORK_ROOT.name
        return {"experiment": "e27_calibration", "partial": False,
                "config": {"groups": e27.GROUPS, "variants": e27.VARIANTS,
                           "seeds": e27.SEEDS, "budgets": e27.BUDGETS,
                           "methods": e27.METHODS, "grid_cells": 48,
                           "model": "qwen2.5:7b",
                           "embedding_model": "nomic-embed-text",
                           "temperature": 0.1},
                "fixture_gates": {"passed": True}, "records": [],
                "pilot_gates": {"passed": False, "gates": {}},
                "aggregate": {}, "diagnostics": {}, "verdict": {}}

    monkeypatch.setattr(e27, "main", fake_main)
    json_p = tmp_path / "e28.json"
    report_p = tmp_path / "e28.md"
    work_p = tmp_path / "work"
    monkeypatch.setattr(e28, "OUT_JSON", json_p)
    monkeypatch.setattr(e28, "OUT_REPORT", report_p)
    monkeypatch.setattr(e28, "WORK_ROOT", work_p)

    rc = e28.main(["--run", "--skip-model-check"])
    assert rc == 0
    assert "--pilot" in seen["argv"]
    assert "--model" in seen["argv"]
    assert "qwen2.5:7b" in seen["argv"]
    assert "--resume" in seen["argv"] or "--force" in seen["argv"]
    assert seen["json_bound"] == json_p.name
    assert seen["report_bound"] == report_p.name
    assert seen["work_bound"] == work_p.name
    assert e27.OUT_JSON.name == "e27_calibration.json"
    assert e27.WORK_ROOT.name == "e27_work"
    assert json_p.exists() and report_p.exists()


def test_output_globals_restored_even_on_error(tmp_path, monkeypatch):
    def boom(argv):
        raise RuntimeError("delegated failure")

    monkeypatch.setattr(e27, "main", boom)
    monkeypatch.setattr(e28, "OUT_JSON", tmp_path / "e28.json")
    monkeypatch.setattr(e28, "OUT_REPORT", tmp_path / "e28.md")
    monkeypatch.setattr(e28, "WORK_ROOT", tmp_path / "work")
    with pytest.raises(RuntimeError):
        e28.main(["--run", "--skip-model-check"])
    assert e27.OUT_JSON.name == "e27_calibration.json"
    assert e27.OUT_REPORT.name == "e27_calibration_report.md"
    assert e27.WORK_ROOT.name == "e27_work"


def test_force_passes_force_else_resume(tmp_path, monkeypatch):
    seen = {}

    def fake_main(argv):
        seen["argv"] = list(argv)
        return {"config": {"model": "qwen2.5:7b"}, "records": [],
                "pilot_gates": {"passed": False, "gates": {}}, "verdict": {}}

    monkeypatch.setattr(e27, "main", fake_main)
    monkeypatch.setattr(e28, "OUT_JSON", tmp_path / "e28.json")
    monkeypatch.setattr(e28, "OUT_REPORT", tmp_path / "e28.md")
    monkeypatch.setattr(e28, "WORK_ROOT", tmp_path / "work")
    e28.main(["--run", "--force", "--skip-model-check"])
    assert "--force" in seen["argv"] and "--resume" not in seen["argv"]
    e28.main(["--run", "--skip-model-check"])
    assert "--resume" in seen["argv"] and "--force" not in seen["argv"]


# ---------------------------------------------------------------------------
# Dry run (no LLM)
# ---------------------------------------------------------------------------

def test_dry_run_constructs_contexts_without_an_llm(monkeypatch):
    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError("dry-run must never construct an LLM coder")

    monkeypatch.setattr("experiments.coding_benchmark.OllamaCoder", Boom)
    embed_fn = lambda texts: [[1.0, 0.0]] * len(texts)  # noqa: E731
    out = e28.dry_run(endpoint="http://example.invalid",
                      embedding_model="stub", embed_fn=embed_fn)
    assert out["probe"] == f"{e28.PROBE_GROUP}_{e28.PROBE_VARIANT}"
    assert set(out["methods"]) == set(e27.METHODS)
    for name, m in out["methods"].items():
        assert "historical_context_tokens" in m
        assert "context_sha" in m
        assert m["historical_context_tokens"] >= 0


# ---------------------------------------------------------------------------
# E28 result shape
# ---------------------------------------------------------------------------

def test_wrap_payload_shapes_top_level_document():
    result = _wrapped()
    assert result["experiment"] == "e28_model_tier_certification"
    for key in ("experiment", "model", "source_calibration", "config",
                "records", "pilot_gates", "aggregate", "model_comparison",
                "verdict"):
        assert key in result
    assert result["model"] == "qwen2.5:7b"
    assert result["config"]["model"] == "qwen2.5:7b"
    assert result["source_calibration"]["model"] == "llama3.1:8b"
    assert result["source_calibration"]["experiment"] == "e27_calibration"
    assert result["verdict"]["model"] == "qwen2.5:7b"
    assert result["verdict"]["source_calibration"] == \
        "e27_calibration@llama3.1:8b"
    assert len(result["records"]) == 48


def test_wrap_verdict_forwarded_from_locked_e27():
    result = _wrapped()
    assert result["verdict"]["calibration_valid"] is True
    assert result["verdict"]["head_to_head_eligible"] is True
    assert result["verdict"]["gates_passed"]["history_dependence"] is True
    assert result["verdict"]["gates_passed"]["contradiction_state"] is True
    assert result["verdict"]["gates_passed"]["budget_binding"] is True
    assert result["verdict"]["gates_passed"]["context_difference"] is True
    assert result["verdict"]["gates_passed"]["no_leakage"] is True
    for key in ("calibration_valid", "model", "source_calibration",
                "gates_passed", "head_to_head_eligible", "rationale"):
        assert key in result["verdict"]


def test_wrap_verdict_is_false_when_e27_was_not_valid():
    payload = _passing_payload()
    payload["verdict"]["calibration_valid"] = False
    payload["verdict"]["full_head_to_head_eligible"] = False
    result = e28.wrap_payload(payload, {})
    assert result["verdict"]["calibration_valid"] is False
    assert result["verdict"]["head_to_head_eligible"] is False


def test_model_comparison_compares_both_tiers_without_ranking():
    source = {"pilot_gates": {
        "gates": {"history_dependence": {"passed": False},
                  "contradiction_state": {"passed": False},
                  "budget_binding": {"passed": True},
                  "context_difference": {"passed": True},
                  "no_leakage": {"passed": True}}},
        "verdict": {"calibration_valid": False,
                    "full_head_to_head_eligible": False}}
    result = e28.wrap_payload(_passing_payload(), source)
    comp = result["model_comparison"]
    assert comp["source_model"] == "llama3.1:8b"
    assert comp["candidate_model"] == "qwen2.5:7b"
    assert comp["no_model_ranking"] is True
    for name in e28.GATE_NAMES:
        g = comp["gates"][name]
        assert "source" in g and "candidate" in g
        assert "passed_source" in g and "passed_candidate" in g
    assert comp["gates"]["history_dependence"]["passed_source"] is False
    assert comp["gates"]["history_dependence"]["passed_candidate"] is True
    assert comp["verdict"]["candidate"]["calibration_valid"] is True
    assert comp["verdict"]["source"]["calibration_valid"] is False


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def test_report_has_exactly_fifteen_sections():
    text = e28.generate_report(_wrapped())
    for n in range(1, 16):
        assert f"## {n}. " in text
    assert text.count("## ") == len(e28.SECTIONS)


def test_report_has_tables_a_through_e():
    text = e28.generate_report(_wrapped())
    for table in ("Table A", "Table B", "Table C", "Table D", "Table E"):
        assert table in text


def test_report_names_both_models_and_never_ranks():
    text = e28.generate_report(_wrapped())
    assert "qwen2.5:7b" in text
    assert "llama3.1:8b" in text
    lower = text.lower()
    assert "beats" not in lower
    assert "wins" not in lower


def test_report_never_claims_an_adaptive_result():
    text = e28.generate_report(_wrapped())
    assert "adaptive_beats_vanilla" not in text
    assert "head_to_head_eligible" in text
    assert "commissioned" in text


def test_report_only_regenerates_without_ollama(tmp_path, monkeypatch):
    result = _wrapped()
    json_p = tmp_path / "e28.json"
    report_p = tmp_path / "e28.md"
    json_p.write_text(__import__("json").dumps(result))
    monkeypatch.setattr(e28, "OUT_JSON", json_p)
    monkeypatch.setattr(e28, "OUT_REPORT", report_p)
    rc = e28.main(["--report-only"])
    assert rc == 0
    assert report_p.exists()
    assert "## 1. Purpose" in report_p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Offline discipline
# ---------------------------------------------------------------------------

def test_wrapper_has_no_requests_and_no_production_imports():
    source = Path(e28.__file__).read_text(encoding="utf-8")
    assert "import requests" not in source
    assert "from requests" not in source
    assert "import server" not in source
    assert "from server" not in source
    assert "e19_coding_generalization.json" not in source
    assert "e20_counterfactual_history" not in source