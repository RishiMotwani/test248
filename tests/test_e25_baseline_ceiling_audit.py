"""Phase 20 / E25 tests: offline audit of the E19 baseline ceiling and budget geometry.

These tests exercise ``experiments/e25_e19_baseline_ceiling_audit.py`` purely:
no Ollama, no embeddings, no network, no production memory-optimizer code, and
no writes to the immutable E19 artifact. The audit is deterministic over the
locked ``e19_coding_generalization_full_repaired.json`` plus the in-repo
counterfactual fixtures.
"""

import json
import tempfile
from pathlib import Path

import pytest

from experiments import e25_e19_baseline_ceiling_audit as e25


@pytest.fixture(scope="module")
def e19_artifact():
    return e25.load_e19_records()


@pytest.fixture(scope="module")
def e25_result(e19_artifact):
    """The full assembled E25 result dictionary, without writing any files."""
    return e25._assemble(e19_artifact)


@pytest.fixture(scope="module")
def primary_records(e19_artifact):
    return e25.filter_primary_records(e19_artifact)


@pytest.fixture(scope="module")
def primary_index(primary_records):
    return e25.validate_primary_grid(primary_records)


# ---------------------------------------------------------------------------
# 1-3. Module shape: intended methods, budgets, tasks, offline discipline
# ---------------------------------------------------------------------------


def test_e25_intended_methods_budgets_tasks():
    assert e25.METHODS == (
        "raw_clipped", "sliding_window", "llm_summarization", "vanilla_rag", "adaptive"
    )
    assert e25.BUDGETS == (256, 512, 1024)
    assert e25.SEEDS == (1, 2, 3)
    assert e25.PRIMARY_TASKS == ("routing_policy", "retry_policy", "serialization_policy")
    assert e25.PRIMARY_GRID_CELLS == 135


def test_grid_constants_match_artifact(e19_artifact):
    cfg = e19_artifact["config"]
    assert list(cfg["methods"]) == list(e25.METHODS)
    assert list(cfg["budgets"]) == list(e25.BUDGETS)
    assert list(cfg["seeds"]) == list(e25.SEEDS)
    assert list(cfg["primary_tasks"]) == list(e25.PRIMARY_TASKS)


# ---------------------------------------------------------------------------
# 4-5. Loading and grid validation
# ---------------------------------------------------------------------------


def test_source_artifact_loads_224_records(e19_artifact):
    assert e19_artifact["experiment"] == "e19_coding_generalization"
    assert len(e19_artifact["records"]) == 224


def test_primary_grid_is_exactly_135_and_valid(primary_records, primary_index):
    assert len(primary_records) == 135
    assert len(primary_index) == 135
    for t in e25.PRIMARY_TASKS:
        for s in e25.SEEDS:
            for b in e25.BUDGETS:
                for m in e25.METHODS:
                    assert (t, s, b, m) in primary_index


def test_primary_grid_rejects_partial_grid():
    artifact = e25.load_e19_records()
    primary = e25.filter_primary_records(artifact)
    with pytest.raises(RuntimeError):
        e25.validate_primary_grid(primary[:-1])
    with pytest.raises(RuntimeError):
        e25.validate_primary_grid(primary + primary[:1])


# ---------------------------------------------------------------------------
# 6. Correction-state classification
# ---------------------------------------------------------------------------


def test_correction_state_mapping():
    assert e25.classify_correction_state(1.0, 0.0) == "CLEAN_CURRENT"
    assert e25.classify_correction_state(1.0, 1.0) == "CURRENT_PLUS_OBSOLETE"
    assert e25.classify_correction_state(0.0, 1.0) == "OBSOLETE_ONLY"
    assert e25.classify_correction_state(0.0, 0.0) == "NEITHER"


def test_correction_state_rejects_unknown_pair():
    with pytest.raises(ValueError):
        e25.classify_correction_state(0.5, 0.5)
    with pytest.raises(ValueError):
        e25.classify_correction_state(None, 1.0)


# ---------------------------------------------------------------------------
# 7. Budget geometry and binding
# ---------------------------------------------------------------------------


def test_budget_geometry_matches_locked_e19_findings(e25_result):
    g = e25_result["budget_geometry"]["by_method_budget"]
    assert g["adaptive"][256]["mean_utilization"] < 0.50
    assert g["vanilla_rag"][256]["mean_utilization"] < 0.50
    assert g["raw_clipped"][1024]["mean_utilization"] > 0.90
    assert g["raw_clipped"][256]["fraction_budget_binding"] == 1.0
    assert g["adaptive"][256]["fraction_budget_binding"] == 0.0
    assert e25_result["budget_geometry"]["primary_cells"] == 135


def test_budget_binding_uses_0_90_threshold(primary_records, primary_index):
    elasticity = e25.build_budget_elasticity(primary_records, primary_index)
    binding = e25.build_budget_binding_by_method(primary_records, elasticity)
    for m in ("adaptive", "vanilla_rag", "sliding_window", "llm_summarization"):
        assert binding[m]["by_budget"][256]["fraction_budget_binding"] == 0.0
    assert binding["raw_clipped"]["by_budget"][256]["fraction_budget_binding"] == 1.0


# ---------------------------------------------------------------------------
# 8. Correction states over the real grid
# ---------------------------------------------------------------------------


def test_correction_state_analysis_matches_locked_e19(e25_result):
    dist = e25_result["correction_state_analysis"]["per_method_correction_state_distribution"]
    assert dist["adaptive"] == {"CLEAN_CURRENT": 27}
    assert dist["vanilla_rag"] == {"OBSOLETE_ONLY": 27}
    assert dist["raw_clipped"] == {"NEITHER": 27}
    assert dist["sliding_window"] == {"NEITHER": 27}
    success = e25_result["correction_state_analysis"]["per_method_state_success"]
    assert success["adaptive"]["CLEAN_CURRENT"]["success_rate"] == 1.0
    assert success["vanilla_rag"]["OBSOLETE_ONLY"]["success_rate"] >= 0.80


# ---------------------------------------------------------------------------
# 9. Vanilla_rag ceiling failure
# ---------------------------------------------------------------------------


def test_vanilla_failure_is_the_single_expected_cell(e25_result):
    v = e25_result["vanilla_failures"]
    assert v["cells"] == 27
    assert v["successes"] == 26
    assert v["failed_cells"] == 1
    failure = v["failures"][0]
    assert failure["task_id"] == "serialization_policy"
    assert failure["seed"] == 2
    assert failure["historical_budget"] == 1024
    assert failure["failure_class"] == "OBSOLETE_INFORMATION_USED"
    assert failure["correction_state"] == "OBSOLETE_ONLY"


# ---------------------------------------------------------------------------
# 10. Context-hash (SHA-based) stability
# ---------------------------------------------------------------------------


def test_adaptive_and_vanilla_are_context_stable_via_sha(primary_records, primary_index):
    elasticity = e25.build_budget_elasticity(primary_records, primary_index)
    for m in ("adaptive", "vanilla_rag"):
        tracks = [v for v in elasticity["tracks"].values() if v["method"] == m]
        assert len(tracks) == 9
        assert all(v["budget_context_stable"] for v in tracks)
        assert all(v["distinct_contexts_across_budgets"] == 1 for v in tracks)


def test_adaptive_and_vanilla_contexts_never_tied(primary_records, primary_index):
    analysis = e25.build_context_hash_analysis(primary_records, primary_index)
    for b in ("256", "512", "1024"):
        eq = analysis["adaptive_vs_vanilla_context_equality_by_budget"][b]
        assert eq["equals"] == 0
        assert eq["differs"] == 9


# ---------------------------------------------------------------------------
# 11. Diagnostic flags
# ---------------------------------------------------------------------------


def test_all_four_diagnostic_flags_fire_and_are_booleans(e25_result):
    flags = e25_result["diagnostic_flags"]
    for key in ("flag_a_vanilla_obsolete_only_success_rate",
                "flag_b_budget_non_binding",
                "flag_c_vanilla_near_ceiling",
                "flag_d_adaptive_advances_false"):
        assert key in flags
        assert isinstance(flags[key], bool)
    assert flags["flag_a_vanilla_obsolete_only_success_rate"] is True
    assert flags["flag_b_budget_non_binding"] is True
    assert flags["flag_c_vanilla_near_ceiling"] is True
    assert flags["flag_d_adaptive_advances_false"] is True


# ---------------------------------------------------------------------------
# 12. Report rendering
# ---------------------------------------------------------------------------


def test_report_renders_required_sections_and_tables(e25_result):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "e25_report.md"
        e25.write_report(e25_result, path)
        text = path.read_text(encoding="utf-8")
    required_sections = [
        "# E25: E19 baseline ceiling and budget-geometry audit",
        "## 1. Purpose", "## 2. Method", "## 3. Source artifact", "## 4. Scope",
        "## 5. Assertions", "## 6. Budget geometry (Table A)",
        "## 11. Success by correction state (Table B)",
        "## 8. Budget binding by method (Table C, right columns)",
        "## 12. Task x budget analysis (Table D)",
        "## 13. Vanilla_rag failure analysis (Table E)",
        "## 17. Next-benchmark requirements (constraints only)",
    ]
    for section in required_sections:
        assert section in text
    for marker in ("---|", "| Method | Budget | Mean tokens", "OBSOLETE_ONLY"):
        assert marker in text


# ---------------------------------------------------------------------------
# 13. Machine artifact shape
# ---------------------------------------------------------------------------


def test_json_top_level_keys(e25_result):
    for key in (
        "experiment", "source_artifact", "source_primary_cells", "scope",
        "budget_geometry", "correction_state_analysis", "task_budget_analysis",
        "vanilla_failures", "context_hash_analysis", "diagnostic_flags",
        "next_benchmark_requirements", "conclusion",
    ):
        assert key in e25_result
    assert e25_result["experiment"] == "E25"
    assert e25_result["source_primary_cells"] == 135


# ---------------------------------------------------------------------------
# 14. Offline discipline: no production/LLM imports, no artifact writes
# ---------------------------------------------------------------------------


def test_e25_source_never_touches_servers_or_the_e19_artifact():
    source = Path(e25.__file__).read_text(encoding="utf-8")
    assert "OllamaCoder" not in source
    assert "import memory_optimizer" not in source
    assert "from memory_optimizer" not in source
    assert "ollama" not in source.lower()
    assert "import requests" not in source
    assert "from requests" not in source
    assert not any(line.strip().startswith("import ")
                   and "memory_optimizer" in line for line in source.splitlines())
    assert "SOURCE_ARTIFACT.write" not in source
    assert "open(" not in source  # artifact I/O uses Path.read_text/write_text only


def test_authored_output_footprint_is_exactly_two_files():
    source = Path(e25.__file__).read_text(encoding="utf-8")
    assert "OUT_JSON" in source
    assert "OUT_REPORT" in source
    assert "write_json(result, OUT_JSON)" in source
    assert "write_report(result, OUT_REPORT)" in source