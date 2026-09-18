"""Offline tests for E17 (Phase 13).

No Ollama, no network: the coder and summarizer generator are injected fakes, and
embeddings are disabled (lexical retrieval). These tests cover the parts most
likely to silently break:

* task-suite determinism and gold-patch harness correctness,
* that every method's retrieved historical context respects the fixed budget,
* that no-history vs oracle (direct-history) diagnostics behave as designed,
* leakage detection flagging test logic but not shared imports,
* failure-classification precedence,
* aggregation, verdict and the 24-section report renderer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.coding_task_suite import build_task, list_task_ids, self_test
from experiments import coding_benchmark as cb
from experiments import e17_coding_capability as e17


TASK_IDS = list_task_ids()
HISTORY_TURNS = 600


class GoldCoder:
    """Injected coder that always returns the task's gold patch."""

    model = "gold"

    def __init__(self, patch: str):
        self.patch = patch

    def generate(self, prompt):
        return {"text": "```diff\n" + self.patch + "\n```",
                "prompt_tokens": 0, "output_tokens": 0, "latency_ms": 0.0}


def _gold(task_id: str) -> str:
    base = Path(__file__).resolve().parent.parent / "data" / "coding_tasks" / task_id
    return (base / "gold.patch").read_text()


# ---------------------------------------------------------------------------
# Task suite
# ---------------------------------------------------------------------------

def test_task_suite_self_test():
    self_test()


def test_task_suite_is_deterministic_and_history_dependent():
    for tid in TASK_IDS:
        t = build_task(tid, seed=3)
        assert len(t.history) == 600
        assert len(t.gold_facts) >= 2
        # history dependence: at least one critical fact is >300 turns old
        assert any((t.history_turns - f.turn_index) > 300 for f in t.gold_facts)
        # corrections (where present) reference an obsolete fact id
        ids = set(t.gold_fact_ids)
        for c in t.corrections:
            assert c.current_fact_id in ids
            assert c.obsolete_fact_id in t.obsolete_fact_ids


# ---------------------------------------------------------------------------
# Gold-patch harness correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_id", TASK_IDS)
def test_gold_patch_passes_every_task(task_id, tmp_path):
    task = build_task(task_id, seed=1)
    rec = cb.run_method_run(
        task=task, seed=1, historical_budget=256,
        method=cb.build_method("no_history", model="gold", endpoint="http://x",
                               embed_fn=None, embedding_model="", task=task),
        coder=GoldCoder(_gold(task_id)),
        work_root=tmp_path / f"gold_{task_id}")
    assert rec["patch_applied"], rec["failure_class"]
    assert rec["final_success"], rec["failure_class"]


# ---------------------------------------------------------------------------
# Complete-file edit blocks (primary editing format)
# ---------------------------------------------------------------------------

def test_extract_and_apply_file_edits(tmp_path):
    text = (
        "sure, here is the fix\n"
        "### FILE: pkg/mod.py\n"
        "def f():\n"
        "    return 1\n"
        "### END FILE\n"
        "### FILE: pkg/new.py\n"
        "VALUE = 2\n"
        "### END FILE\n"
    )
    edits = cb.extract_file_edits(text)
    assert set(edits) == {"pkg/mod.py", "pkg/new.py"}
    assert edits["pkg/mod.py"] == "def f():\n    return 1\n"

    ok, method = cb.apply_edits(tmp_path, edits)
    assert ok and method == "file_replace"
    assert (tmp_path / "pkg" / "mod.py").read_text() == "def f():\n    return 1\n"
    assert (tmp_path / "pkg" / "new.py").read_text() == "VALUE = 2\n"


def test_extract_file_edits_tolerates_real_model_format(tmp_path):
    # observed shape: ### FILE: header, fenced body, NO ### END FILE, then prose
    text = (
        "### FILE: pkg/mod.py\n"
        "```python\n"
        "def f():\n"
        "    return 2\n"
        "```\n"
        "I have implemented the function as requested.\n"
    )
    edits = cb.extract_file_edits(text)
    assert edits == {"pkg/mod.py": "def f():\n    return 2\n"}


def test_apply_edits_rejects_path_traversal(tmp_path):
    ok, _ = cb.apply_edits(tmp_path, {"../escape.py": "x = 1\n"})
    assert ok is False


# ---------------------------------------------------------------------------
# Budget enforcement across methods
# ---------------------------------------------------------------------------

def test_every_method_respects_historical_budget():
    budget = 256
    methods = ["raw_clipped", "sliding_window", "llm_summarization",
               "vanilla_rag", "adaptive"]
    for tid in TASK_IDS[:2]:
        task = build_task(tid, seed=1, history_turns=HISTORY_TURNS)
        history = cb.HistoryBundle(task.task_id, task.history, task.history_text)
        for name in methods:
            fn = (lambda p, mx: "running summary of constraints and corrections"
                  ) if name == "llm_summarization" else None
            method = cb.build_method(name, model="m", endpoint="http://x",
                                     embed_fn=None, embedding_model="",
                                     summarizer_generate_fn=fn, task=task)
            prepared = method.prepare(history, budget, seed=1)
            ctx = method.retrieve(prepared, task.task_prompt, budget)
            assert cb.count_tokens(ctx) <= budget, (tid, name, cb.count_tokens(ctx))


# ---------------------------------------------------------------------------
# Diagnostics: no history vs oracle direct history
# ---------------------------------------------------------------------------

def test_no_history_and_direct_history_diagnostics(tmp_path):
    task = build_task("cache_readonly", seed=1, history_turns=HISTORY_TURNS)
    history = cb.HistoryBundle(task.task_id, task.history, task.history_text)

    noh = cb.build_method("no_history", model="m", endpoint="http://x",
                          embed_fn=None, embedding_model="", task=task)
    noh_p = noh.prepare(history, 256, seed=1)
    noh_ctx = noh.retrieve(noh_p, task.task_prompt, 256)
    noh_diag = cb.memory_diagnostics(task, noh_p, noh_ctx)
    assert noh_ctx == ""
    assert noh_diag["critical_fact_recall"] == 0.0

    direct = cb.build_method("direct_history", model="m", endpoint="http://x",
                             embed_fn=None, embedding_model="", task=task)
    d_p = direct.prepare(history, 256, seed=1)
    d_ctx = direct.retrieve(d_p, task.task_prompt, 256)
    # diagnostics read fact ids from the method's last_context_ids
    prepared_ctx = cb.PreparedMemory(
        method=d_p.method, memory_text=d_p.memory_text, memory_ids=d_p.memory_ids,
        memory_tokens=d_p.memory_tokens, build=d_p.build, payload=d_p.payload)
    prepared_ctx.context_ids = set(direct.last_context_ids)
    d_diag = cb.memory_diagnostics(task, prepared_ctx, d_ctx)
    assert d_diag["critical_fact_recall"] == 1.0
    assert d_diag["correction_recall"] == 1.0
    assert d_diag["obsolete_fact_exposure"] == 0.0


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------

def test_leakage_check_flags_test_logic_not_shared_imports():
    task = build_task("cache_readonly", seed=1)
    hidden = task.hidden_test_path.read_text()

    import_lines = [ln for ln in hidden.splitlines()
                    if ln.strip().startswith("import ") or ln.strip().startswith("from ")]
    logic_lines = [ln.strip() for ln in hidden.splitlines()
                   if "assert" in ln and len(ln.strip()) >= 20]

    clean_prompt = "\n".join(import_lines) + "\nplease implement set_cached"
    assert not cb.leakage_check(task, clean_prompt)["leakage_detected"]

    leaky_prompt = clean_prompt + "\n" + logic_lines[0]
    assert cb.leakage_check(task, leaky_prompt)["leakage_detected"]

    assert cb.leakage_check(task, "run test_hidden.py")["leakage_detected"]


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------

def test_classify_failure_precedence():
    k = cb.classify_failure
    assert k(success=True, overflow=True, patch_valid=False, test_pass=True,
             gold_in_memory=False, gold_in_context=False, obsolete_exposed=True,
             has_correction=True, correction_in_context=False) is None
    assert k(success=False, overflow=True, patch_valid=True, test_pass=False,
             gold_in_memory=True, gold_in_context=True, obsolete_exposed=False,
             has_correction=False, correction_in_context=False) == "CONTEXT_OVERFLOW"
    assert k(success=False, overflow=False, patch_valid=False, test_pass=False,
             gold_in_memory=True, gold_in_context=True, obsolete_exposed=False,
             has_correction=False, correction_in_context=False) == "PATCH_INVALID"
    assert k(success=False, overflow=False, patch_valid=True, test_pass=False,
             gold_in_memory=False, gold_in_context=True, obsolete_exposed=False,
             has_correction=False, correction_in_context=False) == "MEMORY_MISS"
    assert k(success=False, overflow=False, patch_valid=True, test_pass=False,
             gold_in_memory=True, gold_in_context=False, obsolete_exposed=False,
             has_correction=False, correction_in_context=False) == "RETRIEVAL_MISS"
    assert k(success=False, overflow=False, patch_valid=True, test_pass=False,
             gold_in_memory=True, gold_in_context=True, obsolete_exposed=True,
             has_correction=True, correction_in_context=False) == "OBSOLETE_INFORMATION_USED"
    assert k(success=False, overflow=False, patch_valid=True, test_pass=False,
             gold_in_memory=True, gold_in_context=True, obsolete_exposed=False,
             has_correction=True, correction_in_context=False) == "CORRECTION_MISSED"
    assert k(success=False, overflow=False, patch_valid=True, test_pass=False,
             gold_in_memory=True, gold_in_context=True, obsolete_exposed=False,
             has_correction=False, correction_in_context=False) == "CODING_ERROR"


# ---------------------------------------------------------------------------
# Aggregation / verdict / report
# ---------------------------------------------------------------------------

def _rec(task_id, seed, budget, method, success, first=False, overflow=False,
         patch_valid=True):
    return {
        "task_id": task_id, "seed": seed, "historical_budget": budget,
        "method": method, "final_success": success, "first_pass_success": first,
        "patch_valid": patch_valid, "patch_applied": patch_valid,
        "historical_context_tokens": budget,
        "total_prompt_tokens": 6000,
        "coding_attempts": 1,
        "failure_class": None if success else ("CONTEXT_OVERFLOW" if overflow
                                               else "HIDDEN_TEST_FAILURE"),
        "diagnostics": {
            "critical_fact_recall": 1.0 if success else 0.5,
            "critical_fact_in_memory": 1.0,
            "correction_recall": 1.0 if success else 0.0,
            "negative_constraint_recall": 1.0,
            "long_range_fact_recall": 1.0 if success else 0.0,
            "obsolete_fact_exposure": 0.0,
        },
    }


def _synthetic_result():
    records = []
    methods = ["adaptive", "raw_clipped"]
    for budget in (256, 1024):
        for tid in ("t1", "t2"):
            for i, m in enumerate(methods):
                records.append(_rec(tid, 1, budget, m, success=(m == "adaptive"
                                                                or i == 0 and budget == 1024)))
    diagnostics = [_rec(tid, 1, 1024, "no_history", success=False, first=False)
                   for tid in ("t1", "t2")]
    cfg = {"mode": "pilot", "tasks": ["t1", "t2"], "seeds": [1],
           "budgets": [256, 1024], "methods": methods, "model": "fake",
           "embedding_model": "", "use_embeddings": False,
           "max_output_tokens": 700, "temperature": 0.1,
           "tokenizer": cb.TOKENIZER_NAME, "generated_at": 0.0}
    result = {"config": cfg, "records": records, "diagnostics": diagnostics,
              "gold_checks": [{"task_id": "t1", "final_success": True}]}
    result["aggregate"] = e17.aggregate(records, diagnostics)
    result["gates"] = {"all_passed": True}
    result["verdict"] = e17.classify_verdict(result)
    return result


def test_aggregate_shape_and_diagnostics():
    agg = e17.aggregate(_synthetic_result()["records"],
                        _synthetic_result()["diagnostics"])
    assert agg["by_method_budget"]["adaptive"]["256"]["runs"] == 2
    assert agg["overall"]["adaptive"]["runs"] == 4
    assert set(agg["paired_vs_adaptive"]) == {"raw_clipped"}
    comp = agg["paired_vs_adaptive"]["raw_clipped"]
    assert comp["pairs"] == 4
    assert "mean_diff" in comp and len(comp["mean_diff_ci95"]) == 2


def test_verdict_requires_gates_and_positive_ci():
    result = _synthetic_result()
    result["gates"] = {"all_passed": False}
    v = e17.classify_verdict(result)
    assert v["adaptive_advances"] is False

    result["gates"] = {"all_passed": True}
    v = e17.classify_verdict(result)
    # synthetic adaptive strictly beats raw_clipped on every paired cell
    assert v["adaptive_beats_raw_clipped"] is True
    assert "adaptive_advances" in v


def test_generate_report_has_all_24_sections():
    report = e17.generate_report(_synthetic_result())
    for i in range(1, 25):
        assert f"## {i}." in report, f"missing section {i}"
    assert "adaptive_advances" in report
    assert len(report) > 3000


def test_output_paths_are_under_repo():
    # /tmp is unavailable; all E17 artifacts must live under the repo.
    repo = Path(__file__).resolve().parent.parent
    assert str(e17.OUT_JSON).startswith(str(repo))
    assert str(e17.OUT_REPORT).startswith(str(repo))
    assert str(e17.WORK_ROOT).startswith(str(repo))
