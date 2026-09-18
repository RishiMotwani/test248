"""E17 — Phase 13: adaptive memory vs simpler context managers on long-horizon
coding tasks under a fixed historical-context budget.

Research question
-----------------
Does the adaptive memory system help a real coding model complete long-running
repository-editing tasks better than simpler ways of managing historical context
when the historical context supplied to the model is capped at a fixed budget?

The dependent variable is *whether the produced patch passes deterministic
hidden tests* (``final_success``). Historical-fact recall is measured too, but
only as a diagnostic explanation of success/failure — it is not the outcome.

Design (locked, see report sections 1-10 and brain.md D34)
---------------------------------------------------------
* 4 real tasks under ``data/coding_tasks``; each hidden test fails if a buried
  historical constraint is violated.
* Methods (only the history mechanism differs): ``raw_clipped``,
  ``sliding_window``, ``llm_summarization`` (real same-model summarizer),
  ``vanilla_rag`` (static cosine store, no decay/importance/eviction),
  ``adaptive`` (the production pipeline, ``retention=dual_score``,
  ``protect_corrections=True``). Diagnostics: ``no_history``, ``full_context``
  (labelled unconstrained), ``direct_history`` (oracle gold-fact context).
* Historical budgets: 256 / 512 / 1024 tokens.
* Ingestion is oracle-pre-extracted: E17 measures retention + retrieval +
  context allocation, not extraction cost.
* The coding model, prompt scaffold, retry policy (initial + one repair),
  temperature (0.1) and tokenizer (shared word count) are identical across arms.
* No gold labels, hidden tests or future turns ever reach a method; the harness
  asserts this per run (``leakage``).

Pilot gates (9) must all pass before the full grid is considered an experiment;
pilot results alone are labelled PILOT.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.coding_task_suite import build_task, list_task_ids  # noqa: E402
from experiments import coding_benchmark as cb  # noqa: E402
from experiments.statistics import bootstrap_ci_mean_diff, bootstrap_mean_ci  # noqa: E402
from memory_optimizer.tokenizer import TOKENIZER_NAME  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PILOT_TASKS = ["cache_readonly", "user_ids", "validation_pure"]
PILOT_SEEDS = [1, 2]
PILOT_BUDGETS = [256, 1024]
PILOT_METHODS = list(cb.METHOD_NAMES)

FULL_TASKS = list_task_ids()
FULL_SEEDS = [1, 2, 3]
FULL_BUDGETS = list(cb.HISTORICAL_BUDGETS)

DIAG_SEED = 1
DIAG_BUDGET = max(PILOT_BUDGETS)

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_ENDPOINT = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
MAX_OUTPUT_TOKENS = cb.MAX_OUTPUT_TOKENS

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_JSON = RESULTS_DIR / "e17_coding_capability.json"
OUT_REPORT = RESULTS_DIR / "e17_coding_capability_report.md"
WORK_ROOT = RESULTS_DIR / "e17_work"

GATE_NAMES = [
    "budget_pressure",
    "workspace_independence",
    "methods_differ",
    "real_summarization",
    "adaptive_production_path",
    "no_leakage",
    "history_dependence",
    "test_determinism",
    "patch_determinism",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _run_key(task_id: str, seed: int, budget: int, method: str) -> str:
    return f"{task_id}:{seed}:{budget}:{method}"


def _diag_key(task_id: str, method: str) -> str:
    return f"{task_id}:{method}"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _success(records: List[Dict]) -> Optional[float]:
    if not records:
        return None
    return round(sum(1 for r in records if r["final_success"]) / len(records), 4)


def _workspace_fingerprint(task) -> str:
    import hashlib
    h = hashlib.sha256()
    for p in sorted(task.workspace_path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(task.workspace_path)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _resolve_embed(use_embeddings: bool, embedding_model: str, endpoint: str):
    if not use_embeddings:
        return None
    from memory_optimizer.embeddings import embed_ollama
    return lambda texts: embed_ollama(texts, model=embedding_model, endpoint=endpoint)


# ---------------------------------------------------------------------------
# Coders
# ---------------------------------------------------------------------------

class StaticCoder:
    """Returns a fixed patch (used for gold sanity checks)."""

    def __init__(self, patch: str, model: str = "static"):
        self.patch = patch
        self.model = model

    def generate(self, prompt: str) -> Dict:
        return {"text": "```diff\n" + self.patch + "\n```",
                "prompt_tokens": 0, "output_tokens": 0, "latency_ms": 0.0}


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def run_one(task_id: str, seed: int, budget: int, method_name: str, *,
            coder, embed_fn, model: str, endpoint: str,
            embedding_model: str, summarizer_generate_fn=None) -> Dict:
    task = build_task(task_id, seed=seed)
    method = cb.build_method(
        method_name, model=model, endpoint=endpoint, embed_fn=embed_fn,
        embedding_model=embedding_model,
        summarizer_generate_fn=summarizer_generate_fn, task=task)
    root = WORK_ROOT / f"{task_id}_{seed}_{budget}_{method_name}"
    return cb.run_method_run(
        task=task, seed=seed, historical_budget=budget, method=method,
        coder=coder, work_root=root)


# ---------------------------------------------------------------------------
# Determinism / gold sanity checks
# ---------------------------------------------------------------------------

def check_gold(task_id: str, seed: int = 1) -> Dict:
    """The harness must be able to pass every task given the gold patch."""
    task = build_task(task_id, seed=seed)
    gold = (task.workspace_path.parent / "gold.patch").read_text()
    coder = StaticCoder(gold)
    root = WORK_ROOT / f"goldcheck_{task_id}"
    method = cb.build_method("no_history", model="static", endpoint="http://x",
                             embed_fn=None, embedding_model="", task=task)
    rec = cb.run_method_run(task=task, seed=seed, historical_budget=256,
                            method=method, coder=coder, work_root=root)
    return {"task_id": task_id, "final_success": rec["final_success"],
            "patch_applied": rec["patch_applied"]}


def check_test_determinism(task_id: str, seed: int = 1) -> Dict:
    """Hidden tests must be deterministic on the same (unpatched) workspace."""
    task = build_task(task_id, seed=seed)
    root = WORK_ROOT / f"det_{task_id}"
    repo = cb.prepare_workspace(task, root)
    ok1, out1, err1 = cb.run_hidden_tests(repo, task)
    repo2 = cb.prepare_workspace(task, root)
    ok2, out2, err2 = cb.run_hidden_tests(repo2, task)
    return {"task_id": task_id, "run1_pass": ok1, "run2_pass": ok2,
            "same_pass_fail": (ok1 == ok2),
            "same_output": (out1 == out2 and err1 == err2),
            "deterministic": (ok1 == ok2)}


def check_patch_determinism(record: Dict) -> Dict:
    """Applying one real patch twice to fresh workspaces must give one result."""
    patch = record.get("patch")
    if not patch:
        return {"ok": False, "reason": "no patch in record"}
    task = build_task(record["task_id"], seed=record["seed"])
    results = []
    for i in range(2):
        root = WORK_ROOT / f"pdet_{record['task_id']}_{i}"
        repo = cb.prepare_workspace(task, root)
        applied, method = cb.apply_patch(repo, patch)
        if not applied:
            results.append({"applied": False, "passed": False})
            continue
        passed, _out, _err = cb.run_hidden_tests(repo, task)
        results.append({"applied": True, "passed": passed})
    deterministic = results[0] == results[1]
    return {"ok": deterministic, "results": results,
            "task_id": record["task_id"], "method": record["method"]}


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def compute_gates(records: List[Dict], diagnostics: List[Dict],
                  task_ids: List[str]) -> Dict:
    pilot = [r for r in records if r["method"] in PILOT_METHODS]

    # 1. budget pressure: raw history must exceed the largest budget (so clipping
    #    is a real constraint) and at least one run must fill its budget. Ranked
    #    methods legitimately select a bounded set regardless of budget, so growth
    #    is reported but not required of every method.
    budgets = sorted({r["historical_budget"] for r in pilot})
    lo_b, hi_b = (budgets[0], budgets[-1]) if budgets else (256, 1024)
    raw_lo = _mean([r["historical_context_tokens"] for r in pilot
                    if r["method"] == "raw_clipped" and r["historical_budget"] == lo_b])
    raw_hi = _mean([r["historical_context_tokens"] for r in pilot
                    if r["method"] == "raw_clipped" and r["historical_budget"] == hi_b])
    raw_grows = (raw_lo is not None and raw_hi is not None and raw_hi > raw_lo + 20)
    fc = [d for d in diagnostics if d["method"] == "full_context"]
    history_exceeds = bool(fc) and max(d["historical_context_tokens"] for d in fc) > hi_b
    fills = any(r["historical_context_tokens"] >= 0.6 * r["historical_budget"] for r in pilot)
    growth = {}
    for m in {r["method"] for r in pilot}:
        growth[m] = {
            "tokens_lo": _mean([r["historical_context_tokens"] for r in pilot
                                if r["method"] == m and r["historical_budget"] == lo_b]),
            "tokens_hi": _mean([r["historical_context_tokens"] for r in pilot
                                if r["method"] == m and r["historical_budget"] == hi_b]),
        }
    g1 = raw_grows and history_exceeds and fills

    # 2. workspace independence: distinct base workspaces across tasks
    fps = {}
    for tid in task_ids:
        t = build_task(tid, seed=1)
        fps[tid] = _workspace_fingerprint(t)
    g2 = len(set(fps.values())) == len(fps)

    # 3. methods differ: at least one (task,seed,budget) has >=3 distinct contexts
    groups: Dict[str, set] = defaultdict(set)
    for r in pilot:
        groups[f"{r['task_id']}:{r['seed']}:{r['historical_budget']}"].add(
            (r["method"], r["context_sha"]))
    max_distinct = max((len({sha for _, sha in v}) for v in groups.values()), default=0)
    g3 = max_distinct >= 3

    # 4. real summarization: the summarizer actually called the model
    summ = [r for r in pilot if r["method"] == "llm_summarization"]
    g4 = bool(summ) and sum(r.get("summary_update_calls", 0) for r in summ) > 0

    # 5. adaptive production path
    adapt = [r for r in pilot if r["method"] == "adaptive"]
    g5 = bool(adapt) and all(r.get("uses_production_pipeline") for r in adapt)

    # 6. no leakage
    leaks = [r for r in records if (r.get("leakage") or {}).get("leakage_detected")]
    g6 = len(leaks) == 0

    # 7. history dependence: no-history fails but the oracle (direct_history, the
    #    designed upper bound) succeeds -> the task needs historical memory AND is
    #    solvable when the right facts are supplied. Using direct_history (not a
    #    grid method) is the correct reachability test.
    no_hist = {r["task_id"]: r for r in diagnostics if r["method"] == "no_history"}
    direct = {r["task_id"]: r for r in diagnostics if r["method"] == "direct_history"}
    history_dependent = 0
    per_task = {}
    for tid in task_ids:
        nh = no_hist.get(tid)
        dh = direct.get(tid)
        nh_ok = bool(nh and nh["final_success"])
        dh_ok = bool(dh and dh["final_success"])
        dependent = (not nh_ok) and dh_ok
        per_task[tid] = {"no_history_success": nh_ok,
                         "oracle_direct_history_success": dh_ok,
                         "history_dependent": dependent}
        if dependent:
            history_dependent += 1
    need = max(1, len(task_ids) // 2)
    g7 = history_dependent >= need

    # 8. test determinism
    det = [check_test_determinism(tid) for tid in task_ids]
    g8 = all(d["deterministic"] for d in det)

    # 9. patch determinism
    with_patch = [r for r in pilot if r.get("patch")]
    pdet = check_patch_determinism(with_patch[0]) if with_patch else {"ok": False,
                                                                     "reason": "no patch"}
    g9 = bool(pdet.get("ok"))

    gates = {
        "budget_pressure": {"passed": g1, "growth": growth, "raw_grows": raw_grows,
                            "history_exceeds_max_budget": history_exceeds,
                            "some_run_fills_60pct": fills},
        "workspace_independence": {"passed": g2, "fingerprints": fps},
        "methods_differ": {"passed": g3, "max_distinct_contexts": max_distinct},
        "real_summarization": {"passed": g4,
                               "summary_update_calls": sum(r.get("summary_update_calls", 0)
                                                           for r in summ)},
        "adaptive_production_path": {"passed": g5, "adaptive_runs": len(adapt)},
        "no_leakage": {"passed": g6, "leaking_runs": len(leaks)},
        "history_dependence": {"passed": g7, "history_dependent_tasks": history_dependent,
                               "per_task": per_task},
        "test_determinism": {"passed": g8, "checks": det},
        "patch_determinism": {"passed": g9, "check": pdet},
    }
    gates["all_passed"] = all(g["passed"] for k, g in gates.items() if k in GATE_NAMES)
    return gates


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(records: List[Dict], diagnostics: List[Dict]) -> Dict:
    by_method_budget: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_method_budget[r["method"]][str(r["historical_budget"])].append(r)

    diag_keys = ["critical_fact_recall", "critical_fact_in_memory",
                 "correction_recall", "negative_constraint_recall",
                 "long_range_fact_recall", "obsolete_fact_exposure"]
    out: Dict[str, Dict] = {}
    for m, by_b in by_method_budget.items():
        out[m] = {}
        for b, rows in by_b.items():
            out[m][b] = {
                "runs": len(rows),
                "success_rate": _success(rows),
                "first_pass_rate": round(
                    sum(1 for r in rows if r["first_pass_success"]) / len(rows), 4),
                "repair_recovery_rate": round(
                    sum(1 for r in rows if r["final_success"] and not r["first_pass_success"])
                    / len(rows), 4),
                "mean_context_tokens": _mean([r["historical_context_tokens"] for r in rows]),
                "mean_total_prompt_tokens": _mean([r["total_prompt_tokens"] for r in rows]),
                "mean_coding_attempts": _mean([r["coding_attempts"] for r in rows]),
                "diagnostics": {k: _mean([r["diagnostics"].get(k) for r in rows])
                                for k in diag_keys},
                "failure_classes": _count_failures(rows),
            }

    # paired comparisons adaptive vs each other method, over shared cells
    def cell_success(method):
        d = {}
        for r in records:
            if r["method"] == method:
                d[(r["task_id"], r["seed"], r["historical_budget"])] = int(r["final_success"])
        return d

    adapt = cell_success("adaptive")
    comparisons = {}
    for m in PILOT_METHODS:
        if m == "adaptive":
            continue
        other = cell_success(m)
        keys = sorted(set(adapt) & set(other))
        if len(keys) < 2:
            continue
        a = [adapt[k] for k in keys]
        b = [other[k] for k in keys]
        comparisons[m] = {
            "pairs": len(keys),
            "adaptive_success": round(sum(a) / len(a), 4),
            "other_success": round(sum(b) / len(b), 4),
            "mean_diff": round(sum(x - y for x, y in zip(a, b)) / len(a), 4),
            "mean_diff_ci95": bootstrap_ci_mean_diff(a, b),
            "adaptive_ci95": bootstrap_mean_ci(a),
            "other_ci95": bootstrap_mean_ci(b),
        }

    overall = {}
    for m in sorted({r["method"] for r in records}):
        rows = [r for r in records if r["method"] == m]
        overall[m] = {"runs": len(rows), "success_rate": _success(rows),
                      "first_pass_rate": round(
                          sum(1 for r in rows if r["first_pass_success"]) / len(rows), 4)}

    diag = {}
    for r in diagnostics:
        diag[_diag_key(r["task_id"], r["method"])] = {
            "task_id": r["task_id"], "method": r["method"],
            "final_success": r["final_success"],
            "historical_context_tokens": r["historical_context_tokens"],
            "failure_class": r["failure_class"],
        }

    return {"by_method_budget": out, "paired_vs_adaptive": comparisons,
            "overall": overall, "diagnostics": diag}


def _count_failures(rows: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["failure_class"] or "SUCCESS"] += 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

MARGIN = 0.0


def classify_verdict(result: Dict) -> Dict:
    gates = result.get("gates") or {}
    agg = result.get("aggregate") or {}
    comp = agg.get("paired_vs_adaptive") or {}
    gates_ok = bool(gates.get("all_passed"))

    def beats(method, margin=MARGIN):
        c = comp.get(method)
        if not c:
            return False, None
        lo = c["mean_diff_ci95"][0]
        return (lo is not None and lo > margin), c

    b_raw, c_raw = beats("raw_clipped")
    b_win, c_win = beats("sliding_window")
    b_sum, c_sum = beats("llm_summarization")

    def ci(method):
        c = comp.get(method)
        return c["mean_diff_ci95"] if c else [None, None]

    c_rag = ci("vanilla_rag")
    b_rag = c_rag[0] is not None and c_rag[0] > MARGIN

    verdict = {
        "pilot_only": result.get("config", {}).get("mode") == "pilot",
        "gates_all_passed": gates_ok,
        "adaptive_beats_raw_clipped": b_raw,
        "adaptive_beats_sliding_window": b_win,
        "adaptive_beats_llm_summarization": b_sum,
        "comparisons": comp,
        "criteria": {
            "gates_pass": gates_ok,
            "adaptive_vs_raw_clipped_ci95": c_raw["mean_diff_ci95"] if c_raw else [None, None],
            "adaptive_vs_sliding_window_ci95": c_win["mean_diff_ci95"] if c_win else [None, None],
            "adaptive_vs_llm_summarization_ci95": c_sum["mean_diff_ci95"] if c_sum else [None, None],
            "adaptive_vs_vanilla_rag_ci95": c_rag,
        },
    }
    if not gates_ok:
        verdict["adaptive_advances"] = False
        verdict["rationale"] = ("Pilot gates did not all pass; the grid is not a valid "
                                "experiment and adaptive cannot be said to advance.")
    else:
        advances = b_raw and b_win and b_sum and b_rag
        verdict["adaptive_advances"] = bool(advances)
        verdict["rationale"] = (
            "adaptive_advances requires every paired 95% CI lower bound vs "
            "raw_clipped / sliding_window / llm_summarization / vanilla_rag to "
            "exceed 0 (adaptive strictly better) AND all pilot gates to pass."
        )
    return verdict


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(x, nd=3):
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def generate_report(result: Dict) -> str:
    cfg = result["config"]
    agg = result["aggregate"]
    gates = result.get("gates") or {}
    verdict = result.get("verdict") or {}
    L: List[str] = []
    L.append("# E17 Coding Capability Report (Phase 13)")
    L.append("")
    L.append("Does adaptive memory help a real coding model complete "
             "long-horizon repository-editing tasks better than simpler "
             "historical-context managers when the historical context is capped "
             "at a fixed budget? Primary outcome: **hidden-test patch pass**.")
    L.append("")

    L.append("## 1. Purpose & Research Question")
    L.append("")
    L.append("Every earlier experiment measured memory recall or injected-context "
             "answerability. E17 measures the downstream quantity the memory system "
             "exists to support: whether a coding model produces, in a real "
             "workspace, a patch that passes deterministic hidden tests after a "
             "~600-turn conversation whose constraints no longer fit any small "
             "context window. Historical-fact recall is diagnostic only and never "
             "the dependent variable.")
    L.append("")

    L.append("## 2. Hypothesis & Falsification Criteria")
    L.append("")
    L.append("H1: at a fixed historical-context budget, the adaptive pipeline "
             "(`dual_score` retention + similarity/importance retrieval + "
             "correction-aware compression) yields a higher hidden-test pass rate "
             "than `raw_clipped`, `sliding_window`, `llm_summarization` and "
             "`vanilla_rag`. Falsified if any paired 95% bootstrap CI on the mean "
             "success difference has a lower bound <= 0, or if the pilot gates do "
             "not all pass.")
    L.append("")

    L.append("## 3. Methodology / Controlled Comparison")
    L.append("")
    L.append("One code path runs every arm (`experiments/coding_benchmark.py`): "
             "the method builds a historical memory representation, retrieves a "
             "context string bounded by the budget, and the harness assembles the "
             "prompt, calls the coding model, writes the model's complete-file "
             "edit blocks into a fresh workspace copy (a unified-diff fallback is "
             "kept for robustness) and runs "
             "the hidden tests. The model, prompt "
             "scaffold, temperature (0.1), retry policy (initial + one repair) and "
             "tokenizer are identical across arms; only the history mechanism "
             "differs.")
    L.append("")

    L.append("## 4. Task Suite Description")
    L.append("")
    L.append("| task | title | critical facts | corrections | negative constraints |")
    L.append("| --- | --- | --- | --- | --- |")
    for tid in FULL_TASKS:
        t = build_task(tid, seed=1)
        L.append(f"| {tid} | {t.title} | {len(t.gold_facts)} | {len(t.corrections)} | "
                 f"{sum(1 for f in t.gold_facts if f.kind == 'negative')} |")
    L.append("")
    L.append(f"Ingestion mode: `{build_task(FULL_TASKS[0], seed=1).metadata.get('ingestion')}` "
             "(oracle-pre-extracted): E17 measures retention + retrieval + context "
             "allocation, not extraction. Each task's workspace is a small real "
             "package; its hidden test fails if a buried constraint is violated and "
             "is never seen by the model.")
    L.append("")

    L.append("## 5. Historical Dependence Design")
    L.append("")
    L.append("Each task spreads critical facts across a ~600-turn transcript among "
             "distractors; >300-turn-old facts and explicit corrections are present. "
             "The `no_history` diagnostic (section 20) quantifies whether the task "
             "is genuinely history-dependent. Hidden-test pass requires the model to "
             "respect constraints that the current workspace alone does not make "
             "obvious.")
    L.append("")

    L.append("## 6. Memory Methods Under Test")
    L.append("")
    L.append("| method | historical-context mechanism |")
    L.append("| --- | --- |")
    L.append("| raw_clipped | newest raw turns that fit the budget; no retrieval |")
    L.append("| sliding_window | last 10 raw turns (production `SlidingWindowBaseline`) |")
    L.append("| llm_summarization | running summary by the same model every 50 turns |")
    L.append("| vanilla_rag | static fact store, pure cosine top-k fit to budget |")
    L.append("| adaptive | production pipeline, `dual_score` retention, `protect_corrections=True` |")
    L.append("")
    L.append("Diagnostics (not competitors): `no_history`, `full_context` "
             "(unconstrained upper bound), `direct_history` (oracle gold-fact "
             "context).")
    L.append("")

    L.append("## 7. Fixed Budgets & Adaptive Allocation Policy")
    L.append("")
    L.append(f"Historical budgets: {cfg.get('budgets')} tokens (shared word-count "
             f"tokenizer `{TOKENIZER_NAME}`; no exact tokenizer available offline). "
             "For adaptive, `injection_token_limit = budget` and "
             "`memory_store_token_budget = max(4096, budget*4)`, mirroring the repo "
             "convention; this is documented, not tuned.")
    L.append("")

    L.append("## 8. Coding Model, Prompt & Harness Determinism")
    L.append("")
    L.append(f"Coding model: `{cfg.get('model')}` via Ollama "
             f"(`temperature={cb.CODING_TEMPERATURE}`, `num_ctx={cb.MODEL_CONTEXT_TOKENS}`, "
             f"`keep_alive=30m`, `num_predict={cfg.get('max_output_tokens')}`). "
             "Prompt = instructions + current workspace files (visible files only) + "
             "current task + historical engineering context. The model returns "
             "complete-file blocks that the harness writes verbatim (the resulting "
             "change is captured as a git diff); a diff fallback exists. Workspace "
             "reset, edit application and hidden tests are deterministic (gates 8-9).")
    L.append("")

    L.append("## 9. Evaluation Metrics & Failure Taxonomy")
    L.append("")
    L.append("Primary: `final_success` (hidden test passes). Secondary: "
             "`first_pass_success`, attempts. Diagnostics: `critical_fact_recall`, "
             "`correction_recall`, `negative_constraint_recall`, "
             "`long_range_fact_recall`, `obsolete_fact_exposure`. Failure classes "
             "(deterministic precedence): MEMORY_MISS, RETRIEVAL_MISS, "
             "CONTEXT_OVERFLOW, PATCH_INVALID, OBSOLETE_INFORMATION_USED, "
             "CORRECTION_MISSED, CODING_ERROR, HIDDEN_TEST_FAILURE, OTHER.")
    L.append("")

    L.append("## 10. Configuration & Reproducibility")
    L.append("")
    for k, v in cfg.items():
        L.append(f"- **{k}**: {v}")
    L.append("")

    L.append("## 11. Results: Success Rate by Method × Budget")
    L.append("")
    budgets = cfg.get("budgets") or []
    header = "| method | " + " | ".join(str(b) for b in budgets) + " | overall |"
    L.append(header)
    L.append("| --- | " + " | ".join("---" for _ in budgets) + " | --- |")
    for m in cfg.get("methods", []) + ["no_history", "full_context", "direct_history"]:
        row = []
        for b in budgets:
            v = (agg.get("by_method_budget", {}).get(m, {}).get(str(b), {}) or {}).get(
                "success_rate")
            row.append(_fmt(v))
        overall = (agg.get("overall", {}).get(m, {}) or {}).get("success_rate")
        L.append(f"| {m} | " + " | ".join(row) + f" | {_fmt(overall)} |")
    L.append("")

    L.append("## 12. Results: First-Pass vs Final Success")
    L.append("")
    L.append("| method | budget | first-pass | final | repair recovery | runs |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for m in cfg.get("methods", []):
        for b in budgets:
            row = agg.get("by_method_budget", {}).get(m, {}).get(str(b), {})
            if row:
                L.append(f"| {m} | {b} | {_fmt(row.get('first_pass_rate'))} | "
                         f"{_fmt(row.get('success_rate'))} | "
                         f"{_fmt(row.get('repair_recovery_rate'))} | {row.get('runs')} |")
    L.append("")

    L.append("## 13. Paired Comparisons & Effect Sizes (adaptive − other)")
    L.append("")
    L.append("| vs | pairs | adaptive | other | mean diff | diff 95% CI | adaptive 95% CI |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for m, c in (agg.get("paired_vs_adaptive") or {}).items():
        L.append(f"| {m} | {c['pairs']} | {_fmt(c['adaptive_success'])} | "
                 f"{_fmt(c['other_success'])} | {_fmt(c['mean_diff'])} | "
                 f"{c['mean_diff_ci95']} | {c['adaptive_ci95']} |")
    L.append("")

    L.append("## 14. Historical-Context Usage & Budget Pressure")
    L.append("")
    L.append("| method | budget | mean historical tokens | mean total prompt tokens |")
    L.append("| --- | --- | --- | --- |")
    for m in cfg.get("methods", []):
        for b in budgets:
            row = agg.get("by_method_budget", {}).get(m, {}).get(str(b), {})
            if row:
                L.append(f"| {m} | {b} | {_fmt(row.get('mean_context_tokens'))} | "
                         f"{_fmt(row.get('mean_total_prompt_tokens'))} |")
    L.append("")

    L.append("## 15. Diagnostic: Critical-Fact Recall")
    L.append("")
    L.extend(_diag_table(agg, budgets, "critical_fact_recall"))
    L.append("")

    L.append("## 16. Diagnostic: Correction & Negative-Constraint Recall")
    L.append("")
    L.append("### correction_recall")
    L.append("")
    L.extend(_diag_table(agg, budgets, "correction_recall"))
    L.append("")
    L.append("### negative_constraint_recall")
    L.append("")
    L.extend(_diag_table(agg, budgets, "negative_constraint_recall"))
    L.append("")

    L.append("## 17. Diagnostic: Long-Range Recall & Obsolete Exposure")
    L.append("")
    L.append("### long_range_fact_recall")
    L.append("")
    L.extend(_diag_table(agg, budgets, "long_range_fact_recall"))
    L.append("")
    L.append("### obsolete_fact_exposure")
    L.append("")
    L.extend(_diag_table(agg, budgets, "obsolete_fact_exposure"))
    L.append("")

    L.append("## 18. Failure Taxonomy Distribution")
    L.append("")
    L.append("| method | budget | failure classes |")
    L.append("| --- | --- | --- |")
    for m in cfg.get("methods", []):
        for b in budgets:
            row = agg.get("by_method_budget", {}).get(m, {}).get(str(b), {})
            if row:
                L.append(f"| {m} | {b} | {row.get('failure_classes')} |")
    L.append("")

    L.append("## 19. Context-Overflow / Patch-Valid / Failure Modes")
    L.append("")
    overflow = sum(1 for r in result["records"] if r["failure_class"] == "CONTEXT_OVERFLOW")
    patch_bad = sum(1 for r in result["records"] if not r["patch_valid"])
    L.append(f"- Runs with `CONTEXT_OVERFLOW`: **{overflow}**.")
    L.append(f"- Runs whose model output contained no valid patch: **{patch_bad}**.")
    L.append("- Full per-run records (including the produced patch) are in the JSON.")
    L.append("")

    L.append("## 20. No-History & Full/Direct-Context Diagnostics")
    L.append("")
    L.append("| task | no_history | full_context | direct_history |")
    L.append("| --- | --- | --- | --- |")
    diags = agg.get("diagnostics", {})
    for tid in cfg.get("tasks", []):
        L.append(f"| {tid} | "
                 f"{_fmt((diags.get(_diag_key(tid, 'no_history')) or {}).get('final_success'))} | "
                 f"{_fmt((diags.get(_diag_key(tid, 'full_context')) or {}).get('final_success'))} | "
                 f"{_fmt((diags.get(_diag_key(tid, 'direct_history')) or {}).get('final_success'))} |")
    L.append("")
    gold = result.get("gold_checks") or []
    L.append(f"Gold-patch harness sanity: "
             f"{sum(1 for g in gold if g['final_success'])}/{len(gold)} tasks pass when "
             "the gold patch is supplied.")
    L.append("")

    L.append("## 21. Pilot Gates (9/9 required)")
    L.append("")
    L.append("| gate | passed | detail |")
    L.append("| --- | --- | --- |")
    for name in GATE_NAMES:
        g = gates.get(name, {})
        detail = {k: v for k, v in g.items() if k != "passed"}
        L.append(f"| {name} | {'PASS' if g.get('passed') else 'FAIL'} | "
                 f"{json.dumps(detail, default=str)[:400]} |")
    L.append("")
    L.append(f"**All gates passed: {gates.get('all_passed')}.**")
    L.append("")

    L.append("## 22. Threats to Validity & Limitations")
    L.append("")
    L.append("- Word-count tokenizer approximates the model's real tokenization; "
             "budgets are therefore approximate (documented in the manifest).")
    L.append("- Small task count and seeds imply very wide success-rate CIs; absence "
             "of significance is not evidence of equivalence.")
    L.append("- Task suite is synthetic-but-real-code; findings may not transfer to "
             "large repositories.")
    L.append("- `full_context` is a context-unconstrained diagnostic, not a "
             "fixed-budget competitor.")
    L.append("- Ollama-served `llama3.1:8b` at temperature 0.1 is not deterministic "
             "across runs; per-run success is the unit and the grid is resumable.")
    L.append("")

    L.append("## 23. Conclusion & Evidence-Based Verdict")
    L.append("")
    L.append(f"- Mode: **{cfg.get('mode')}**.")
    L.append(f"- Gates all passed: **{gates.get('all_passed')}**.")
    L.append(f"- adaptive_advances: **{verdict.get('adaptive_advances')}**.")
    L.append("")
    L.append(f"rationale: {verdict.get('rationale', '')}")
    L.append("")
    if cfg.get("mode") == "pilot":
        L.append("This is a PILOT. No production decision is made from it.")
    else:
        L.append("A positive verdict requires every paired adaptive-vs-baseline "
                 "95% CI to exclude 0 in adaptive's favour; otherwise the honest "
                 "conclusion is that adaptive did not demonstrate an advantage on "
                 "this benchmark.")
    L.append("")

    L.append("## 24. Reproduction / Artifacts / Provenance")
    L.append("")
    L.append(f"- JSON: `{OUT_JSON.name}` (records include per-run patches).")
    L.append(f"- generated_at: {cfg.get('generated_at')}")
    L.append(f"- tokenizer: `{TOKENIZER_NAME}`; model: `{cfg.get('model')}`; "
             f"embedding: `{cfg.get('embedding_model')}`")
    L.append(f"- work dir: `experiments/results/e17_work/` (gitignored)")
    L.append("")
    L.append("— All numbers are generated from the E17 JSON by `generate_report()`; "
             "no hand-typed figures.")
    return "\n".join(L)


def _diag_table(agg: Dict, budgets: List[int], metric: str) -> List[str]:
    lines = ["| method | " + " | ".join(str(b) for b in budgets) + " |",
             "| --- | " + " | ".join("---" for _ in budgets) + " |"]
    for m in sorted(agg.get("by_method_budget", {})):
        row = []
        for b in budgets:
            d = (agg.get("by_method_budget", {}).get(m, {}).get(str(b), {}) or {}).get(
                "diagnostics", {}).get(metric)
            row.append(_fmt(d))
        lines.append(f"| {m} | " + " | ".join(row) + " |")
    return lines


# ---------------------------------------------------------------------------
# Model availability
# ---------------------------------------------------------------------------

def check_models(model: str, embedding_model: str, endpoint: str,
                 use_embeddings: bool) -> None:
    import requests
    try:
        r = requests.get(f"{endpoint}/api/tags", timeout=10)
        r.raise_for_status()
        names = {m.get("name") for m in r.json().get("models", [])}
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(f"Ollama not reachable at {endpoint}: {exc}")
    missing = [m for m in [model] if not any(n == m or n.startswith(m + ":")
                                             for n in names)]
    if use_embeddings:
        missing += [embedding_model for m in [embedding_model]
                    if not any(n == m or n.startswith(m + ":") for n in names)]
    if missing:
        raise SystemExit(f"required Ollama model(s) missing: {missing}; "
                         f"available: {sorted(names)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="E17 coding capability")
    parser.add_argument("--full", action="store_true",
                        help="run the full grid (4 tasks x 3 seeds x 3 budgets x 5 methods)")
    parser.add_argument("--quick", action="store_true",
                        help="pilot grid (3 tasks x 2 seeds x 2 budgets x 5 methods)")
    parser.add_argument("--report-only", action="store_true",
                        help="regenerate the report/verdict from the existing JSON")
    parser.add_argument("--force", action="store_true", help="recompute all runs")
    parser.add_argument("--no-embed", action="store_true",
                        help="disable embeddings (lexical retrieval; diagnostic only)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="limit number of runs (debugging)")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    use_embeddings = not args.no_embed

    if args.report_only:
        if not OUT_JSON.exists():
            parser.error("no e17_coding_capability.json to classify")
        result = json.loads(OUT_JSON.read_text())
        result["gates"] = compute_gates(result["records"],
                                        result.get("diagnostics", []),
                                        result["config"]["tasks"])
        result["verdict"] = classify_verdict(result)
        result["aggregate"] = aggregate(result["records"], result.get("diagnostics", []))
        result.pop("partial", None)
        OUT_JSON.write_text(json.dumps(result, indent=2, default=str))
        OUT_REPORT.write_text(generate_report(result))
        print(f"Re-classified; adaptive_advances="
              f"{result['verdict'].get('adaptive_advances')}")
        return result

    if not args.skip_model_check:
        check_models(args.model, args.embedding_model, args.endpoint, use_embeddings)

    if args.full:
        mode = "full"
        tasks, seeds, budgets, methods = FULL_TASKS, FULL_SEEDS, FULL_BUDGETS, PILOT_METHODS
    else:
        mode = "pilot"
        tasks, seeds, budgets, methods = PILOT_TASKS, PILOT_SEEDS, PILOT_BUDGETS, PILOT_METHODS

    print("=" * 64)
    print(f"E17 Coding Capability ({mode})")
    print("=" * 64)

    existing = {}
    diag_existing: Dict[str, Dict] = {}
    gold_existing: Dict[str, Dict] = {}
    if OUT_JSON.exists() and not args.force:
        prev = json.loads(OUT_JSON.read_text())
        for r in prev.get("records", []):
            existing[_run_key(r["task_id"], r["seed"], r["historical_budget"], r["method"])] = r
        for r in prev.get("diagnostics", []):
            if r["method"] in ("no_history", "full_context", "direct_history"):
                diag_existing[_diag_key(r["task_id"], r["method"])] = r
        for g in prev.get("gold_checks", []):
            gold_existing[g["task_id"]] = g

    embed_fn = _resolve_embed(use_embeddings, args.embedding_model, args.endpoint)
    coder = cb.OllamaCoder(args.model, args.endpoint, args.max_output_tokens)
    summarizer_generate_fn = lambda prompt, mx: coder.generate(prompt)["text"]

    records = list(existing.values())
    diagnostics = list(diag_existing.values())
    gold_checks = list(gold_existing.values())

    # gold sanity (harness can pass every task)
    for tid in tasks:
        if tid in gold_existing:
            continue
        g = check_gold(tid)
        gold_checks.append(g)
        print(f"  [gold] {tid}: success={g['final_success']}", flush=True)

    def persist(partial=True):
        payload = {
            "experiment": "e17_coding_capability",
            "partial": partial,
            "config": {
                "mode": mode, "tasks": tasks, "seeds": seeds, "budgets": budgets,
                "methods": methods, "model": args.model, "embedding_model":
                args.embedding_model, "use_embeddings": use_embeddings,
                "max_output_tokens": args.max_output_tokens,
                "temperature": cb.CODING_TEMPERATURE,
                "tokenizer": TOKENIZER_NAME,
                "ingestion": "oracle_pre_extracted",
                "adaptive": {"retention_mode": "dual_score",
                             "protect_corrections": True,
                             "injection_token_limit": "historical_budget",
                             "memory_store_token_budget": "max(4096, budget*4)"},
                "generated_at": time.time(),
            },
            "records": records,
            "diagnostics": diagnostics,
            "gold_checks": gold_checks,
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))

    total = len(tasks) * len(seeds) * len(budgets) * len(methods)
    done = 0
    limit = args.limit or total
    for tid in tasks:
        for seed in seeds:
            for budget in budgets:
                for method in methods:
                    key = _run_key(tid, seed, budget, method)
                    if key in existing:
                        continue
                    if done >= limit:
                        break
                    done += 1
                    print(f"  [{done}] {key} ...", flush=True)
                    try:
                        rec = run_one(tid, seed, budget, method, coder=coder,
                                      embed_fn=embed_fn, model=args.model,
                                      endpoint=args.endpoint,
                                      embedding_model=args.embedding_model,
                                      summarizer_generate_fn=summarizer_generate_fn)
                    except Exception as exc:  # keep the grid resumable
                        print(f"      ERROR {key}: {exc}", flush=True)
                        continue
                    records.append(rec)
                    existing[key] = rec
                    persist()

    # no-history / full-context / direct-history diagnostics
    for tid in tasks:
        for method in ("no_history", "full_context", "direct_history"):
            key = _diag_key(tid, method)
            if key in diag_existing:
                continue
            print(f"  [diag] {tid}:{method} ...", flush=True)
            try:
                rec = run_one(tid, DIAG_SEED, DIAG_BUDGET, method, coder=coder,
                              embed_fn=embed_fn, model=args.model,
                              endpoint=args.endpoint,
                              embedding_model=args.embedding_model,
                              summarizer_generate_fn=summarizer_generate_fn)
            except Exception as exc:
                print(f"      ERROR {key}: {exc}", flush=True)
                continue
            diagnostics.append(rec)
            diag_existing[key] = rec
            persist()

    result = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    result = {
        "experiment": "e17_coding_capability",
        "config": {
            "mode": mode, "tasks": tasks, "seeds": seeds, "budgets": budgets,
            "methods": methods, "model": args.model,
            "embedding_model": args.embedding_model, "use_embeddings": use_embeddings,
            "max_output_tokens": args.max_output_tokens,
            "temperature": cb.CODING_TEMPERATURE, "tokenizer": TOKENIZER_NAME,
            "ingestion": "oracle_pre_extracted",
            "adaptive": {"retention_mode": "dual_score",
                         "protect_corrections": True,
                         "injection_token_limit": "historical_budget",
                         "memory_store_token_budget": "max(4096, budget*4)"},
            "generated_at": time.time(),
        },
        "records": records,
        "diagnostics": diagnostics,
        "gold_checks": gold_checks,
    }
    result["aggregate"] = aggregate(records, diagnostics)
    result["gates"] = compute_gates(records, diagnostics, tasks)
    result["verdict"] = classify_verdict(result)
    result.pop("partial", None)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {OUT_JSON}")
    OUT_REPORT.write_text(generate_report(result))
    print(f"Wrote {OUT_REPORT}")
    print(f"gates_all_passed={result['gates'].get('all_passed')} "
          f"adaptive_advances={result['verdict'].get('adaptive_advances')}")
    return result


if __name__ == "__main__":
    main()
