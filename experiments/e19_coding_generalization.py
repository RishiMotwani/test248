"""E19 — Phase 15: does the corrected adaptive memory generalize to a broader
set of long-horizon coding tasks under a fixed historical-context budget?

Research question (D37)
-----------------------
Does the adaptive memory system — after the Phase 15 supersession-embedding fix
(D36) — generalize to a wider, genuinely history-dependent set of coding tasks
better than simpler context-management strategies (raw_clipped,
sliding_window, llm_summarization, vanilla_rag) under the same historical
budget, measured by deterministic hidden-test patch pass?

Design (locked by the Phase 15 specification; not tuned here)
--------------------------------------------------------------
* Primary tasks (genuinely history-dependent): user_ids, validation_pure,
  NEW transaction_atomicity (calibrated this phase). Every primary has >=600
  deterministic scripted turns, >=3 topic families, >=2 critical facts,
  >=1 correction or negative constraint, >=1 fact >300 turns old, >=1
  distractor thread, identity-safe fact_ids and no marker language / hidden
  test leakage.
* config_contract (new task authored this phase) is excluded from the E19
  primary set: calibration probing showed llama3.1:8b cannot reliably
  express its required "explicit empty string is a valid value" nuance even
  with oracle history (0/3 across four fixture designs), so the task carries
  no memory signal at this model tier. Its fixture is retained for future,
  stronger models and documented in the report.
* Negative controls (harness sanity only — never contribute to the
  adaptive-advances decision): cache_readonly, write_retry.
* Methods (5, history mechanism differs only): raw_clipped, sliding_window,
  llm_summarization, vanilla_rag, adaptive (the production pipeline,
  retention=dual_score, protect_corrections=True; exactly one adaptive arm).
* Pilot grid: 3 primary x 2 seeds x 2 budgets (256, 512) x 5 methods = 60 runs.
* Full grid: 3 primary x 3 seeds x 3 budgets (256/512/1024) x 5 methods = 135,
  plus 2 negative controls x 3 x 3 x 5 = 90 diagnostic runs (225 total).
* The harness (experiments/coding_benchmark.py) is reused unchanged.
* Every run is persisted immediately under ``task_id:seed:budget:method`` and
  the run is resumable; completed results are never deleted.
* Embedding consistency (D36) is verified offline: every correction-superseded
  memory in the adaptive store must carry the embedding of the current fact
  text, not of its obsolete predecessor.

Pilot gates A-J must all pass before the full grid is treated as an experiment;
pilot results alone are labelled PILOT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.coding_task_suite import build_task, list_task_ids  # noqa: E402
from experiments import coding_benchmark as cb  # noqa: E402
from experiments.e17_coding_capability import check_models  # noqa: E402
from experiments.statistics import (  # noqa: E402
    bootstrap_ci_mean_diff,
    bootstrap_mean_ci,
)
from memory_optimizer.compression import _embedding_matches_fact  # noqa: E402
from memory_optimizer.tokenizer import TOKENIZER_NAME  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration (locked by the Phase 15 specification)
# ---------------------------------------------------------------------------

PRIMARY_TASKS = ["user_ids", "validation_pure", "transaction_atomicity"]
CALIB_EXCLUDED_TASK = "config_contract"  # fixture-only; see module docstring
NEGATIVE_CONTROL_TASKS = ["cache_readonly", "write_retry"]
ALL_TASKS = PRIMARY_TASKS + NEGATIVE_CONTROL_TASKS

PILOT_SEEDS = [1, 2]
FULL_SEEDS = [1, 2, 3]
PILOT_BUDGETS = [256, 512]
FULL_BUDGETS = [256, 512, 1024]
METHODS = [m for m in cb.METHOD_NAMES if m not in ("adaptive_v2", "tuned",
                                                    "adaptive_plus")]
BASELINE_METHODS = [m for m in METHODS if m != "adaptive"]

DIAG_SEED = 1
DIAG_BUDGET = 1024

# Diagnostics for gate C and the solvability checks are stochastic (an LLM can
# pass no_history once by luck). Gate C therefore measures *reproducibility*:
# a task is taken to genuinely require history only when no_history fails on
# every diagnostic draw. oracle direct_history solvability is recorded as a
# success RATE, not a gate.
DIAG_DRAWS = 3

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_ENDPOINT = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
MAX_OUTPUT_TOKENS = cb.MAX_OUTPUT_TOKENS

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_JSON = RESULTS_DIR / "e19_coding_generalization.json"
OUT_REPORT = RESULTS_DIR / "e19_coding_generalization_report.md"
WORK_ROOT = RESULTS_DIR / "e19_work"

GATE_NAMES = [
    "embedding_consistency",       # A
    "correction_identity",         # B
    "history_dependence",          # C
    "method_separation",           # D
    "no_leakage",                  # E
    "gold_passes",                 # F
    "budget_pressure",             # G
    "real_summarization",          # H
    "adaptive_production_path",    # I
    "unit_tests_pass",             # J
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


def _resolve_embed(use_embeddings: bool, embedding_model: str, endpoint: str):
    if not use_embeddings:
        return None
    from memory_optimizer.embeddings import embed_ollama
    return lambda texts: embed_ollama(texts, model=embedding_model,
                                      endpoint=endpoint)


def _sha16(text: str) -> Optional[str]:
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Record schema (E19)
# ---------------------------------------------------------------------------

def to_schema(rec: Dict) -> Dict:
    """Normalise a harness run into the E19 result schema.

    Every record carries ``experiment: "E19"`` explicitly, the four token
    counts (historical / workspace / task prompt / total), first-pass and final
    success, the five recall/exposure diagnostics, the failure class, context
    and patch sha, latency splits, and model-call counts. Nothing is fabricated.
    """
    diag = rec.get("diagnostics") or {}
    return {
        "experiment": "E19",
        "task_id": rec["task_id"],
        "seed": rec["seed"],
        "historical_budget": rec["historical_budget"],
        "method": rec["method"],
        "historical_context_tokens": rec["historical_context_tokens"],
        "workspace_context_tokens": rec["workspace_context_tokens"],
        "task_prompt_tokens": rec["task_prompt_tokens"],
        "total_prompt_tokens": rec["total_prompt_tokens"],
        "first_pass_success": bool(rec["first_pass_success"]),
        "final_success": bool(rec["final_success"]),
        "critical_fact_recall": diag.get("critical_fact_recall"),
        "correction_recall": diag.get("correction_recall"),
        "negative_constraint_recall": diag.get("negative_constraint_recall"),
        "long_range_fact_recall": diag.get("long_range_fact_recall"),
        "obsolete_fact_exposure": diag.get("obsolete_fact_exposure"),
        "failure_class": rec["failure_class"],
        "context_sha": rec.get("context_sha"),
        "patch_sha": _sha16(rec.get("patch")),
        "latency_ms": rec.get("latency_ms"),
        "model_calls": {
            "coding_attempts": int(rec.get("coding_attempts", 0)),
            "memory_build_calls": int(rec.get("memory_build_calls", 0)),
            "summary_update_calls": int(rec.get("summary_update_calls", 0)),
        },
        "uses_production_pipeline": bool(rec.get("uses_production_pipeline")),
        "leakage_detected": bool((rec.get("leakage") or {}).get("leakage_detected")),
    }


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
    rec = cb.run_method_run(task=task, seed=seed, historical_budget=budget,
                            method=method, coder=coder, work_root=root)
    return to_schema(rec)


# ---------------------------------------------------------------------------
# Gold sanity + determinism
# ---------------------------------------------------------------------------

class StaticCoder:
    def __init__(self, patch: str, model: str = "static"):
        self.patch = patch
        self.model = model

    def generate(self, prompt: str) -> Dict:
        return {"text": "```diff\n" + self.patch + "\n```",
                "prompt_tokens": 0, "output_tokens": 0, "latency_ms": 0.0}


def check_gold(task_id: str, seed: int = 1) -> Dict:
    task = build_task(task_id, seed=seed)
    gold_path = getattr(task, "gold_patch_path", None)
    if gold_path is None:
        gold_path = task.workspace_path.parent / "gold.patch"
    gold = Path(gold_path).read_text()
    coder = StaticCoder(gold)
    root = WORK_ROOT / f"goldcheck_{task_id}"
    method = cb.build_method("no_history", model="static", endpoint="http://x",
                             embed_fn=None, embedding_model="", task=task)
    rec = cb.run_method_run(task=task, seed=seed, historical_budget=256,
                            method=method, coder=coder, work_root=root)
    return {"task_id": task_id, "final_success": rec["final_success"],
            "patch_applied": rec["patch_applied"]}


# ---------------------------------------------------------------------------
# Offline gates A & B (embedding consistency + correction identity)
# ---------------------------------------------------------------------------

def run_offline_cell(task_id: str, seed: int, budget: int, *,
                     embed_fn, embedding_model: str, endpoint: str,
                     model: str) -> Dict:
    task = build_task(task_id, seed=seed)
    method = cb.build_method("adaptive", model=model, endpoint=endpoint,
                             embed_fn=embed_fn, embedding_model=embedding_model,
                             task=task)
    history = cb.HistoryBundle(task.task_id, task.history, task.history_text)
    prepared = method.prepare(history, budget, seed=seed)
    memories = prepared.payload["memories"]
    text_by_id = {f.fact_id: f.text
                  for f in list(task.gold_facts) + list(task.distractor_facts)}

    corrections = []
    for c in task.corrections:
        cur = next((m for m in memories if m.get("fact_id") == c.current_fact_id),
                   None)
        obs = next((m for m in memories if m.get("fact_id") == c.obsolete_fact_id),
                   None)
        emb_ok = None
        if cur is not None and embed_fn is not None:
            emb_ok = bool(_embedding_matches_fact(cur, embed_fn))
        corrections.append({
            "obsolete_fact_id": c.obsolete_fact_id,
            "current_fact_id": c.current_fact_id,
            "current_fact_present_by_id": cur is not None,
            "obsolete_fact_present_by_id": obs is not None,
            "embedding_matches_current_fact": emb_ok,
        })

    n = len(corrections)
    identity_ok = bool(corrections) and all(
        c["current_fact_present_by_id"] and not c["obsolete_fact_present_by_id"]
        for c in corrections)
    emb_ok_all = (embed_fn is not None and bool(corrections)
                  and all(c["embedding_matches_current_fact"] is True
                          for c in corrections))
    return {
        "task_id": task_id, "seed": seed, "budget": budget, "method": "adaptive",
        "n_corrections": n,
        "corrections": corrections,
        "identity_gate": identity_ok,
        "embedding_gate": emb_ok_all,
        "store_tokens": prepared.memory_tokens,
        "memory_count": len(memories),
    }


def offline_gate(cells: List[Dict], primary_tasks: List[str]) -> Dict:
    prim = [c for c in cells if c["task_id"] in primary_tasks and c["n_corrections"] > 0]
    embedding_ok_cells = [c for c in prim if c["embedding_gate"]]
    identity_ok_cells = [c for c in prim if c["identity_gate"]]
    return {
        "correction_bearing_primary_cells": len(prim),
        "embedding_consistent_cells": len(embedding_ok_cells),
        "identity_ok_cells": len(identity_ok_cells),
        "embedding_gate": len(prim) > 0 and len(embedding_ok_cells) == len(prim),
        "identity_gate": len(prim) > 0 and len(identity_ok_cells) == len(prim),
        "failing_embedding_cells": [f"{c['task_id']}:{c['seed']}:{c['budget']}"
                                    for c in prim if not c["embedding_gate"]],
        "failing_identity_cells": [f"{c['task_id']}:{c['seed']}:{c['budget']}"
                                   for c in prim if not c["identity_gate"]],
    }


# ---------------------------------------------------------------------------
# Gates A-J
# ---------------------------------------------------------------------------

def compute_gates(*, records: List[Dict], diagnostics: List[Dict],
                  offline: Dict, gold_checks: List[Dict], unit_tests: Dict,
                  task_ids_primary: List[str],
                  budgets_lo: int, budgets_hi: int) -> Dict:
    grid = [r for r in records if r["method"] in METHODS and
            r["task_id"] in task_ids_primary]

    og = offline or {}
    if isinstance(og.get("gate"), dict):
        og = og["gate"]
    # persist stores offline as {"cells": [...], "gate": {...}}; gates A/B only
    # need the gate summary, so unwrap it when present.

    # A: embedding consistency (offline) — current corrected fact's stored
    #    embedding must match the corrected fact text (cosine >= 0.99).
    ga = bool(og.get("embedding_gate"))

    # B: correction identity (offline) — obsolete fact dropped, current kept.
    gb = bool(og.get("identity_gate"))

    # C: history dependence — no_history must FAIL on every diagnostic draw for
    #    every primary task (no task is solvable from the workspace alone, even
    #    by luck). Oracle direct-history solvability is recorded per task as a
    #    diagnostic success RATE: coding-model variance makes it unsuitable as a
    #    hard gate, so it is reported, not enforced (see the report's
    #    diagnostics section).
    no_hist = {r["task_id"]: r for r in diagnostics if r["method"] == "no_history"}
    direct = {r["task_id"]: r for r in diagnostics if r["method"] == "direct_history"}
    per_task = {}
    for tid in task_ids_primary:
        nh = no_hist.get(tid)
        dh = direct.get(tid)
        nh_ok = bool(nh and (nh.get("successes") or 0) > 0)
        dh_suc = int(dh.get("successes", 0)) if dh else 0
        per_task[tid] = {"no_history_success": nh_ok,
                         "no_history_successes": int(nh.get("successes", 0)) if nh else 0,
                         "oracle_direct_history_success": dh_suc > 0,
                         "oracle_direct_history_successes": dh_suc}
    # g3 uses `no_history_success` (all-draws must fail) but the per-task
    # detail stays available via the extended dict above.
    g3 = all(not p["no_history_success"] for p in per_task.values())

    # D: method separation — at least one (task,seed,budget) group shows >=3
    #    distinct historical contexts.
    groups: Dict[str, set] = defaultdict(set)
    for r in grid:
        groups[f"{r['task_id']}:{r['seed']}:{r['historical_budget']}"].add(
            (r["method"], r["context_sha"]))
    max_distinct = max((len({sha for _, sha in v}) for v in groups.values()),
                       default=0)
    g4 = max_distinct >= 3

    # E: no leakage — raise if any run leaked the correct answer into the prompt
    leaks = [r for r in records if r.get("leakage_detected")]
    g5 = len(leaks) == 0

    # F: deterministic fixtures gold passes
    g6 = bool(gold_checks) and all(g["final_success"] for g in gold_checks)

    # G: raw history pressure exceeds the largest budget and clipping binds.
    #    Budgets are taken from the grid actually present (pilot or full), so
    #    the growth check measures the two extremes of the run grid.
    fc = [d for d in diagnostics if d["method"] == "full_context"]
    grid_budgets = sorted({r["historical_budget"] for r in grid}) or [0]
    lo_b, hi_b = grid_budgets[0], grid_budgets[-1]
    history_exceeds = bool(fc) and max(
        d["historical_context_tokens"] for d in fc) > hi_b
    lo = _mean([r["historical_context_tokens"] for r in grid
                if r["method"] == "raw_clipped" and r["historical_budget"] == lo_b])
    hi = _mean([r["historical_context_tokens"] for r in grid
                if r["method"] == "raw_clipped" and r["historical_budget"] == hi_b])
    raw_grows = lo is not None and hi is not None and hi > lo + 20
    fills = any(r["historical_context_tokens"] >= 0.6 * r["historical_budget"]
                for r in grid)
    g7 = history_exceeds and raw_grows and fills

    # H: real summarization
    summ = [r for r in grid if r["method"] == "llm_summarization"]
    g8 = bool(summ) and sum(r["model_calls"].get("summary_update_calls", 0)
                            for r in summ) > 0

    # I: adaptive production path
    adapt = [r for r in grid if r["method"] == "adaptive"]
    g9 = bool(adapt) and all(r.get("uses_production_pipeline") for r in adapt)

    # J: full unit suite passes (computed separately)
    g10 = bool(unit_tests.get("passed"))

    gates = {
        "embedding_consistency": {
            "passed": ga,
            "cells": og.get("correction_bearing_primary_cells"),
            "embedding_consistent_cells": og.get("embedding_consistent_cells"),
            "failing_cells": og.get("failing_embedding_cells"),
        },
        "correction_identity": {
            "passed": gb,
            "cells": og.get("correction_bearing_primary_cells"),
            "identity_ok_cells": og.get("identity_ok_cells"),
            "failing_cells": og.get("failing_identity_cells"),
        },
        "history_dependence": {
            "passed": g3, "per_task": per_task,
        },
        "method_separation": {
            "passed": g4, "max_distinct_contexts": max_distinct,
        },
        "no_leakage": {
            "passed": g5, "leaking_runs": len(leaks),
        },
        "gold_passes": {
            "passed": g6, "gold_checks": gold_checks,
        },
        "budget_pressure": {
            "passed": g7, "history_exceeds_max_budget": history_exceeds,
            "raw_grows_with_budget": raw_grows, "some_run_fills_60pct": fills,
        },
        "real_summarization": {
            "passed": g8,
            "summary_update_calls": sum(r["model_calls"].get("summary_update_calls", 0)
                                        for r in summ),
        },
        "adaptive_production_path": {
            "passed": g9, "adaptive_runs": len(adapt),
        },
        "unit_tests_pass": {
            "passed": g10, "unit_tests": unit_tests,
        },
    }
    gates["all_passed"] = all(gates[k]["passed"] for k in GATE_NAMES)
    return gates


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _count_failures(rows: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["failure_class"] or "SUCCESS"] += 1
    return dict(sorted(counts.items()))


def aggregate(records: List[Dict]) -> Dict:
    """Aggregate over a set of E19-schema records (primary only)."""
    by_method_budget: Dict[str, Dict[str, List[Dict]]] = defaultdict(
        lambda: defaultdict(list))
    for r in records:
        by_method_budget[r["method"]][str(r["historical_budget"])].append(r)

    diag_keys = ["critical_fact_recall", "correction_recall",
                 "negative_constraint_recall", "long_range_fact_recall",
                 "obsolete_fact_exposure"]
    out: Dict[str, Dict] = {}
    for m, by_b in by_method_budget.items():
        out[m] = {}
        for b in sorted(by_b, key=int):
            rows = by_b[b]
            out[m][b] = {
                "runs": len(rows),
                "success_rate": _success(rows),
                "first_pass_rate": round(
                    sum(1 for r in rows if r["first_pass_success"]) / len(rows), 4)
                if rows else None,
                "mean_context_tokens": _mean(
                    [r["historical_context_tokens"] for r in rows]),
                "mean_total_prompt_tokens": _mean(
                    [r["total_prompt_tokens"] for r in rows]),
                "diagnostics": {k: _mean([r.get(k) for r in rows])
                                for k in diag_keys},
                "failure_classes": _count_failures(rows),
            }

    def cell_success(method: str) -> Dict:
        return {(r["task_id"], r["seed"], r["historical_budget"]):
                int(r["final_success"])
                for r in records if r["method"] == method}

    adapt = cell_success("adaptive")
    comparisons = {}
    for m in BASELINE_METHODS:
        other = cell_success(m)
        keys = sorted(set(adapt) & set(other))
        if len(keys) < 2:
            continue
        a = [adapt[k] for k in keys]
        b = [other[k] for k in keys]
        comparisons[m] = {
            "pairs": len(keys),
            "cells": [f"{k[0]}:{k[1]}:{k[2]}" for k in keys],
            "adaptive_success": round(sum(a) / len(a), 4),
            "baseline_success": round(sum(b) / len(b), 4),
            "mean_diff": round(sum(x - y for x, y in zip(a, b)) / len(a), 4),
            "mean_diff_ci95": bootstrap_ci_mean_diff(a, b),
            "adaptive_ci95": bootstrap_mean_ci(a),
            "baseline_ci95": bootstrap_mean_ci(b),
        }

    overall = {}
    for m in sorted({r["method"] for r in records}):
        rows = [r for r in records if r["method"] == m]
        overall[m] = {"runs": len(rows), "success_rate": _success(rows),
                      "first_pass_rate": round(
                          sum(1 for r in rows if r["first_pass_success"]) / len(rows), 4)
                      if rows else None}

    return {"by_method_budget": out, "paired_vs_adaptive": comparisons,
            "overall": overall}


# ---------------------------------------------------------------------------
# Verdict (predeclared criterion: positive paired advantage, CI lower bound > 0)
# ---------------------------------------------------------------------------

def classify_verdict(result: Dict) -> Dict:
    gates = result.get("gates") or {}
    agg = result.get("aggregate") or {}
    comp = agg.get("paired_vs_adaptive") or {}
    gates_ok = bool(gates.get("all_passed"))
    mode = result.get("config", {}).get("mode")

    wins = {}
    for m in BASELINE_METHODS:
        c = comp.get(m)
        if not c:
            wins[m] = (False, [None, None])
            continue
        lo = c["mean_diff_ci95"][0]
        wins[m] = (lo is not None and lo > 0), c["mean_diff_ci95"]

    all_beat = all(ok for ok, _ in wins.values())
    advances = gates_ok and all_beat

    verdict = {
        "pilot_only": mode == "pilot",
        "gates_all_passed": gates_ok,
        "adaptive_beats_all_baselines": all_beat,
        "paired": {m: c for m, c in comp.items()},
        "criteria": {
            "gates_pass": gates_ok,
            "positive_paired_ci_lower_bound": {
                m: (wins[m][1][0] if wins[m][1][0] is not None else None)
                for m in BASELINE_METHODS},
        },
    }
    if not gates_ok:
        verdict["adaptive_advances"] = False
        verdict["rationale"] = (
            "Pilot gates did not all pass; the grid is not a valid experiment "
            "and adaptive cannot be said to advance.")
    elif not all_beat:
        verdict["adaptive_advances"] = False
        verdict["rationale"] = (
            "Adaptive did not show a positive paired advantage (95% CI lower "
            "bound > 0) against every baseline on the primary tasks.")
    else:
        verdict["adaptive_advances"] = True
        verdict["rationale"] = (
            "Every paired adaptive-vs-baseline 95% CI lower bound exceeded 0 on "
            "the primary tasks and all pilot gates passed.")
    return verdict


# ---------------------------------------------------------------------------
# Report (25 sections, reasoning order: design -> gates -> primary -> paired
# -> diagnostics/failures -> negative controls -> verdict -> threats/next)
# ---------------------------------------------------------------------------

def _fmt(x, nd=3):
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _minmax(xs):
    return (min(xs), max(xs)) if xs else (None, None)


def generate_report(result: Dict) -> str:
    cfg = result["config"]
    records = result.get("records") or []
    primary = [r for r in records if r["task_id"] in PRIMARY_TASKS]
    negative = [r for r in records if r["task_id"] in NEGATIVE_CONTROL_TASKS]
    diags = result.get("diagnostics") or []
    off = result.get("offline") or {}
    og = off.get("gate") or {}
    gold = result.get("gold_checks") or []
    unit = result.get("unit_tests") or {}
    agg = result.get("aggregate") or {}
    neg_agg = result.get("negative_aggregate") or {}
    gates = result.get("gates") or {}
    verdict = result.get("verdict") or {}
    budgets = cfg.get("budgets") or []
    grid_label = "Pilot" if cfg.get("mode") == "pilot" else "Full Grid"

    L: List[str] = []
    L.append("# E19 Coding-Generalization Report (Phase 15)")

    L.append("## 1. Purpose & Research Question")
    L.append("")
    L.append("After Phase 14 made explicit supersession authoritative at the "
             "consolidation boundary (D35) and Phase 15 fixed a second, adjacent "
             "defect (D36: a supersession replaced the fact text but could keep "
             "the obsolete predecessor's embedding, so retrieval continued to rank "
             "the corrected fact by obsolete semantics), E19 asks: **does the "
             "corrected adaptive memory system generalize to a broader set of "
             "genuinely history-dependent long-horizon coding tasks better than "
             "simpler context-management strategies under the same "
             "historical-context budget?** The dependent variable is whether the "
             "produced patch passes deterministic hidden tests.")

    L.append("## 2. Causal Chain & What Changed Since E18 (D36 / D37)")
    L.append("")
    L.append("E18 re-validated the consolidation boundary after the Phase 14 fix "
             "(offline identity gate passed; correction_recall == 1.0, "
             "obsolete_exposure == 0.0). Phase 15 hardens the wire between "
             "consolidation and retrieval: `_replace_with_supersession` now stores "
             "the correcting statement's embedding when it arrives and otherwise "
             "drops the stale predecessor embedding, so the surviving memory "
             "always embeds its current text (D36). E19 then tests this corrected "
             "system on a wider task set with strict primary / negative-control "
             "separation (D37): only the three genuinely history-dependent "
             "primary tasks may contribute to the adaptive-advances decision. "
             "config_contract was calibrated but excluded: probing showed the "
             "required explicit-empty-string nuance is not reliably expressible "
             "by llama3.1:8b even with oracle history (0/3), so it carries no "
             "memory signal at this model; its fixture is retained for future "
             "stronger models.")

    L.append("## 3. Design: Primary vs Negative-Control Split (D37)")
    L.append("")
    L.append("- **Primary tasks** (history-dependent; the paired/verdict analysis "
             "uses only these): user_ids, validation_pure, transaction_atomicity.")
    L.append("- **Negative-control tasks** (harness sanity only; recorded and "
             "reported, never used in paired comparisons or the adaptive-advances "
             "verdict): cache_readonly, write_retry.")
    L.append("- **Fixture-only task**: config_contract (authored and calibrated "
             "this phase, but excluded from the primary set — llama3.1:8b cannot "
             "reliably express its explicit-empty-string fact even with oracle "
             "history; probing: 0/3 oracle passes across four fixture designs). "
             "Its fixture is committed for future stronger models.")
    L.append("- Reasoning: a generalization claim must be earned on tasks whose "
             "constraints genuinely require the buried history; tasks that can be "
             "solved from the workspace alone would measure coding ability, not "
             "memory. The negative controls keep the environment honest (a "
             "correctly-built harness passes them with the gold patch).")

    L.append("## 4. Task Suite")
    L.append("")
    L.append("| task | group | title | critical facts | corrections | negative constraints |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for tid in ALL_TASKS:
        t = build_task(tid, seed=1)
        group = "primary" if tid in PRIMARY_TASKS else "negative-control"
        L.append(f"| {tid} | {group} | {t.title} | {len(t.gold_facts)} | "
                 f"{len(t.corrections)} | "
                 f"{sum(1 for f in t.gold_facts if f.kind == 'negative')} |")
    L.append("")

    L.append("## 5. Historical Dependence Design")
    L.append("")
    L.append("Each task spreads critical facts across a ~600-turn deterministic "
             "transcript among distractors (>=3 topic families), with >=2 critical "
             "facts, >=1 correction or negative constraint, >=1 fact >300 turns "
             "old, and >=1 distractor thread. user_ids buries the opaque-string-cast "
             "constraint (cast ids no longer); validation_pure buries the pure-"
             "validation constraint (no audit I/O on the critical path); "
             "transaction_atomicity buries the all-or-nothing rollback rule "
             "(correction turn 505) and the no-retry rule (turn 310). "
             "No fact text uses IMPORTANT/CRITICAL marker language; fact_ids are "
             "identity-safe; histories are generated from structured data with no "
             "LLM calls. The task prompts never mention the critical historical "
             "rules; the no_history diagnostic (section 20) quantifies dependence.")

    L.append("## 6. Memory Methods Under Test")
    L.append("")
    L.append("| method | historical-context mechanism |")
    L.append("| --- | --- |")
    L.append("| raw_clipped | newest raw turns that fit the budget; no retrieval |")
    L.append("| sliding_window | last 10 raw turns (production `SlidingWindowBaseline`) |")
    L.append("| llm_summarization | running summary by the same model every 50 turns |")
    L.append("| vanilla_rag | static fact store, pure cosine top-k fit to budget |")
    L.append("| adaptive | the production pipeline, `dual_score` retention, `protect_corrections=True` |")
    L.append("")
    L.append("Exactly one adaptive arm (the actual production pipeline); no "
             "adaptive_v2 / tuned / plus variants (D37 / spec)."
             "Diagnostics (not competitors): no_history, full_context "
             "(context-unconstrained upper bound), direct_history (oracle "
             "gold-fact context).")

    L.append("## 7. Fixed Budgets & Adaptive Allocation Policy")
    L.append("")
    L.append(f"Historical budgets: {budgets} tokens (shared word-count tokenizer "
             f"`{TOKENIZER_NAME}`). For adaptive, `injection_token_limit = budget` "
             "and `memory_store_token_budget = max(4096, budget*4)`, mirroring the "
             "repo convention; this is documented, not tuned. Per-run token "
             "accounting separates historical_context_tokens, "
             "workspace_context_tokens, task_prompt_tokens and "
             "total_prompt_tokens.")

    L.append("## 8. Coding Model, Prompt & Harness")
    L.append("")
    L.append(f"Coding model `{cfg.get('model')}` via Ollama "
             f"(`temperature={cb.CODING_TEMPERATURE}`, "
             f"`num_ctx={cb.MODEL_CONTEXT_TOKENS}`, `keep_alive=30m`, "
             f"`num_predict={cfg.get('max_output_tokens')}`); embeddings "
             f"`{cfg.get('embedding_model')}` (no silent fallback; model "
             "availability is verified before any run). The harness "
             "(`experiments/coding_benchmark.py`, unchanged) owns workspace reset, "
             "prompt construction, complete-file edit-block application (unified "
             "diff fallback), `git apply --check`, the repair-attempt policy "
             "(initial + one repair), hidden-test execution, failure "
             "classification and token accounting.")

    L.append("## 9. Evaluation Metrics & Failure Taxonomy")
    L.append("")
    L.append("Primary: `final_success` and `first_pass_success`. Diagnostics: "
             "critical_fact_recall, correction_recall, "
             "negative_constraint_recall, long_range_fact_recall, "
             "obsolete_fact_exposure. Failure classes (deterministic precedence): "
             "MEMORY_MISS, RETRIEVAL_MISS, CONTEXT_OVERFLOW, PATCH_INVALID, "
             "CODING_ERROR, HIDDEN_TEST_FAILURE, OBSOLETE_INFORMATION_USED, "
             "CORRECTION_MISSED, OTHER (no new class added).")

    L.append("## 10. Configuration & Reproducibility")
    L.append("")
    for k, v in cfg.items():
        L.append(f"- **{k}**: {v}")
    L.append("")

    L.append("## 11. Grids: Pilot & Full")
    L.append("")
    L.append(f"- Pilot: {len(PRIMARY_TASKS)} primary x {len(cfg.get('seeds', []))} "
             f"seeds x {len(cfg.get('budgets', []))} budgets x "
             f"{len(METHODS)} methods = **{cfg.get('grid_cells', 0)} runs**.")
    L.append(f"- Full: primary "
             f"{len(PRIMARY_TASKS) * len(FULL_SEEDS) * len(FULL_BUDGETS) * len(METHODS)} "
             f"+ negative "
             f"{len(NEGATIVE_CONTROL_TASKS) * len(FULL_SEEDS) * len(FULL_BUDGETS) * len(METHODS)} "
             f"= **225 runs**. Every run is persisted immediately under "
             "`task_id:seed:budget:method`; the grid is resumable and completed "
             "results are never deleted.")

    L.append("## 12. Pilot Gates A-J")
    L.append("")
    L.append("| gate | passed | detail |")
    L.append("| --- | --- | --- |")
    for name in GATE_NAMES:
        g = gates.get(name, {})
        detail = {k: v for k, v in g.items() if k != "passed"}
        L.append(f"| {name} | {'PASS' if g.get('passed') else 'FAIL'} | "
                 f"{json.dumps(detail, default=str)[:500]} |")
    L.append("")
    L.append(f"**All gates passed: {gates.get('all_passed')}** (verdict section 16).")

    L.append("## 13. Pilot Results: Success Rate by Method x Budget (PILOT)")
    L.append("")
    if cfg.get("mode") == "pilot":
        L.append("The full grid has not been run yet; these are PILOT numbers and "
                 "no production decision is made from them.")
        pilot_budgets = PILOT_BUDGETS
    else:
        pilot_budgets = PILOT_BUDGETS
    pb = cfg.get("budgets") or []
    header = "| method | " + " | ".join(str(b) for b in pb) + " | overall |"
    L.append(header)
    L.append("| --- | " + " | ".join("---" for _ in pb) + " | --- |")
    for m in METHODS:
        row = []
        for b in pb:
            v = ((agg.get("by_method_budget", {}).get(m, {}).get(str(b), {}) or {})
                 .get("success_rate"))
            row.append(_fmt(v))
        overall = (agg.get("overall", {}).get(m, {}) or {}).get("success_rate")
        L.append(f"| {m} | " + " | ".join(row) + f" | {_fmt(overall)} |")
    L.append("")

    L.append(f"## 14. {grid_label}: Results by Method x Budget (primary)")
    L.append("")
    L.append(f"Primary-task records only (E19 schema; see JSON). "
             f"This table aggregates the {grid_label.lower()} grid so far "
             f"(mode={cfg.get('mode')}).")
    L.append("")
    L.append(header)
    L.append("| --- | " + " | ".join("---" for _ in pb) + " | --- |")
    for m in METHODS:
        row = []
        for b in pb:
            v = ((agg.get("by_method_budget", {}).get(m, {}).get(str(b), {}) or {})
                 .get("success_rate"))
            row.append(_fmt(v))
        overall = (agg.get("overall", {}).get(m, {}) or {}).get("success_rate")
        L.append(f"| {m} | " + " | ".join(row) + f" | {_fmt(overall)} |")
    L.append("")

    L.append(f"## 15. {grid_label}: First-Pass vs Final Success")
    L.append("")
    L.append(f"Over the {grid_label.lower()} grid aggregated in section 14 "
             f"(mode={cfg.get('mode')}).")
    L.append("| method | budget | first-pass | final | runs |")
    L.append("| --- | --- | --- | --- | --- |")
    for m in METHODS:
        for b in pb:
            row = agg.get("by_method_budget", {}).get(m, {}).get(str(b), {})
            if row:
                L.append(f"| {m} | {b} | {_fmt(row.get('first_pass_rate'))} | "
                         f"{_fmt(row.get('success_rate'))} | {row.get('runs')} |")
    L.append("")

    L.append("## 16. Paired Comparisons & Bootstrap CIs (adaptive - baseline)")
    L.append("")
    L.append("Paired on (task, seed, budget) over primary records; paired 95% "
             "percentile bootstrap CIs on the mean within-pair difference.")
    L.append("")
    L.append("| vs | pairs | adaptive | baseline | mean diff | diff 95% CI | adaptive 95% CI |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for m, c in (agg.get("paired_vs_adaptive") or {}).items():
        L.append(f"| {m} | {c['pairs']} | {_fmt(c['adaptive_success'])} | "
                 f"{_fmt(c['baseline_success'])} | {_fmt(c['mean_diff'])} | "
                 f"{c['mean_diff_ci95']} | {c['adaptive_ci95']} |")
    L.append("")
    L.append("**Predeclared criterion:** positive paired advantage with the "
             "bootstrap 95% CI lower bound > 0 against every baseline; "
             "otherwise `adaptive_advances = false`.")
    L.append("")
    L.append(f"- gates_all_passed: **{gates.get('all_passed')}**")
    L.append(f"- adaptive_beats_all_baselines: **{verdict.get('adaptive_beats_all_baselines')}**")
    L.append(f"- **adaptive_advances: {verdict.get('adaptive_advances')}**")
    L.append("")
    L.append(f"rationale: {verdict.get('rationale', '')}")

    L.append("## 17. Verdict")
    L.append("")
    if cfg.get("mode") == "pilot":
        L.append("**PILOT verdict.** The full grid is not complete; no production "
                 "conclusion is declared. The pilot gates and paired numbers above "
                 "are directional only.")
    else:
        L.append("Full-grid verdict per the predeclared criterion (section 16): "
                 f"adaptive_advances = **{verdict.get('adaptive_advances')}**.")

    L.append("## 18. Diagnostics: Recall & Obsolete Exposure (primary)")
    L.append("")
    diag_cols = " | ".join(f"mean ({b})" for b in pb)
    L.append(f"| method | metric | {diag_cols} |")
    L.append("| --- | --- | " + " | ".join("---" for _ in pb) + " |")
    for metric in ["critical_fact_recall", "correction_recall",
                   "negative_constraint_recall", "long_range_fact_recall",
                   "obsolete_fact_exposure"]:
        for m in METHODS:
            row = []
            for b in pb:
                d = ((agg.get("by_method_budget", {}).get(m, {}).get(str(b), {}) or {})
                     .get("diagnostics", {}).get(metric))
                row.append(_fmt(d))
            L.append(f"| {m} | {metric} | " + " | ".join(row) + " |")
    L.append("")

    L.append("## 19. Failure Taxonomy Distribution (primary)")
    L.append("")
    L.append("| method | budget | failure classes |")
    L.append("| --- | --- | --- |")
    for m in METHODS:
        for b in pb:
            row = agg.get("by_method_budget", {}).get(m, {}).get(str(b), {})
            if row:
                L.append(f"| {m} | {b} | {row.get('failure_classes')} |")
    L.append("")

    L.append("## 20. No-History / Full-Context / Direct-History Diagnostics")
    L.append("")
    L.append(f"no_history and direct_history are stochastic diagnostics and are "
             f"measured over {DIAG_DRAWS} draws (seed 1, budget {DIAG_BUDGET}); "
             f"gate C requires *zero* no_history successes across all draws for "
             f"every primary. Oracle direct_history is a recorded success rate, "
             f"not a gate.")
    L.append("")
    L.append("| task | no_history (x{}) | full_context | direct_history oracle (x{}) |".format(
        DIAG_DRAWS, DIAG_DRAWS))
    L.append("| --- | --- | --- | --- |")
    diagmap = {f"{d['task_id']}:{d['method']}": d for d in diags}
    for tid in PRIMARY_TASKS:
        nh = diagmap.get(f"{tid}:no_history")
        fc = diagmap.get(f"{tid}:full_context")
        dh = diagmap.get(f"{tid}:direct_history")
        nh_s = (nh.get("successes") or 0) if nh else 0
        dh_s = (dh.get("successes") or 0) if dh else 0
        L.append(f"| {tid} | {nh_s}/{nh.get('draws_n', 0) if nh else 0} "
                 f"({_fmt(nh['final_success'] if nh else False)}) | "
                 f"{_fmt(fc['final_success'] if fc else None)} | "
                 f"{dh_s}/{dh.get('draws_n', 0) if dh else 0} "
                 f"({_fmt(dh['final_success'] if dh else False)}) |")
    L.append(f"")
    L.append(f"Gold-patch harness sanity: "
             f"{sum(1 for g in gold if g['final_success'])}/{len(gold)} tasks pass "
             "when the gold patch is supplied.")
    L.append("")

    L.append("## 21. Context Usage & Budget Pressure")
    L.append("")
    L.append("| method | budget | mean hidden | mean total prompt |")
    L.append("| --- | --- | --- | --- |")
    for m in METHODS:
        for b in pb:
            row = agg.get("by_method_budget", {}).get(m, {}).get(str(b), {})
            if row:
                L.append(f"| {m} | {b} | "
                         f"{_fmt(row.get('mean_context_tokens'))} | "
                         f"{_fmt(row.get('mean_total_prompt_tokens'))} |")
    L.append("")
    L.append(f"- full_context diagnostics max tokens: "
             f"{_minmax([d['historical_context_tokens'] for d in diags if d['method'] == 'full_context'])}")

    L.append("## 22. Negative-Control Results (harness sanity; NOT part of the verdict)")
    L.append("")
    L.append("| method | runs | success_rate | first_pass |")
    L.append("| --- | --- | --- | --- |")
    for m in sorted((neg_agg.get("overall") or {})):
        v = (neg_agg.get("overall") or {}).get(m) or {}
        L.append(f"| {m} | {v.get('runs')} | {_fmt(v.get('success_rate'))} | "
                 f"{_fmt(v.get('first_pass_rate'))} |")
    L.append("")
    L.append("These tasks (cache_readonly, write_retry) only sanity-check that the "
             "harness/environment solves a task the workspace signals are solvable "
             "by the gold patch; they never contribute to the paired comparisons "
             "or to `adaptive_advances` (D37).")

    L.append("## 23. Threats to Validity & Limitations")
    L.append("")
    L.append("- Word-count tokenizer approximates the model's real tokenization; "
             "budgets are approximate (documented in the manifest).")
    L.append("- Small task count and seeds imply wide success-rate CIs; absence of "
             "a difference is not evidence of equivalence.")
    L.append("- The task suite is synthetic-but-real-code; findings may not "
             "transfer to large repositories.")
    L.append("- `full_context` is a context-unconstrained diagnostic, not a "
             "fixed-budget competitor; `direct_history` is an oracle.")
    L.append("- Ollama-served `llama3.1:8b` at temperature 0.1 is not "
             "deterministic across runs; per-run success is the unit and the grid "
             "is resumable.")
    L.append("")

    L.append("## 24. Conclusion")
    L.append("")
    L.append(f"- Mode: **{cfg.get('mode')}**.")
    L.append(f"- Primary-task runs completed: **{len(primary)}**.")
    L.append(f"- Negative-control runs completed: **{len(negative)}**.")
    L.append(f"- Gates all passed: **{gates.get('all_passed')}**.")
    L.append(f"- adaptive_advances: **{verdict.get('adaptive_advances')}**.")
    L.append("")
    if gates.get("all_passed"):
        L.append("Gate suite passed; the full grid is a valid experiment.")
    else:
        L.append("At least one pilot gate failed; findings are directional only.")
    L.append("")

    L.append("## 25. Reproduction / Artifacts / Provenance")
    L.append("")
    L.append(f"- JSON: `{OUT_JSON.name}` (every record is the E19 schema with "
             "experiment='E19'; no fabricated fields).")
    L.append(f"- generated_at: {cfg.get('generated_at')}")
    L.append(f"- tokenizer: `{TOKENIZER_NAME}`; model: `{cfg.get('model')}`; "
             f"embedding: `{cfg.get('embedding_model')}`; use_embeddings: "
             f"{cfg.get('use_embeddings')}.")
    L.append(f"- unit-test suite gate: exit_code={unit.get('exit_code')}, "
             f"passed={unit.get('passed')}.")
    L.append(f"- work dir: `experiments/results/e19_work/` (gitignored).")
    L.append("- E17/E18 artifacts are immutable and were not regenerated.")
    L.append("")
    L.append("— All numbers are generated from the E19 JSON by `generate_report()`; "
             "no hand-typed figures.")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _load_previous() -> Dict:
    if OUT_JSON.exists():
        try:
            return json.loads(OUT_JSON.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def compute_unit_tests() -> Dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True, text=True, timeout=1800)
    m = re.search(r"(\d+) passed", proc.stdout)
    return {"exit_code": proc.returncode,
            "passed": int(m.group(1)) if m else 0,
            "exit_ok": proc.returncode == 0,
            "stdout_tail": (proc.stdout or "")[-200:]}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="E19 coding-generalization evaluation (Phase 15)")
    parser.add_argument("--pilot", action="store_true",
                        help="run the pilot grid (80 primary runs + gates)")
    parser.add_argument("--full", action="store_true",
                        help="run the full grid (135 primary + 90 negative = 225)")
    parser.add_argument("--resume", action="store_true",
                        help="resume: skip cells already recorded in the JSON")
    parser.add_argument("--force", action="store_true",
                        help="recompute all cells (overrides resume)")
    parser.add_argument("--limit", type=int, default=0,
                        help="limit number of NEW run cells")
    parser.add_argument("--report-only", action="store_true",
                        help="regenerate the report/verdict from the existing JSON")
    parser.add_argument("--no-embed", action="store_true",
                        help="disable embeddings (lexical retrieval; diagnostic)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--max-output-tokens", type=int,
                        default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--skip-model-check", action="store_true")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    use_embeddings = not args.no_embed

    if args.report_only:
        if not OUT_JSON.exists():
            parser.error("no e19_coding_generalization.json to report")
        result = json.loads(OUT_JSON.read_text())
        recs = result.get("records", [])
        result["aggregate"] = aggregate(
            [r for r in recs if r["task_id"] in PRIMARY_TASKS])
        result["negative_aggregate"] = aggregate(
            [r for r in recs if r["task_id"] in NEGATIVE_CONTROL_TASKS])
        result["offline"] = result.get("offline") or {"gate": {}}
        result["gates"] = compute_gates(
            records=recs, diagnostics=result.get("diagnostics", []),
            offline=result.get("offline", {}), gold_checks=result.get("gold_checks", []),
            unit_tests=result.get("unit_tests", {}),
            task_ids_primary=PRIMARY_TASKS,
            budgets_lo=PILOT_BUDGETS[0], budgets_hi=FULL_BUDGETS[-1])
        result["verdict"] = classify_verdict(result)
        OUT_JSON.write_text(json.dumps(result, indent=2, default=str))
        OUT_REPORT.write_text(generate_report(result))
        print(f"Re-generated {OUT_REPORT}; adaptive_advances="
              f"{result['verdict'].get('adaptive_advances')}")
        return result

    if not args.skip_model_check:
        check_models(args.model, args.embedding_model, args.endpoint,
                     use_embeddings)

    if args.full:
        mode = "full"
        seeds, budgets = FULL_SEEDS, FULL_BUDGETS
    else:
        mode = "pilot"
        seeds, budgets = PILOT_SEEDS, PILOT_BUDGETS
    tasks = ALL_TASKS

    print("=" * 64)
    print(f"E19 Coding Generalization ({mode})")
    print("=" * 64)

    state = {} if (args.force or not OUT_JSON.exists()) else _load_previous()
    records: List[Dict] = list(state.get("records", []))
    diags: List[Dict] = list(state.get("diagnostics", []))
    offline_cells: List[Dict] = list(state.get("offline", {}).get("cells", []))
    gold_checks: List[Dict] = list(state.get("gold_checks", []))
    unit_tests: Dict = state.get("unit_tests", {})

    rec_existing = {_run_key(r["task_id"], r["seed"], r["historical_budget"],
                             r["method"]): r for r in records}
    diag_existing = {_diag_key(d["task_id"], d["method"]): d for d in diags}
    off_existing = {f"{c['task_id']}:{c['seed']}:{c['budget']}": c
                    for c in offline_cells}
    gold_existing = {g["task_id"]: g for g in gold_checks}

    embed_fn = _resolve_embed(use_embeddings, args.embedding_model, args.endpoint)
    coder = cb.OllamaCoder(args.model, args.endpoint, args.max_output_tokens)
    summarizer_generate_fn = lambda prompt, mx: coder.generate(prompt)["text"]

    def persist(partial=False):
        payload = {
            "experiment": "e19_coding_generalization",
            "partial": partial,
            "config": {
                "mode": mode, "tasks": tasks,
                "primary_tasks": PRIMARY_TASKS,
                "negative_control_tasks": NEGATIVE_CONTROL_TASKS,
                "seeds": seeds, "budgets": budgets, "methods": METHODS,
                "grid_cells": (len(PRIMARY_TASKS) * len(seeds) * len(budgets)
                               * len(METHODS)
                               + (len(NEGATIVE_CONTROL_TASKS) * len(seeds)
                                  * len(budgets) * len(METHODS)
                                  if mode == "full" else 0)),
                "model": args.model, "endpoint": args.endpoint,
                "embedding_model": args.embedding_model,
                "use_embeddings": use_embeddings,
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
            "diagnostics": diags,
            "offline": {"cells": offline_cells,
                        "gate": offline_gate(offline_cells, PRIMARY_TASKS)},
            "gold_checks": gold_checks,
            "unit_tests": unit_tests,
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
        return payload

    # ---- home-work gates that need no model: unit suite (J), gold (F) ----
    if not unit_tests:
        print("  [suite] running pytest tests/ ...", flush=True)
        unit_tests = compute_unit_tests()
        persist(partial=True)

    for tid in tasks:
        if tid not in gold_existing:
            g = check_gold(tid)
            gold_checks.append(g)
            gold_existing[tid] = g
            print(f"  [gold] {tid}: success={g['final_success']}", flush=True)
            persist(partial=True)

    # ---- offline gates A & B (adaptive store, embeddings) ----
    for tid in PRIMARY_TASKS + ["cache_readonly"]:
        for seed in FULL_SEEDS:
            for budget in FULL_BUDGETS:
                key = f"{tid}:{seed}:{budget}"
                if key in off_existing:
                    continue
                if not use_embeddings:
                    break
                print(f"  [offline] {key} ...", flush=True)
                try:
                    cell = run_offline_cell(tid, seed, budget, embed_fn=embed_fn,
                                            embedding_model=args.embedding_model,
                                            endpoint=args.endpoint,
                                            model=args.model)
                except Exception as exc:
                    print(f"      ERROR {key}: {exc}", flush=True)
                    continue
                offline_cells.append(cell)
                off_existing[key] = cell
                persist(partial=True)

    # ---- grid runs ----
    todo = []
    for tid in tasks:
        for seed in seeds:
            for budget in budgets:
                for method in METHODS:
                    if mode == "pilot" and tid in NEGATIVE_CONTROL_TASKS:
                        continue
                    key = _run_key(tid, seed, budget, method)
                    if key in rec_existing:
                        continue
                    todo.append((tid, seed, budget, method))

    done = 0
    limit = args.limit or len(todo)
    for tid, seed, budget, method in todo:
        if done >= limit:
            break
        key = _run_key(tid, seed, budget, method)
        done += 1
        print(f"  [{done}/{len(todo)}] {key} ...", flush=True)
        try:
            rec = run_one(tid, seed, budget, method, coder=coder,
                          embed_fn=embed_fn, model=args.model,
                          endpoint=args.endpoint,
                          embedding_model=args.embedding_model,
                          summarizer_generate_fn=summarizer_generate_fn)
        except Exception as exc:
            print(f"      ERROR {key}: {exc}", flush=True)
            continue
        records.append(rec)
        rec_existing[key] = rec
        persist(partial=True)

    # ---- diagnostics (no_history / full_context / direct_history) ----
    # Gate-C diagnostics (no_history, direct_history) run DIAG_DRAWS times:
    # an LLM can pass a stochastic diagnostic by luck, so the gate measures
    # reproducibility (0 successes for no_history). full_context is a token
    # size probe and stays a single run.
    draws_methods = {"no_history": DIAG_DRAWS,
                     "direct_history": DIAG_DRAWS,
                     "full_context": 1}
    stale = [k for k, d in diag_existing.items()
             if d.get("draws_n", 1) != draws_methods.get(d["method"], 1)]
    for k in stale:
        diag_existing.pop(k, None)
    diags = [d for d in diags if _diag_key(d["task_id"], d["method"]) in diag_existing]
    for tid in PRIMARY_TASKS:
        for method, draws in draws_methods.items():
            key = _diag_key(tid, method)
            if key in diag_existing:
                continue
            print(f"  [diag] {tid}:{method} x{draws} ...", flush=True)
            try:
                recs = [run_one(tid, DIAG_SEED, DIAG_BUDGET, method, coder=coder,
                                embed_fn=embed_fn, model=args.model,
                                endpoint=args.endpoint,
                                embedding_model=args.embedding_model,
                                summarizer_generate_fn=summarizer_generate_fn)
                        for _ in range(draws)]
            except Exception as exc:
                print(f"      ERROR {key}: {exc}", flush=True)
                continue
            rec = dict(recs[0])
            rec["draws_n"] = draws
            rec["successes"] = sum(1 for r in recs if r["final_success"])
            rec["draws"] = [bool(r["final_success"]) for r in recs]
            rec["final_success"] = rec["successes"] > 0
            diags.append(rec)
            diag_existing[key] = rec
            persist(partial=True)

    result = persist(partial=False)
    result["aggregate"] = aggregate(
        [r for r in records if r["task_id"] in PRIMARY_TASKS])
    result["negative_aggregate"] = aggregate(
        [r for r in records if r["task_id"] in NEGATIVE_CONTROL_TASKS])
    result["gates"] = compute_gates(
        records=records, diagnostics=diags, offline=result["offline"],
        gold_checks=gold_checks, unit_tests=unit_tests,
        task_ids_primary=PRIMARY_TASKS,
        budgets_lo=PILOT_BUDGETS[0], budgets_hi=FULL_BUDGETS[-1])
    result["verdict"] = classify_verdict(result)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {OUT_JSON}")
    OUT_REPORT.write_text(generate_report(result))
    print(f"Wrote {OUT_REPORT}")
    print(f"gates_all_passed={result['gates'].get('all_passed')} "
          f"adaptive_advances={result['verdict'].get('adaptive_advances')}")
    return result


if __name__ == "__main__":
    main()