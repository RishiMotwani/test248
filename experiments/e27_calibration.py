"""E27 counterfactual calibration repair (Phase 22 / D44).

Purpose
-------
E26 (Phase 21) could not separate the production adaptive pipeline from the
vanilla RAG baseline on the discriminative policy fixtures: the pilot was
stopped at two failed gates (``history_dependence`` and ``contradiction_state``)
before any full grid was legal. E27 therefore does NOT re-run E26, does NOT
run a full grid and does NOT compare adaptive to vanilla RAG. Its only job is
to calibrate a cleaner measurement surface for a future head-to-head: every
task collapsed to exactly ONE decisive historical policy, byte-identical
shared workspace/prompt per group, a single 96-token budget, and fixed 48-cell
geometry. This runner measures whether that calibrated surface behaves as the
memory system expects.

Grid (fixed, 48 cells)
----------------------
3 groups (``route_contract``, ``serialization_contract``, ``retry_contract``) x
2 variants x seeds {1, 2} x budget 96 x
``{no_history, direct_history, vanilla_rag, adaptive}``.

Offline fixture gates (must pass before any LLM runs)
-----------------------------------------------------
A base workspace fails the hidden test, B gold patch passes, C obsolete patch
fails, D/E workspace + prompt byte-identical across variants, F history
differs across variants, G fact token geometry / placement / depth, and H
structured correction metadata — plus hygiene checks (no policy mapping in the
visible surface, no hidden-test language in history, probe terms in fact text,
current correction not a duplicate of the obsolete wording).

Pilot gates (locked; nothing is tuned on pilot data)
----------------------------------------------------
G1 history dependence (per group over 4 cells): ``direct_history`` >= 3/4
pass, ``no_history`` <= 1/4 pass, per-group difference >= 2 cells.
G2 contradiction state (pooled per recall method over the 12 pilot cells):
``adaptive`` correction_recall >= 0.90 and obsolete_fact_exposure <= 0.10;
``vanilla_rag`` obsolete_fact_exposure >= 0.75.
G3 budget binding: >= 75% of each recall method's cells sit at >= 0.8x the
96-token budget.
G4 context difference: >= 80% of the 12 adaptive-vs-vanilla paired cells
differ in ``context_sha``.
G5 no leakage: no pilot run leaked hidden-test material into its prompt.

Verdict
-------
``calibration_valid`` (and ``full_head_to_head_eligible``) is True only when
every pilot gate passes on a complete 48-cell grid. E27 never declares a
winner and never claims an adaptive-vs-RAG result; a head-to-head grid is a
separate future phase that must be explicitly commissioned.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from experiments import coding_benchmark as cb
from experiments.e17_coding_capability import check_models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.e27_calibration_suite import (  # noqa: E402
    as_coding_task,
    build_variant,
    validate_all_fixtures,
)

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_ENDPOINT = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_JSON = RESULTS_DIR / "e27_calibration.json"
OUT_REPORT = RESULTS_DIR / "e27_calibration_report.md"
WORK_ROOT = RESULTS_DIR / "e27_work"

GROUPS = ["route_contract", "serialization_contract", "retry_contract"]
VARIANTS = ["A", "B"]
SEEDS = [1, 2]
BUDGETS = [96]
METHODS = ["no_history", "direct_history", "vanilla_rag", "adaptive"]

RECALL_METHODS = ["vanilla_rag", "adaptive"]
NON_RECALL_METHODS = ["no_history", "direct_history"]

DIRECT_MIN_SUCCESS = 0.75        # >= 3/4
NO_HISTORY_MAX_SUCCESS = 0.25    # <= 1/4
HISTORY_DIFF_MIN = 2.0           # per-group diff in cells

ADAPTIVE_MIN_CORR_RECALL = 0.90
ADAPTIVE_MAX_OBS_EXPOSURE = 0.10
VANILLA_MIN_OBS_EXPOSURE = 0.75

BUDGET_BINDING_RATIO = 0.8
BUDGET_BINDING_MIN_FRACTION = 0.75   # >= 9/12 cells
CONTEXT_DIFF_MIN_FRACTION = 0.8

STATE_CLEAN_CURRENT = "CLEAN_CURRENT"
STATE_CURRENT_PLUS_OBSOLETE = "CURRENT_PLUS_OBSOLETE"
STATE_OBSOLETE_ONLY = "OBSOLETE_ONLY"
STATE_NEITHER = "NEITHER"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _run_key(group: str, variant: str, seed: int, budget: int, method: str) -> str:
    return f"{group}:{variant}:{seed}:{budget}:{method}"


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _success(records: List[Dict]) -> Optional[float]:
    if not records:
        return None
    return round(
        sum(1 for r in records if r["final_success"]) / len(records), 4)


def _sha16(text: str) -> Optional[str]:
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _resolve_embed(embedding_model: str, endpoint: str):
    from memory_optimizer.embeddings import embed_ollama
    return lambda texts: embed_ollama(texts, model=embedding_model,
                                      endpoint=endpoint)


class StaticCoder:
    """Gold-patch coder used to prove the fixtures are solvable offline."""

    def __init__(self, patch: str, model: str = "static"):
        self.patch = patch
        self.model = model

    def generate(self, prompt: str) -> Dict:
        return {"text": "```diff\n" + self.patch + "\n```",
                "prompt_tokens": 0, "output_tokens": 0, "latency_ms": 0.0}


def _correction_states(corrections, context_ids: set) -> Dict[str, int]:
    counts = {STATE_CLEAN_CURRENT: 0, STATE_CURRENT_PLUS_OBSOLETE: 0,
              STATE_OBSOLETE_ONLY: 0, STATE_NEITHER: 0}
    for c in corrections:
        cur_in = c.current_fact_id in context_ids
        obs_in = c.obsolete_fact_id in context_ids
        if cur_in and not obs_in:
            counts[STATE_CLEAN_CURRENT] += 1
        elif cur_in and obs_in:
            counts[STATE_CURRENT_PLUS_OBSOLETE] += 1
        elif obs_in:
            counts[STATE_OBSOLETE_ONLY] += 1
        else:
            counts[STATE_NEITHER] += 1
    return counts


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def run_one(group: str, variant: str, seed: int, budget: int, method_name: str,
            *, coder, embed_fn, model: str, endpoint: str,
            embedding_model: str) -> Dict:
    task = build_variant(group, variant, seed=seed)
    ctask = as_coding_task(task)
    method = cb.build_method(
        method_name, model=model, endpoint=endpoint, embed_fn=embed_fn,
        embedding_model=embedding_model, task=ctask)
    root = WORK_ROOT / f"{group}_{variant}_{seed}_{budget}_{method_name}"
    rec = cb.run_method_run(task=ctask, seed=seed, historical_budget=budget,
                            method=method, coder=coder, work_root=root)
    context_ids = set(getattr(method, "last_context_ids", set()) or set())
    diag = rec.get("diagnostics") or {}
    budget_ratio = (round(rec["historical_context_tokens"] / budget, 3)
                    if budget else 0.0)
    return {
        "experiment": "E27",
        "task_id": f"{group}_{variant}",
        "group": group,
        "variant": variant,
        "seed": seed,
        "historical_budget": budget,
        "method": method_name,
        "historical_context_tokens": rec["historical_context_tokens"],
        "workspace_context_tokens": rec["workspace_context_tokens"],
        "task_prompt_tokens": rec["task_prompt_tokens"],
        "total_prompt_tokens": rec["total_prompt_tokens"],
        "first_pass_success": bool(rec["first_pass_success"]),
        "final_success": bool(rec["final_success"]),
        "hidden_test_pass": bool(rec["final_success"]),
        "critical_fact_recall": diag.get("critical_fact_recall"),
        "correction_recall": diag.get("correction_recall"),
        "long_range_fact_recall": diag.get("long_range_fact_recall"),
        "obsolete_fact_exposure": diag.get("obsolete_fact_exposure"),
        "budget_ratio": budget_ratio,
        "correction_state_counts": _correction_states(ctask.corrections,
                                                      context_ids),
        "correction_current_ids": sorted(
            {c.current_fact_id for c in ctask.corrections}
            & context_ids),
        "correction_obsolete_ids": sorted(
            {c.obsolete_fact_id for c in ctask.corrections}
            & context_ids),
        "failure_class": rec["failure_class"],
        "context_sha": rec.get("context_sha"),
        "patch_sha": _sha16(rec.get("patch")),
        "leakage_detected": bool((rec.get("leakage") or {})
                                 .get("leakage_detected")),
        "uses_production_pipeline": bool(rec.get("uses_production_pipeline")),
        "latency_ms": rec.get("latency_ms"),
    }


def gold_check(group: str, variant: str, seed: int = 1) -> Dict:
    task = build_variant(group, variant, seed=seed)
    ctask = as_coding_task(task)
    gold = ctask.gold_patch_path.read_text(encoding="utf-8")
    coder = StaticCoder(gold)
    root = WORK_ROOT / f"goldcheck_{group}_{variant}"
    method = cb.build_method("no_history", model="static", endpoint="http://x",
                             embed_fn=None, embedding_model="", task=ctask)
    rec = cb.run_method_run(task=ctask, seed=seed, historical_budget=256,
                            method=method, coder=coder, work_root=root)
    return {"group": group, "variant": variant, "seed": seed,
            "final_success": bool(rec["final_success"])}


# ---------------------------------------------------------------------------
# Pilot gates
# ---------------------------------------------------------------------------

def _history_dependence(records: List[Dict]) -> Dict:
    per_group = {}
    all_pass = True
    for gid in GROUPS:
        grp = [r for r in records if r["group"] == gid]
        dh = [r for r in grp if r["method"] == "direct_history"]
        nh = [r for r in grp if r["method"] == "no_history"]
        dh_rate = _success(dh) if dh else None
        nh_rate = _success(nh) if nh else None
        dh_ok = dh_rate is not None and dh_rate >= DIRECT_MIN_SUCCESS
        nh_ok = nh_rate is not None and nh_rate <= NO_HISTORY_MAX_SUCCESS
        diff = (sum(1 for r in dh if r["final_success"])
                - sum(1 for r in nh if r["final_success"]))
        diff_ok = diff >= HISTORY_DIFF_MIN
        passed = dh_ok and nh_ok and diff_ok
        all_pass = all_pass and passed
        per_group[gid] = {
            "cells": len(dh),
            "direct_history_success": dh_rate,
            "no_history_success": nh_rate,
            "direct_ok": dh_ok, "no_history_ok": nh_ok,
            "diff_cells": diff, "diff_ok": diff_ok,
            "passed": passed,
        }
    return {"per_group": per_group, "passed": all_pass}


def _contradiction_state(records: List[Dict]) -> Dict:
    out = {}
    for m in RECALL_METHODS:
        rows = [r for r in records if r["method"] == m]
        recall = _mean([r["correction_recall"] for r in rows])
        exposure = _mean([r["obsolete_fact_exposure"] for r in rows])
        if m == "adaptive":
            ok = (recall is not None and recall >= ADAPTIVE_MIN_CORR_RECALL
                  and exposure is not None
                  and exposure <= ADAPTIVE_MAX_OBS_EXPOSURE)
        else:
            ok = (exposure is not None
                  and exposure >= VANILLA_MIN_OBS_EXPOSURE)
        out[m] = {"cells": len(rows), "correction_recall": recall,
                  "obsolete_fact_exposure": exposure, "passed": ok}
    passed = all(out[m]["passed"] for m in RECALL_METHODS)
    return {"methods": out, "passed": passed}


def _budget_binding(records: List[Dict], budget: int) -> Dict:
    out = {}
    for m in RECALL_METHODS:
        rows = [r for r in records if r["method"] == m]
        bound = [r for r in rows
                 if r["budget_ratio"] >= BUDGET_BINDING_RATIO]
        frac = len(bound) / len(rows) if rows else 0.0
        out[m] = {"cells": len(rows), "bound_cells": len(bound),
                  "fraction": round(frac, 3),
                  "passed": frac >= BUDGET_BINDING_MIN_FRACTION}
    passed = all(out[m]["passed"] for m in RECALL_METHODS)
    return {"methods": out, "passed": passed}


def _context_difference(records: List[Dict]) -> Dict:
    pairs = []
    for gid in GROUPS:
        for variant in VARIANTS:
            for seed in SEEDS:
                key_adapted = _run_key(gid, variant, seed, BUDGETS[0],
                                       "adaptive")
                key_vanilla = _run_key(gid, variant, seed, BUDGETS[0],
                                       "vanilla_rag")
                by_key = {_run_key(r["group"], r["variant"], r["seed"],
                                   r["historical_budget"], r["method"]): r
                          for r in records}
                a = by_key.get(key_adapted)
                b = by_key.get(key_vanilla)
                if a is None or b is None:
                    continue
                pairs.append((gid, variant, seed,
                              a["context_sha"] != b["context_sha"]))
    differ = sum(1 for _, _, _, d in pairs if d)
    frac = differ / len(pairs) if pairs else 0.0
    return {"pairs": len(pairs), "differ": differ,
            "fraction": round(frac, 3),
            "passed": bool(pairs) and frac >= CONTEXT_DIFF_MIN_FRACTION}


def compute_pilot_gates(records: List[Dict]) -> Dict:
    g1 = _history_dependence(records)
    g2 = _contradiction_state(records)
    g3 = _budget_binding(records, BUDGETS[0])
    g4 = _context_difference(records)
    leaks = [r for r in records if r.get("leakage_detected")]
    g5 = {"leaked_cells": len(leaks),
          "passed": len(leaks) == 0,
          "leaked": [f"{r['task_id']}:{r['seed']}" for r in leaks]}
    gates = {
        "history_dependence": g1,
        "contradiction_state": g2,
        "budget_binding": g3,
        "context_difference": g4,
        "no_leakage": g5,
    }
    passed = all(gates[k]["passed"] for k in gates)
    return {"gates": gates, "passed": passed}


# ---------------------------------------------------------------------------
# Grid completeness + verdict
# ---------------------------------------------------------------------------

def _expected_keys() -> List[str]:
    return sorted(
        _run_key(g, v, s, b, m)
        for g in GROUPS for v in VARIANTS for s in SEEDS
        for b in BUDGETS for m in METHODS)


def grid_completeness(records: List[Dict]) -> Dict:
    have = {_run_key(r["group"], r["variant"], r["seed"],
                     r["historical_budget"], r["method"]) for r in records}
    expected = set(_expected_keys())
    missing = sorted(expected - have)
    return {"expected": len(expected), "present": len(expected - set(missing)),
            "missing": missing,
            "complete": not missing}


def compute_verdict(*, pilot_gates: Dict, fixture_gates: Dict,
                    grid: Dict) -> Dict:
    gates_ok = bool(pilot_gates.get("passed"))
    fixtures_ok = bool(fixture_gates.get("passed"))
    grid_ok = bool(grid.get("complete"))
    calibration_valid = bool(gates_ok and fixtures_ok and grid_ok)
    per_flag = {name: bool((pilot_gates.get("gates") or {}).get(name, {})
                           .get("passed"))
                for name in ["history_dependence", "contradiction_state",
                             "budget_binding", "context_difference",
                             "no_leakage"]}
    rationale = []
    if not fixtures_ok:
        rationale.append("offline fixture gates did not pass")
    if not grid_ok:
        rationale.append("pilot grid incomplete")
    if not gates_ok:
        rationale.append("pilot gates did not pass")
    return {
        "calibration_valid": calibration_valid,
        "history_dependence_passed": per_flag["history_dependence"],
        "contradiction_state_passed": per_flag["contradiction_state"],
        "budget_binding_passed": per_flag["budget_binding"],
        "context_difference_passed": per_flag["context_difference"],
        "no_leakage_passed": per_flag["no_leakage"],
        "full_head_to_head_eligible": calibration_valid,
        "grid_complete": grid_ok,
        "fixture_gates_passed": fixtures_ok,
        "pilot_gates_passed": gates_ok,
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Diagnostics / aggregate
# ---------------------------------------------------------------------------

def _state_totals(records: List[Dict]) -> Dict[str, int]:
    total = defaultdict(int)
    for r in records:
        for state, n in (r.get("correction_state_counts") or {}).items():
            total[state] += n
    return dict(total)


def aggregate(records: List[Dict]) -> Dict:
    out = {}
    for m in sorted({r["method"] for r in records}):
        rows = [r for r in records if r["method"] == m]
        out[m] = {
            "runs": len(rows),
            "success_rate": _success(rows),
            "first_pass_rate": round(
                sum(1 for r in rows if r["first_pass_success"]) / len(rows), 4)
            if rows else None,
            "mean_context_tokens": _mean(
                [r["historical_context_tokens"] for r in rows]),
            "mean_budget_ratio": _mean([r["budget_ratio"] for r in rows]),
            "correction_recall": _mean([r["correction_recall"]
                                        for r in rows]),
            "obsolete_fact_exposure": _mean([r["obsolete_fact_exposure"]
                                             for r in rows]),
            "long_range_fact_recall": _mean([r["long_range_fact_recall"]
                                             for r in rows]),
            "correction_state_counts": _state_totals(rows),
            "contexts": len({r["context_sha"] for r in rows}),
        }
    return out


def compute_diagnostics(*, records: List[Dict], gold_checks: List[Dict]) -> Dict:
    return {
        "gold_checks": gold_checks,
        "grid_completeness": grid_completeness(records),
        "methods": aggregate(records),
    }


# ---------------------------------------------------------------------------
# Reports (sections 1-15, Tables A-D)
# ---------------------------------------------------------------------------

def _fmt(x, nd=3):
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _gate_table(name: str, gate: Dict) -> List[str]:
    L = [f"### Gate {name}", "",
         f"- status: {'PASS' if gate['passed'] else 'FAIL'}"]
    if "per_group" in gate:
        for gid, g in gate["per_group"].items():
            L.append(f"  - {gid}: "
                     + "; ".join(f"{k}={g[k]}" for k in g if k != "passed"))
    if "methods" in gate:
        for m, g in gate["methods"].items():
            L.append(f"  - {m}: "
                     + "; ".join(f"{k}={g[k]}" for k in g if k != "passed"))
    if "leaked_cells" in gate:
        L.append(f"  - leaked_cells={gate['leaked_cells']}")
        L.append(f"  - leaked={gate['leaked']}")
    L.append("")
    return L


def generate_report(result: Dict) -> str:
    cfg = result["config"]
    records = result.get("records") or []
    fixtures = result.get("fixture_gates") or {}
    gates = result.get("pilot_gates") or {}
    agg = result.get("aggregate") or {}
    diag = result.get("diagnostics") or {}
    verdict = result.get("verdict") or {}
    grid = diag.get("grid_completeness") or {}
    L: List[str] = []
    L.append("# E27 Counterfactual Calibration Repair - Pilot Report (Phase 22)")
    L.append("")
    L.append("## 1. Purpose")
    L.append("")
    L.append("E27 calibrates a *single-policy* counterfactual measurement "
             "surface for a future head-to-head benchmark. It does not run a "
             "full grid, does not declare a winner, and does not claim any "
             "adaptive-vs-RAG result: a head-to-head grid is a separate future "
             "phase that must be explicitly commissioned.")
    L.append("")
    L.append("## 2. Relation to E26 (Phase 21)")
    L.append("")
    L.append("E26 was stopped at the pilot because two gates failed: "
             "``history_dependence`` (release_adapter direct 0/4; "
             "message 2/4; invoice 3/4) and ``contradiction_state`` "
             "(adaptive corr_recall 0.94 / obs_exposure 0.0; vanilla "
             "obs_exposure 0.29 < 0.75). Its 54-cell full grid never ran and "
             "``adaptive_beats_vanilla`` is False. E27 repairs the surface by "
             "reducing each task to exactly ONE decisive historical policy "
             "with one obsolete fact, one current correction and four "
             "distractors at a single 96-token budget. The E26 suite, runner "
             "and results are byte-identical and are never re-run.")
    L.append("")
    L.append("## 3. Design")
    L.append("")
    L.append("- groups: route_contract, serialization_contract, "
             "retry_contract")
    L.append("- each group: byte-identical workspace+prompt across variants; "
             "hidden history is the only difference")
    L.append("- facts per variant: 1 obsolete policy fact (turns 80-150, "
             "50-56 tokens), 1 current correction (turns 470-530, 50-56 "
             "tokens), 4 distractors (18-24 tokens); obsolete+current sum "
             "> 96 tokens so the two sides of the conflict never both fit")
    L.append("- base workspace fails the hidden test; gold patch passes; "
             "obsolete-only patch fails")
    L.append("")
    L.append("## 4. Config")
    L.append("")
    L.append(f"- groups: {cfg['groups']}")
    L.append(f"- variants: {cfg['variants']} seeds: {cfg['seeds']} "
             f"budgets: {cfg['budgets']}")
    L.append(f"- methods: {cfg['methods']}")
    L.append(f"- model: {cfg['model']} embedding: {cfg['embedding_model']}")
    L.append(f"- cells: {len(records)} / {cfg['grid_cells']} expected")
    L.append("")
    L.append("## 5. Offline fixture gates (Table A)")
    L.append("")
    L.append("| variant | status | failing checks |")
    L.append("|---|---|---|")
    for k, v in (fixtures.get("variants") or {}).items():
        L.append(f"| {k} | {'PASS' if v['passed'] else 'FAIL'} | "
                 f"{' '.join(v['failing'])} |")
    L.append("")
    L.append(f"- **all 6 variants validated: "
             f"{'PASS' if fixtures.get('passed') else 'FAIL'}**")
    L.append("")
    L.append("## 6. Grid completeness")
    L.append("")
    L.append(f"- cells: {grid.get('present')} / {grid.get('expected')} "
             f"expected; complete={grid.get('complete')}; "
             f"missing={grid.get('missing')}")
    L.append("")
    L.append("## 7. Pilot gates - history dependence (Table B)")
    L.append("")
    for gid, g in ((gates.get("gates") or {}).get("history_dependence", {})
                   .get("per_group") or {}).items():
        L.append(f"- {gid}: direct={_fmt(g['direct_history_success'], 2)} "
                 f"(ok={g['direct_ok']}) no_history="
                 f"{_fmt(g['no_history_success'], 2)} (ok={g['no_history_ok']}) "
                 f"diff={g['diff_cells']} (ok={g['diff_ok']}) -> "
                 f"{'PASS' if g['passed'] else 'FAIL'}")
    L.append("")
    L.append("## 8. Pilot gates - contradiction state")
    L.append("")
    for name in ["contradiction_state", "budget_binding",
                 "context_difference", "no_leakage"]:
        L.extend(_gate_table(name, (gates.get("gates") or {}).get(name, {
            "passed": False})))
    L.append(f"- **all pilot gates: "
             f"{'PASS' if gates.get('passed') else 'FAIL'}**")
    L.append("")
    L.append("## 9. Per-group x method success (Table C)")
    L.append("")
    L.append("| group | no_history | direct_history | vanilla_rag | adaptive |")
    L.append("|---|---|---|---|---|")
    for gid in GROUPS:
        grp = [r for r in records if r["group"] == gid]
        row = []
        for m in METHODS:
            row.append(_fmt(_success([r for r in grp if r["method"] == m]), 2))
        L.append(f"| {gid} | {' | '.join(row)} |")
    L.append("")
    L.append("## 10. Per-method aggregates (Table D)")
    L.append("")
    L.append("| method | runs | success | first-pass | ctx tokens | "
             "budget ratio | corr recall | obs exposure | states |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for m, g in agg.items():
        st = g["correction_state_counts"]
        states = (f"CC:{st.get('CLEAN_CURRENT', 0)} "
                  f"CPO:{st.get('CURRENT_PLUS_OBSOLETE', 0)} "
                  f"OO:{st.get('OBSOLETE_ONLY', 0)} N:{st.get('NEITHER', 0)}")
        L.append(f"| {m} | {g['runs']} | {_fmt(g['success_rate'], 2)} | "
                 f"{_fmt(g['first_pass_rate'], 2)} | "
                 f"{_fmt(g['mean_context_tokens'], 0)} | "
                 f"{_fmt(g['mean_budget_ratio'], 2)} | "
                 f"{_fmt(g['correction_recall'], 2)} | "
                 f"{_fmt(g['obsolete_fact_exposure'], 2)} | {states} |")
    L.append("")
    L.append("## 11. Correction-state distribution (diagnostics)")
    L.append("")
    for m, g in agg.items():
        st = g["correction_state_counts"]
        L.append(f"- {m}: {st}")
    L.append("")
    L.append("## 12. Gold checks (StaticCoder, offline)")
    L.append("")
    gchecks = diag.get("gold_checks") or []
    ok = all(gc.get("final_success") is True for gc in gchecks)
    L.append(f"- gold patches pass the hidden tests for all 6 variants: "
             f"{'PASS' if ok else 'FAIL'}")
    L.append("")
    L.append("## 13. Verdict")
    L.append("")
    for k in ["calibration_valid", "history_dependence_passed",
              "contradiction_state_passed", "budget_binding_passed",
              "context_difference_passed", "no_leakage_passed",
              "full_head_to_head_eligible"]:
        L.append(f"- {k}: {verdict.get(k)}")
    L.append(f"- rationale: {verdict.get('rationale')}")
    L.append("")
    L.append("## 14. Stop rule")
    L.append("")
    L.append("E27 stops after this pilot. Calibration validity is judged on the "
             "five locked gates and a complete 48-cell grid; nothing is tuned "
             "on pilot outcomes. Even if ``full_head_to_head_eligible`` is "
             "True, E27 does not run or claim a head-to-head and declares no "
             "winner.")
    L.append("")
    L.append("## 15. Threats to validity")
    L.append("")
    L.append("- Facts are oracle-pre-extracted; extraction drift is out of "
             "scope and would bias both recall arms identically.")
    L.append("- A single 96-token budget is used; the calibration does not "
             "sweep budget sensitivity.")
    L.append("- Vanilla RAG observation mode and re-ranking depend on the "
             "embedding model (nomic-embed-text); a different embedder could "
             "shift vanilla exposure.")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="E27 counterfactual calibration repair (Phase 22)")
    parser.add_argument("--validate", action="store_true",
                        help="run offline fixture gates only (no LLM)")
    parser.add_argument("--pilot", action="store_true",
                        help="run the 48-cell calibration pilot + gates")
    parser.add_argument("--resume", action="store_true",
                        help="resume: skip cells already present")
    parser.add_argument("--force", action="store_true",
                        help="recompute all cells")
    parser.add_argument("--report-only", action="store_true",
                        help="regenerate report/verdict from existing JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--skip-model-check", action="store_true")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        out = (json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {})
        if not out:
            parser.error("no existing results to report")
        out["pilot_gates"] = out.get("pilot_gates") or compute_pilot_gates(
            out.get("records") or [])
        out["aggregate"] = out.get("aggregate") or aggregate(
            out.get("records") or [])
        out["diagnostics"] = compute_diagnostics(
            records=out.get("records") or [],
            gold_checks=out.get("diagnostics", {}).get("gold_checks") or [])
        out["verdict"] = compute_verdict(
            pilot_gates=out["pilot_gates"],
            fixture_gates=out.get("fixture_gates") or {},
            grid=out["diagnostics"]["grid_completeness"])
        OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
        OUT_REPORT.write_text(generate_report(out))
        print("calibration_valid="
              f"{out['verdict'].get('calibration_valid')}")
        return

    fixtures = validate_all_fixtures()
    if not fixtures["passed"]:
        print("[STOP] offline fixture gates failed; no LLM may run.")
        OUT_JSON.write_text(json.dumps(
            {"experiment": "e27_calibration", "mode": "validate-failed",
             "fixture_gates": fixtures}, indent=2, default=str))
        for k, v in fixtures["variants"].items():
            print(k, "->", f"FAIL {v['failing']}")
        return

    if args.validate:
        OUT_JSON.write_text(json.dumps(
            {"experiment": "e27_calibration", "mode": "validate",
             "fixture_gates": fixtures}, indent=2, default=str))
        print("fixture gates passed=", fixtures["passed"])
        for k, v in fixtures["variants"].items():
            print(k, "->", "PASS" if v["passed"] else f"FAIL {v['failing']}")
        return

    if args.pilot:
        if not args.skip_model_check:
            check_models(args.model, args.embedding_model, args.endpoint,
                         use_embeddings=True)
        embed_fn = _resolve_embed(args.embedding_model, args.endpoint)
        coder = cb.OllamaCoder(args.model, args.endpoint)
    else:
        parser.error("E27 runs exactly one grid: pass --validate or --pilot")

    records: List[Dict] = []
    if not args.force and OUT_JSON.exists():
        existing = json.loads(OUT_JSON.read_text()) or {}
        keep = set(_expected_keys())
        records = [r for r in existing.get("records", [])
                   if _run_key(r["group"], r["variant"], r["seed"],
                               r["historical_budget"], r["method"]) in keep]

    grid_cells = (len(GROUPS) * len(VARIANTS) * len(SEEDS) * len(BUDGETS)
                  * len(METHODS))
    have = {_run_key(r["group"], r["variant"], r["seed"],
                     r["historical_budget"], r["method"]) for r in records}
    todo = [k for k in _expected_keys() if args.force or k not in have]

    config = {
        "mode": "pilot", "groups": GROUPS, "variants": VARIANTS,
        "seeds": SEEDS, "budgets": BUDGETS, "methods": METHODS,
        "grid_cells": grid_cells,
        "model": args.model, "endpoint": args.endpoint,
        "embedding_model": args.embedding_model,
        "temperature": cb.CODING_TEMPERATURE,
        "tokenizer": cb.TOKENIZER_NAME,
        "ingestion": "oracle_pre_extracted",
        "adaptive": {"retention_mode": "dual_score"},
        "generated_at": time.time(),
    }

    print("=" * 64)
    print(f"E27 Calibration Repair (pilot) - {len(todo)} cells to run")
    print("=" * 64)

    def persist(partial: bool = False) -> Dict:
        payload = {
            "experiment": "e27_calibration",
            "partial": partial,
            "config": config,
            "fixture_gates": fixtures,
            "records": records,
        }
        payload["pilot_gates"] = compute_pilot_gates(records)
        payload["aggregate"] = aggregate(records)
        payload["diagnostics"] = compute_diagnostics(
            records=records, gold_checks=gold_checks)
        OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
        return payload

    gold_checks: List[Dict] = []
    print("[gold] validating gold patches with StaticCoder ...")
    for gid in GROUPS:
        for variant in VARIANTS:
            gold_checks.append(gold_check(gid, variant))
    persist(partial=True)

    if not todo:
        print("nothing to run; count=", len(records))
    else:
        for cell in todo:
            group, variant, seed, budget, method = cell.split(":")
            seed_i, budget_i = int(seed), int(budget)
            rec = run_one(group, variant, seed_i, budget_i, method,
                          coder=coder, embed_fn=embed_fn,
                          model=args.model, endpoint=args.endpoint,
                          embedding_model=args.embedding_model)
            records = [r for r in records
                       if _run_key(r["group"], r["variant"], r["seed"],
                                   r["historical_budget"], r["method"])
                       != cell] + [rec]
            print(f"  [done] {cell}: success={rec['final_success']} "
                  f"tokens={rec['historical_context_tokens']} "
                  f"corr_recall={rec['correction_recall']} "
                  f"obs_exposure={rec['obsolete_fact_exposure']}")
            persist(partial=True)

    pilot_gates = compute_pilot_gates(records)
    grid = grid_completeness(records)
    verdict = compute_verdict(pilot_gates=pilot_gates,
                              fixture_gates=fixtures, grid=grid)
    payload = {
        "experiment": "e27_calibration",
        "partial": False,
        "config": config,
        "fixture_gates": fixtures,
        "pilot_gates": pilot_gates,
        "records": records,
        "aggregate": aggregate(records),
        "diagnostics": compute_diagnostics(records=records,
                                           gold_checks=gold_checks),
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    OUT_REPORT.write_text(generate_report(payload))
    print("calibration_valid=", verdict["calibration_valid"])
    print("full_head_to_head_eligible=", verdict["full_head_to_head_eligible"])
    print("pilot gates passed=", pilot_gates["passed"])
    return payload


if __name__ == "__main__":
    main()