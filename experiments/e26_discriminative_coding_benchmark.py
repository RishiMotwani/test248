"""E26 discriminative coding benchmark (Phase 21).

Head-to-head between the production adaptive pipeline (`adaptive`) and the
vanilla RAG baseline (`vanilla_rag`) on the discriminative policy fixtures
from ``data/discriminative_coding_suite.py``. The grid and gates are fixed and
predeclared; nothing here is tuned after data collection.

Grid
----
Pilot (48 cells): 3 groups x 2 variants x seeds {1,2} x budget 96 x
``{no_history, direct_history, vanilla_rag, adaptive}``.
Full (54 cells): 3 groups x variant A x seeds {1,2,3} x budgets
{64, 96, 128} x ``{vanilla_rag, adaptive}``.

Pilot gates (all must pass before the full grid is legal)
---------------------------------------------------------
G1 history dependence (per group): ``direct_history`` >= 3/4 pass,
``no_history`` <= 1/4 pass, and the per-group difference >= 2 cells.
G2 contradiction state (pooled per recall method over the 12 pilot cells):
``adaptive`` correction_recall >= 0.75 and obsolete_fact_exposure <= 0.25;
``vanilla_rag`` obsolete_fact_exposure >= 0.75.
G3 budget binding: >= 75% of each recall method's pilot cells use >= 0.8x the
96-token budget.
G4 context difference: >= 80% of the 12 adaptive-vs-vanilla paired
(group, variant, seed) cells differ in ``context_sha``.
G5 no leakage: no pilot run leaked hidden-test material into its prompt.

Verdict
-------
``adaptive_beats_vanilla`` requires all pilot gates to pass, the full 54-cell
grid to be complete, and the paired bootstrap 95% CI lower bound
(``statistics.bootstrap_ci_mean_diff``) over the 27 (group, seed, budget)
pairs to be > 0. The E19/E24/E20/E25 artifacts are never modified.
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
from experiments.statistics import bootstrap_ci_mean_diff, bootstrap_mean_ci

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.discriminative_coding_suite import (  # noqa: E402
    as_coding_task,
    build_variant,
    validate_all_fixtures,
)

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_ENDPOINT = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_JSON = RESULTS_DIR / "e26_discriminative_coding_benchmark.json"
OUT_PILOT_JSON = RESULTS_DIR / "e26_discriminative_coding_benchmark_pilot.json"
OUT_REPORT = RESULTS_DIR / "e26_discriminative_coding_benchmark_report.md"
OUT_PILOT_REPORT = (
    RESULTS_DIR / "e26_discriminative_coding_benchmark_pilot_report.md"
)
WORK_ROOT = RESULTS_DIR / "e26_work"

GROUPS = ["release_adapter", "invoice_adapter", "message_adapter"]

PILOT_VARIANTS = ["A", "B"]
PILOT_SEEDS = [1, 2]
PILOT_BUDGETS = [96]
PILOT_METHODS = ["no_history", "direct_history", "vanilla_rag", "adaptive"]

FULL_VARIANTS = ["A"]
FULL_SEEDS = [1, 2, 3]
FULL_BUDGETS = [64, 96, 128]
FULL_METHODS = ["vanilla_rag", "adaptive"]

RECALL_METHODS = ["vanilla_rag", "adaptive"]

DIRECT_MIN_SUCCESS = 0.75
NO_HISTORY_MAX_SUCCESS = 0.25
HISTORY_DIFF_MIN = 2.0

ADAPTIVE_MIN_CORR_RECALL = 0.75
ADAPTIVE_MAX_OBS_EXPOSURE = 0.25
VANILLA_MIN_OBS_EXPOSURE = 0.75

BUDGET_BINDING_RATIO = 0.8
BUDGET_BINDING_MIN_FRACTION = 0.75
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


def _mean(xs):
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
        "experiment": "E26",
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
        for variant in PILOT_VARIANTS:
            for seed in PILOT_SEEDS:
                key_adapted = _run_key(gid, variant, seed, PILOT_BUDGETS[0],
                                       "adaptive")
                key_vanilla = _run_key(gid, variant, seed, PILOT_BUDGETS[0],
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
    g3 = _budget_binding(records, PILOT_BUDGETS[0])
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
# Grid completeness + paired comparison + verdict
# ---------------------------------------------------------------------------

def _expected_keys(*, mode: str) -> List[str]:
    if mode == "pilot":
        return sorted(
            _run_key(g, v, s, b, m)
            for g in GROUPS for v in PILOT_VARIANTS for s in PILOT_SEEDS
            for b in PILOT_BUDGETS for m in PILOT_METHODS)
    return sorted(
        _run_key(g, v, s, b, m)
        for g in GROUPS for v in FULL_VARIANTS for s in FULL_SEEDS
        for b in FULL_BUDGETS for m in FULL_METHODS)


def grid_completeness(records: List[Dict], *, mode: str) -> Dict:
    have = {_run_key(r["group"], r["variant"], r["seed"],
                     r["historical_budget"], r["method"]) for r in records}
    expected = set(_expected_keys(mode=mode))
    missing = sorted(expected - have)
    return {"expected": len(expected), "present": len(expected - set(missing)),
            "missing": missing,
            "complete": not missing}


def paired_comparison(records: List[Dict]) -> Dict:
    by_key = {}
    for r in records:
        by_key[(r["group"], r["seed"], r["historical_budget"],
                r["method"])] = int(r["final_success"])
    keys = sorted({k[:3] for k in by_key
                   if k[3] in RECALL_METHODS})
    a = [by_key[(g, s, b, "adaptive")] for g, s, b in keys
         if (g, s, b, "adaptive") in by_key
         and (g, s, b, "vanilla_rag") in by_key]
    b = [by_key[(g, s, b, "vanilla_rag")] for g, s, b in keys
         if (g, s, b, "adaptive") in by_key
         and (g, s, b, "vanilla_rag") in by_key]
    if len(a) == 0:
        return {"pairs": 0, "cells": [], "adaptive_success": None,
                "vanilla_success": None, "mean_diff": None,
                "mean_diff_ci95": [None, None],
                "adaptive_ci95": [None, None],
                "vanilla_ci95": [None, None]}
    return {
        "pairs": len(a),
        "cells": [f"{g}:{s}:{b}" for g, s, b in keys[:len(a)]],
        "adaptive_success": round(sum(a) / len(a), 4),
        "vanilla_success": round(sum(b) / len(b), 4),
        "mean_diff": round(sum(x - y for x, y in zip(a, b)) / len(a), 4),
        "mean_diff_ci95": bootstrap_ci_mean_diff(a, b),
        "adaptive_ci95": bootstrap_mean_ci(a),
        "vanilla_ci95": bootstrap_mean_ci(b),
    }


def compute_verdict(*, pilot_gates: Dict, full_grid: Dict,
                    paired: Dict, fixture_gates: Dict) -> Dict:
    gates_ok = bool(pilot_gates.get("passed"))
    grid_ok = bool(full_grid.get("complete"))
    fixtures_ok = bool(fixture_gates.get("passed"))
    ci_lo = (paired.get("mean_diff_ci95") or [None, None])[0]
    ci_positive = ci_lo is not None and ci_lo > 0
    beats = gates_ok and grid_ok and fixtures_ok and ci_positive
    rationale = []
    if not fixtures_ok:
        rationale.append("offline fixture gates did not pass")
    if not gates_ok:
        rationale.append("pilot gates did not pass")
    if not grid_ok:
        rationale.append("full grid incomplete")
    if not ci_positive:
        rationale.append("paired CI lower bound not positive")
    return {
        "adaptive_beats_vanilla": bool(beats),
        "pilot_gates_passed": gates_ok,
        "full_grid_complete": grid_ok,
        "fixture_gates_passed": fixtures_ok,
        "ci_lower_bound": ci_lo,
        "paired": paired,
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


# ---------------------------------------------------------------------------
# Reports
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


def generate_pilot_report(result: Dict) -> str:
    cfg = result["config"]
    records = result.get("records") or []
    fixtures = result.get("fixture_gates") or {}
    gates = result.get("pilot_gates") or {}
    agg = result.get("aggregate") or {}
    verdict = result.get("verdict") or {}
    L: List[str] = []
    L.append("# E26 Discriminative Benchmark - Pilot Report (Phase 21)")
    L.append("")
    L.append("## 1. Purpose")
    L.append("")
    L.append("Does the production adaptive pipeline keep only the current "
             "policy facts in context when the historical contract is "
             "contradicted, and does vanilla RAG drown the current contract "
             "in superseded decisions? This pilot fixes the 48-cell grid, the "
             "five gates and the pairs for the later full grid.")
    L.append("")
    L.append(f"## 2. Config")
    L.append("")
    L.append(f"- groups: {cfg['groups']}")
    L.append(f"- variants: {cfg['variants']}")
    L.append(f"- seeds: {cfg['seeds']} budgets: {cfg['budgets']}")
    L.append(f"- methods: {cfg['methods']}")
    L.append(f"- model: {cfg['model']} embedding: {cfg['embedding_model']}")
    L.append(f"- cells: {len(records)} / {cfg['grid_cells']} expected")
    L.append("")
    L.append("## 3. Offline fixture gates")
    L.append("")
    L.append(f"- all 6 variants validated: "
             f"{'PASS' if fixtures.get('passed') else 'FAIL'}")
    for k, v in (fixtures.get("variants") or {}).items():
        L.append(f"  - {k}: {'PASS' if v['passed'] else 'FAIL ' + str(v['failing'])}")
    L.append("")
    L.append("## 4. Pilot gates")
    L.append("")
    for name in ["history_dependence", "contradiction_state",
                 "budget_binding", "context_difference", "no_leakage"]:
        L.extend(_gate_table(name, (gates.get("gates") or {}).get(name, {
            "passed": False})))
    L.append(f"- **all pilot gates: "
             f"{'PASS' if gates.get('passed') else 'FAIL'}**")
    L.append("")
    L.append("## 5. Per-method pilot results")
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
    L.append("## 6. Paired adaptive vs vanilla (pilot, 12 pairs)")
    L.append("")
    p = verdict.get("paired") or {}
    L.append(f"- pairs: {p.get('pairs')} mean_diff {_fmt(p.get('mean_diff'))} "
             f"CI95 {p.get('mean_diff_ci95')}")
    L.append("")
    L.append("## 7. Verdict (pilot)")
    L.append("")
    L.append("- `adaptive_beats_vanilla` (full decision forwarded to the full "
             "grid): the full-grid CI is computed over 27 paired cells; the "
             "pilot only verifies the gates.")
    L.append("")
    L.append("## 8. Threats")
    L.append("")
    L.append("- The fixture gate set is offline and deterministic; any gate "
             "failure here stops the full grid by construction.")
    L.append("- Pilot cells run at a single budget (96); budgets 64 and 128 "
             "are held for the full grid only.")
    L.append("")
    return "\n".join(L)


def generate_full_report(result: Dict) -> str:
    cfg = result["config"]
    records = result.get("records") or []
    fixtures = result.get("fixture_gates") or {}
    gates = result.get("pilot_gates") or {}
    full_grid = result.get("full_grid") or {}
    paired = result.get("paired") or {}
    agg = result.get("aggregate") or {}
    verdict = result.get("verdict") or {}
    L: List[str] = []
    L.append("# E26 Discriminative Benchmark - Full Grid Report (Phase 21)")
    L.append("")
    L.append("## 1. Research question")
    L.append("")
    L.append("Across 3 policy groups, is the adaptive pipeline's injected "
             "context more discriminative (current contract present, "
             "superseded contract absent) than vanilla RAG, and does that "
             "translate to a statistically clean success advantage on "
             "variant-A fixtures?")
    L.append("")
    L.append(f"## 2. Config")
    L.append("")
    L.append(f"- groups: {cfg['groups']}; variants: {cfg['variants']}")
    L.append(f"- seeds: {cfg['seeds']}; budgets: {cfg['budgets']}")
    L.append(f"- methods: {cfg['methods']}")
    L.append(f"- grid cells: {len(records)} / {cfg['grid_cells']} expected; "
             f"missing={full_grid.get('missing')}")
    L.append("")
    L.append("## 3. Offline fixture gates")
    L.append("")
    L.append(f"- all 6 variants validated: "
             f"{'PASS' if fixtures.get('passed') else 'FAIL'}")
    L.append("")
    L.append("## 4. Pilot gates (retained)")
    L.append("")
    L.append(f"- history dependence: "
             f"{'PASS' if (gates.get('gates') or {}).get('history_dependence', {}).get('passed') else 'FAIL'}")
    L.append(f"- contradiction state: "
             f"{'PASS' if (gates.get('gates') or {}).get('contradiction_state', {}).get('passed') else 'FAIL'}")
    L.append(f"- budget binding: "
             f"{'PASS' if (gates.get('gates') or {}).get('budget_binding', {}).get('passed') else 'FAIL'}")
    L.append(f"- context difference: "
             f"{'PASS' if (gates.get('gates') or {}).get('context_difference', {}).get('passed') else 'FAIL'}")
    L.append(f"- no leakage: "
             f"{'PASS' if (gates.get('gates') or {}).get('no_leakage', {}).get('passed') else 'FAIL'}")
    L.append(f"- **all pilot gates: "
             f"{'PASS' if gates.get('passed') else 'FAIL'}**")
    L.append("")
    L.append("## 5. Per-method full-grid results")
    L.append("")
    L.append("| method | runs | success | first-pass | ctx tokens | "
             "budget ratio | corr recall | obs exposure | contexts |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for m, g in agg.items():
        L.append(f"| {m} | {g['runs']} | {_fmt(g['success_rate'], 2)} | "
                 f"{_fmt(g['first_pass_rate'], 2)} | "
                 f"{_fmt(g['mean_context_tokens'], 0)} | "
                 f"{_fmt(g['mean_budget_ratio'], 2)} | "
                 f"{_fmt(g['correction_recall'], 2)} | "
                 f"{_fmt(g['obsolete_fact_exposure'], 2)} | "
                 f"{g['contexts']} |")
    L.append("")
    L.append("## 6. Paired adaptive vs vanilla")
    L.append("")
    L.append(f"- pairs: {paired.get('pairs')}")
    L.append(f"- adaptive success: {_fmt(paired.get('adaptive_success'))}")
    L.append(f"- vanilla success: {_fmt(paired.get('vanilla_success'))}")
    L.append(f"- mean diff: {_fmt(paired.get('mean_diff'))}")
    L.append(f"- paired 95% CI (adaptive - vanilla): "
             f"{paired.get('mean_diff_ci95')}")
    L.append("")
    L.append("## 7. Verdict")
    L.append("")
    for k in ["adaptive_beats_vanilla", "pilot_gates_passed",
              "full_grid_complete", "fixture_gates_passed", "ci_lower_bound"]:
        L.append(f"- {k}: {verdict.get(k)}")
    L.append(f"- rationale: {verdict.get('rationale')}")
    L.append("")
    L.append("## 8. Threats to validity")
    L.append("")
    L.append("- The benchmark uses oracle-pre-extracted facts; production "
             "extraction drift is out of scope and would bias both arms.")
    L.append("- Budgets 64-128 tokens are severe; small budgets favour "
             "aggressive selection.")
    L.append("- Variant B fixtures are used only for the pilot "
             "history-dependence and contradiction-state calibration.")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="E26 discriminative coding benchmark (Phase 21)")
    parser.add_argument("--validate", action="store_true",
                        help="run offline fixture gates (no LLM)")
    parser.add_argument("--pilot", action="store_true",
                        help="run the 48-cell pilot + gates")
    parser.add_argument("--full", action="store_true",
                        help="run the 54-cell full grid (variant A)")
    parser.add_argument("--resume", action="store_true",
                        help="resume: skip cells already present")
    parser.add_argument("--force", action="store_true",
                        help="recompute all cells")
    parser.add_argument("--report-only", action="store_true",
                        help="regenerate reports/verdict from existing JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--skip-model-check", action="store_true")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        pilot = (json.loads(OUT_PILOT_JSON.read_text())
                 if OUT_PILOT_JSON.exists() else {})
        full = (json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {})
        if full:
            out = {**full, "pilot_gates": pilot.get("pilot_gates") or {}}
            out["fixture_gates"] = full.get("fixture_gates") or pilot.get(
                "fixture_gates") or {}
            out["paired"] = paired_comparison(out.get("records") or [])
            out["verdict"] = compute_verdict(
                pilot_gates=out["pilot_gates"],
                full_grid=out.get("full_grid") or {},
                paired=out["paired"],
                fixture_gates=out["fixture_gates"])
            OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
            OUT_REPORT.write_text(generate_full_report(out))
            print("adaptive_beats_vanilla="
                  f"{out['verdict'].get('adaptive_beats_vanilla')}")
        elif pilot:
            pilot_out = dict(pilot)
            pilot_out["pair_pairs"] = paired_comparison(
                pilot_out.get("records") or [])
            OUT_PILOT_JSON.write_text(
                json.dumps(pilot_out, indent=2, default=str))
            OUT_PILOT_REPORT.write_text(generate_pilot_report(pilot_out))
            print("pilot gates passed="
                  f"{pilot_out.get('pilot_gates', {}).get('passed')}")
        return

    if not args.skip_model_check:
        check_models(args.model, args.embedding_model, args.endpoint,
                     use_embeddings=True)

    fixtures = validate_all_fixtures()

    if args.validate:
        OUT_PILOT_JSON.write_text(json.dumps(
            {"experiment": "e26_discriminative_coding_benchmark",
             "mode": "validate", "fixture_gates": fixtures}, indent=2))
        print("fixture gates passed=", fixtures["passed"])
        for k, v in fixtures["variants"].items():
            print(k, "->", "PASS" if v["passed"] else f"FAIL {v['failing']}")
        return

    embed_fn = _resolve_embed(args.embedding_model, args.endpoint)
    coder = cb.OllamaCoder(args.model, args.endpoint)

    if args.pilot:
        mode = "pilot"
        variants, seeds, budgets = PILOT_VARIANTS, PILOT_SEEDS, PILOT_BUDGETS
        out_json, out_report = OUT_PILOT_JSON, OUT_PILOT_REPORT
    else:
        mode = "full"
        variants, seeds, budgets = FULL_VARIANTS, FULL_SEEDS, FULL_BUDGETS
        out_json, out_report = OUT_JSON, OUT_REPORT
        if not OUT_PILOT_JSON.exists():
            parser.error("full grid requires the pilot artifact; run --pilot")
        pilot_gates = (json.loads(OUT_PILOT_JSON.read_text())
                       .get("pilot_gates") or {})
        if not pilot_gates.get("passed"):
            parser.error("pilot gates did not pass; full grid is not legal")

    records: List[Dict] = []
    if not args.force:
        path = out_json if out_json.exists() else OUT_PILOT_JSON
        if path.exists():
            existing = json.loads(path.read_text()) or {}
            records = [r for r in existing.get("records", [])
                       if _run_key(r["group"], r["variant"], r["seed"],
                                   r["historical_budget"], r["method"])
                       in set(_expected_keys(mode=mode))]

    grid_cells = (len(GROUPS) * len(variants) * len(seeds) * len(budgets)
                  * len(PILOT_METHODS)) if mode == "pilot" else \
        (len(GROUPS) * len(variants) * len(seeds) * len(budgets)
         * len(FULL_METHODS))
    have = {_run_key(r["group"], r["variant"], r["seed"],
                     r["historical_budget"], r["method"]) for r in records}
    todo = [k for k in _expected_keys(mode=mode) if args.force or k not in have]

    config = {
        "mode": mode, "groups": GROUPS, "variants": variants,
        "seeds": seeds, "budgets": budgets,
        "methods": PILOT_METHODS if mode == "pilot" else FULL_METHODS,
        "grid_cells": grid_cells,
        "model": args.model, "endpoint": args.endpoint,
        "embedding_model": args.embedding_model,
        "temperature": cb.CODING_TEMPERATURE,
        "tokenizer": cb.TOKENIZER_NAME,
        "ingestion": "oracle_pre_extracted",
        "adaptive": {"retention_mode": "dual_score",
                     "protect_corrections": True},
        "generated_at": time.time(),
    }

    print("=" * 64)
    print(f"E26 Discriminative Coding Benchmark ({mode}) - {len(todo)} cells")
    print("=" * 64)

    def persist(partial: bool = False) -> Dict:
        payload = {
            "experiment": "e26_discriminative_coding_benchmark",
            "partial": partial,
            "mode": mode,
            "config": config,
            "fixture_gates": fixtures,
            "records": records,
        }
        if mode == "pilot":
            payload["pilot_gates"] = compute_pilot_gates(records)
            payload["aggregate"] = aggregate(records)
        else:
            payload["pilot_gates"] = pilot_gates
        out_json.write_text(json.dumps(payload, indent=2, default=str))
        return payload

    gold_checks: List[Dict] = []
    if mode == "pilot":
        print("[gold] validating gold patches with StaticCoder ...")
        for gid in GROUPS:
            for variant in PILOT_VARIANTS:
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
        if mode == "pilot":
            gates = compute_pilot_gates(records)
            if not gates["passed"]:
                print("[STOP] pilot gates failed; full grid is NOT legal.")
                print(json.dumps(gates, indent=2, default=str))
                persist(partial=False)
                OUT_PILOT_REPORT.write_text(generate_pilot_report(
                    persist(partial=False)))
                sys.exit(1)

    payload = persist(partial=False)
    if mode == "pilot":
        payload["paired"] = paired_comparison(records)
        payload["verdict"] = {"pilot_gates_passed": payload["pilot_gates"]["passed"],
                              "note": "full verdict requires the full grid"}
        payload["gold_checks"] = gold_checks
        OUT_PILOT_JSON.write_text(json.dumps(payload, indent=2, default=str))
        OUT_PILOT_REPORT.write_text(generate_pilot_report(payload))
        print("pilot complete; gates passed=", payload["pilot_gates"]["passed"])
        return

    full_grid = grid_completeness(records, mode="full")
    paired = paired_comparison(records)
    verdict = compute_verdict(pilot_gates=pilot_gates,
                              full_grid=full_grid, paired=paired,
                              fixture_gates=fixtures)
    payload["full_grid"] = full_grid
    payload["paired"] = paired
    payload["verdict"] = verdict
    payload["gold_checks"] = gold_checks
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    OUT_REPORT.write_text(generate_full_report(payload))
    print("full grid complete; paired CI lower bound="
          f"{verdict.get('ci_lower_bound')}")
    print("adaptive_beats_vanilla=", verdict["adaptive_beats_vanilla"])
    return payload


if __name__ == "__main__":
    main()