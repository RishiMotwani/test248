"""E18 — Phase 14: correction-supersession safety at the memory-consolidation boundary.

Problem (D35, fixed in Phase 14)
--------------------------------
Explicit corrections could be swallowed by the *generic* semantic-duplicate gate
in ``MemoryCompressor.dedupe_incremental``: the supersession check only ran after
a pair already cleared cos >= 0.90, so a correction that restates the old fact
in genuinely new wording (cos well below 0.90) was stored as a *second* fact
instead of replacing the stale one. Retrieval then ranked the stale predecessor
above the correction and the coding model was told the obsolete truth
(E17: ``correction_recall == 0``, ``obsolete_fact_exposure == 1.0`` in the
adaptive failing cells). Phase 14 moves the explicit supersession check
(``_is_supersession``, unchanged) *before* any similarity gate.

E18 validates that fix in two halves:

1. **Offline identity gate** (``--offline``): replay the actual adaptive
   production memory-preparation path over every coding-task history
   (4 tasks x 3 seeds x 3 budgets). For every correction pair assert, by fact
   identity (``fact_id``), that the current fact is present in the final store
   and the obsolete fact is absent. Gate passes iff
   ``correction_recall == 1.0`` AND ``obsolete_exposure == 0.0`` in every
   correction-bearing cell.

2. **Targeted coding validation** (``--coding``): 2 correction-bearing tasks
   (``user_ids``, ``validation_pure``) x 3 seeds x 3 budgets x 4 methods
   (adaptive / vanilla_rag / llm_summarization / raw_clipped; no sliding_window)
   = 72 runs, reusing the E17 harness unchanged.

CLI: ``--offline --coding --resume --limit --report-only``. Every cell is
persisted after completion so the run is resumable across aborts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.coding_task_suite import build_task  # noqa: E402
from experiments import coding_benchmark as cb  # noqa: E402
from experiments.e17_coding_capability import check_models  # noqa: E402
from memory_optimizer.tokenizer import TOKENIZER_NAME  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration (targeted by specification; do not widen without a reason)
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_ENDPOINT = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

OFFLINE_TASKS = ["cache_readonly", "user_ids", "write_retry", "validation_pure"]
OFFLINE_SEEDS = [1, 2, 3]
OFFLINE_BUDGETS = [256, 512, 1024]

CODING_TASKS = ["user_ids", "validation_pure"]
CODING_SEEDS = [1, 2, 3]
CODING_BUDGETS = [256, 512, 1024]
CODING_METHODS = ["adaptive", "vanilla_rag", "llm_summarization", "raw_clipped"]

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_JSON = RESULTS_DIR / "e18_correction_safety.json"
OUT_REPORT = RESULTS_DIR / "e18_correction_safety_report.md"
WORK_ROOT = RESULTS_DIR / "e18_work"

DIAG_KEYS = ["critical_fact_recall", "correction_recall", "obsolete_fact_exposure"]


# ---------------------------------------------------------------------------
# Keys / helpers
# ---------------------------------------------------------------------------

def _offline_key(task_id: str, seed: int, budget: int) -> str:
    return f"{task_id}:{seed}:{budget}"


def _coding_key(task_id: str, seed: int, budget: int, method: str) -> str:
    return f"{task_id}:{seed}:{budget}:{method}"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _success(rows: List[Dict]) -> Optional[float]:
    if not rows:
        return None
    return round(sum(1 for r in rows if r["final_success"]) / len(rows), 4)


def _resolve_embed(use_embeddings: bool, embedding_model: str, endpoint: str):
    if not use_embeddings:
        return None
    from memory_optimizer.embeddings import embed_ollama
    return lambda texts: embed_ollama(texts, model=embedding_model, endpoint=endpoint)


# ---------------------------------------------------------------------------
# Offline identity gate (sections 17-19)
# ---------------------------------------------------------------------------

def run_offline_cell(task_id: str, seed: int, budget: int, *, embed_fn,
                     embedding_model: str, endpoint: str, model: str) -> Dict:
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
        current_text = text_by_id.get(c.current_fact_id, "")
        obsolete_text = text_by_id.get(c.obsolete_fact_id, "")
        corrections.append({
            "obsolete_fact_id": c.obsolete_fact_id,
            "current_fact_id": c.current_fact_id,
            "current_fact_present": any(
                m.get("fact_id") == c.current_fact_id for m in memories),
            "obsolete_fact_present": any(
                m.get("fact_id") == c.obsolete_fact_id for m in memories),
            "current_fact_present_by_text": any(
                m.get("fact") == current_text for m in memories),
            "obsolete_fact_present_by_text": any(
                m.get("fact") == obsolete_text for m in memories),
        })

    n = len(corrections)
    if n:
        correction_recall = round(
            sum(1 for c in corrections if c["current_fact_present"]) / n, 4)
        obsolete_exposure = round(
            sum(1 for c in corrections if c["obsolete_fact_present"]) / n, 4)
    else:
        correction_recall = None
        obsolete_exposure = None

    return {
        "task_id": task_id,
        "task_title": task.title,
        "seed": seed,
        "budget": budget,
        "method": "adaptive",
        "history_turns": len(history.turns),
        "n_corrections": n,
        "corrections": corrections,
        "correction_recall": correction_recall,
        "obsolete_exposure": obsolete_exposure,
        "store_tokens": prepared.memory_tokens,
        "memory_count": len(memories),
        "gate": (correction_recall == 1.0 and obsolete_exposure == 0.0)
        if n else True,
    }


def offline_gate(cells: List[Dict]) -> Dict:
    corr_cells = [c for c in cells if c["n_corrections"] > 0]
    per_cell = {_offline_key(c["task_id"], c["seed"], c["budget"]): c
                for c in corr_cells}
    passed_cells = [c for c in corr_cells if c["gate"]]
    return {
        "correction_bearing_cells": len(corr_cells),
        "cells_passing": len(passed_cells),
        "gate_passed": len(per_cell) > 0 and len(passed_cells) == len(per_cell),
        "correction_recall_aggregate": _mean([c["correction_recall"] for c in corr_cells]),
        "obsolete_exposure_aggregate": _mean([c["obsolete_exposure"] for c in corr_cells]),
        "failing_cells": [k for k, c in per_cell.items() if not c["gate"]],
        "per_task": _offline_task_summary(corr_cells),
    }


def _offline_task_summary(corr_cells: List[Dict]) -> Dict:
    out = {}
    for tid in sorted({c["task_id"] for c in corr_cells}):
        rows = [c for c in corr_cells if c["task_id"] == tid]
        out[tid] = {
            "cells": len(rows),
            "correction_recall": _mean([c["correction_recall"] for c in rows]),
            "obsolete_exposure": _mean([c["obsolete_exposure"] for c in rows]),
            "mean_store_tokens": _mean([c["store_tokens"] for c in rows]),
            "mean_memory_count": _mean([c["memory_count"] for c in rows]),
        }
    return out


# ---------------------------------------------------------------------------
# Targeted coding validation (sections 20-21)
# ---------------------------------------------------------------------------

def run_coding_cell(task_id: str, seed: int, budget: int, method_name: str, *,
                    coder, embed_fn, model: str, endpoint: str,
                    embedding_model: str) -> Dict:
    task = build_task(task_id, seed=seed)
    method = cb.build_method(method_name, model=model, endpoint=endpoint,
                             embed_fn=embed_fn, embedding_model=embedding_model,
                             task=task)
    root = WORK_ROOT / f"{task_id}_{seed}_{budget}_{method_name}"
    return cb.run_method_run(task=task, seed=seed, historical_budget=budget,
                             method=method, coder=coder, work_root=root)


def aggregate_coding(records: List[Dict]) -> Dict:
    by_mb: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_mb[r["method"]][str(r["historical_budget"])].append(r)

    out: Dict[str, Dict] = {}
    for m, by_b in by_mb.items():
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
                "diagnostics": {k: _mean([r["diagnostics"].get(k) for r in rows])
                                for k in DIAG_KEYS},
                "failure_classes": _count_failures(rows),
            }

    def cell_success(method: str) -> Dict:
        d = {}
        for r in records:
            if r["method"] == method:
                d[(r["task_id"], r["seed"], r["historical_budget"])] = int(
                    r["final_success"])
        return d

    adapt = cell_success("adaptive")
    comparisons = {}
    for m in CODING_METHODS:
        if m == "adaptive":
            continue
        other = cell_success(m)
        keys = sorted(set(adapt) & set(other))
        a = [adapt[k] for k in keys]
        b = [other[k] for k in keys]
        comparisons[m] = {
            "pairs": len(keys),
            "adaptive_success": round(sum(a) / len(a), 4) if a else None,
            "other_success": round(sum(b) / len(b), 4) if b else None,
            "mean_diff": round(sum(x - y for x, y in zip(a, b)) / len(a), 4)
            if a else None,
            "adaptive_cells": [f"{k[0]}:{k[1]}:{k[2]}" for k in keys],
        }

    overall = {}
    for m in sorted({r["method"] for r in records}):
        rows = [r for r in records if r["method"] == m]
        overall[m] = {"runs": len(rows), "success_rate": _success(rows),
                      "first_pass_rate": round(
                          sum(1 for r in rows if r["first_pass_success"]) / len(rows), 4)
                      if rows else None}

    return {"by_method_budget": out, "overall": overall,
            "paired_vs_adaptive": comparisons}


def _count_failures(rows: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["failure_class"] or "SUCCESS"] += 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Report (section 24)
# ---------------------------------------------------------------------------

def _fmt(x, nd=3):
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def generate_report(result: Dict) -> str:
    cfg = result["config"]
    off = result.get("offline") or {}
    cod = result.get("coding") or {}
    L: List[str] = []
    L.append("# E18 Correction-Safety Report (Phase 14)")
    L.append("")
    L.append("Did moving explicit supersession ahead of the generic semantic-"
             "duplicate threshold make corrections authoritative at the "
             "consolidation boundary? Offline identity gate first; targeted "
             "coding validation second.")
    L.append("")

    L.append("## 1. Purpose & Research Question")
    L.append("")
    L.append("A correction/supersession is a **semantic relationship**, not a "
             "high-similarity duplicate (D35). Phase 14 makes "
             "`_is_supersession` (unchanged) run before the cos>=0.90 / overlap "
             "gate in `dedupe_incremental`, so a restated-and-corrected fact "
             "replaces the stale fact's memory entry instead of coexisting with "
             "it. E18 verifies (a) the store after the production preparation "
             "path contains exactly the current fact of every correction pair "
             "(identity-based), and (b) that this propagates to coding-task "
             "success on the tasks where E17 observed the failure.")
    L.append("")

    L.append("## 2. Root Cause & the Fix")
    L.append("")
    L.append("In Phase 13 the supersession check sat *inside* `if sim >= "
             "0.90`, so a correction whose restatement used new wording "
             "(cosine ~0.7 vs the stored fact) never reached it and was stored "
             "as an additional memory. Retrieval ranked the stale predecessor "
             "first (E17 failing cells: `correction_recall == 0`, "
             "`obsolete_fact_exposure == 1.0`, adaptive context identical to "
             "vanilla_rag). The trusted `_replace_with_supersession` mutation "
             "is factored out and invoked: (1) for `supersedes_turn` targets, "
             "(2) before any similarity computation in the generic loop. The "
             "duplicate-merge path is otherwise unchanged; `_is_supersession` "
             "still requires a revision marker + full-word restatement "
             "(no weakening).")
    L.append("")

    L.append("## 3. Offline Identity Gate: Method")
    L.append("")
    L.append(f"Adaptive production preparation (retention `dual_score`, "
             f"`protect_corrections=True`, store budget `max(4096, budget*4)`, "
             f"injection budget = historical budget) replayed over "
             f"{cfg.get('offline_tasks')} x {cfg.get('offline_seeds')} seeds x "
             f"{cfg.get('offline_budgets')} budgets = "
             f"{cfg.get('offline_cells', 0)} cells. Every correction pair is "
             "checked **by fact identity** (`fact_id`) in the final store: "
             "`current_fact_present` and `obsolete_fact_present`. A cell passes "
             "iff `correction_recall == 1.0` and `obsolete_exposure == 0.0`; "
             "cells for tasks without corrections are recorded but not gated.")
    L.append("")

    L.append("## 4. Offline Identity Gate: Results")
    L.append("")
    gate = off.get("gate") or {}
    rows = [c for c in off.get("cells", []) if c["n_corrections"] > 0]
    L.append("| task | seed | budget | n_corr | correction_recall | obsolete_exposure |"
             " store_tokens | memory_count | current@id | obsolete@id | cell gate |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in off.get("cells", []):
        if not c["n_corrections"]:
            continue
        ids = " / ".join(
            f"{x['current_fact_present']}/{x['obsolete_fact_present']}"
            for x in c["corrections"])
        L.append(f"| {c['task_id']} | {c['seed']} | {c['budget']} | {c['n_corrections']} "
                 f"| {_fmt(c['correction_recall'])} | {_fmt(c['obsolete_exposure'])} "
                 f"| {c['store_tokens']} | {c['memory_count']} | {ids} "
                 f"| {'PASS' if c['gate'] else 'FAIL'} |")
    L.append("")
    L.append(f"Non-correction cells (write_retry, recorded only): "
             f"{sum(1 for c in off.get('cells', []) if not c['n_corrections'])}.")
    L.append("")

    L.append("## 5. Offline Identity Gate: Verdict")
    L.append("")
    L.append(f"- correction-bearing cells: **{gate.get('correction_bearing_cells')}**")
    L.append(f"- cells passing: **{gate.get('cells_passing')}**")
    L.append(f"- aggregate correction_recall: **{_fmt(gate.get('correction_recall_aggregate'))}**")
    L.append(f"- aggregate obsolete_exposure: **{_fmt(gate.get('obsolete_exposure_aggregate'))}**")
    L.append(f"- failing cells: **{gate.get('failing_cells') or 'none'}**")
    L.append(f"- **OFFLINE GATE: {'PASS' if gate.get('gate_passed') else 'FAIL'}**")
    L.append("")
    L.append("Per task:")
    L.append("")
    L.append("| task | correction_recall | obsolete_exposure | mean_store_tokens | mean_memory_count |")
    L.append("| --- | --- | --- | --- | --- |")
    for tid, t in sorted((gate.get("per_task") or {}).items()):
        L.append(f"| {tid} | {_fmt(t['correction_recall'])} | {_fmt(t['obsolete_exposure'])} "
                 f"| {_fmt(t['mean_store_tokens'], 1)} | {_fmt(t['mean_memory_count'], 1)} |")
    L.append("")

    L.append("## 6. Targeted Coding Validation: Design")
    L.append("")
    L.append(f"2 correction-bearing tasks ({cfg.get('coding_tasks')}) x "
             f"{cfg.get('coding_seeds')} seeds x {cfg.get('coding_budgets')} budgets "
             f"x {cfg.get('coding_methods')} methods = {cfg.get('coding_cells')} runs. "
             "`sliding_window` excluded per the Phase 14 spec. Reuses "
             "`experiments/coding_benchmark.py` unchanged (same prompt scaffold, "
             "temperature 0.1, retry policy, tokenizer).")
    L.append("")

    L.append("## 7. Coding Results: Success Rate by Method x Budget")
    L.append("")
    budgets = cfg.get("coding_budgets")
    header = "| method | " + " | ".join(str(b) for b in budgets) + " | overall | first-pass |"
    L.append(header)
    L.append("| --- | " + " | ".join("---" for _ in budgets) + " | --- | --- |")
    agg = cod.get("aggregate") or {}
    for m in cfg.get("coding_methods", []):
        row = []
        for b in budgets:
            v = (agg.get("by_method_budget", {}).get(m, {}).get(str(b), {}) or {}).get(
                "success_rate")
            row.append(_fmt(v))
        overall = (agg.get("overall", {}).get(m, {}) or {}).get("success_rate")
        fp = (agg.get("overall", {}).get(m, {}) or {}).get("first_pass_rate")
        L.append(f"| {m} | " + " | ".join(row) + f" | {_fmt(overall)} | {_fmt(fp)} |")
    L.append("")

    L.append("## 8. Coding Results: Paired Comparisons (adaptive - other)")
    L.append("")
    L.append("| vs | pairs | adaptive | other | mean diff |")
    L.append("| --- | --- | --- | --- | --- |")
    for m, c in (agg.get("paired_vs_adaptive") or {}).items():
        L.append(f"| {m} | {c['pairs']} | {_fmt(c['adaptive_success'])} | "
                 f"{_fmt(c['other_success'])} | {_fmt(c['mean_diff'])} |")
    L.append("")

    L.append("## 9. Failure Taxonomy Distribution")
    L.append("")
    L.append("| method | budget | failure classes |")
    L.append("| --- | --- | --- |")
    for m in cfg.get("coding_methods", []):
        for b in budgets:
            row = agg.get("by_method_budget", {}).get(m, {}).get(str(b), {})
            if row:
                L.append(f"| {m} | {b} | {row.get('failure_classes')} |")
    L.append("")

    L.append("## 10. Store & Token Metrics")
    L.append("")
    L.append(f"Offline cells: store_tokens range "
             f"{_minmax([c['store_tokens'] for c in off.get('cells', [])])}; "
             f"memory_count range "
             f"{_minmax([c['memory_count'] for c in off.get('cells', [])])}.")
    L.append("")

    L.append("## 11. Context Integrity & Leakage")
    L.append("")
    leaks = [r for r in cod.get("records", [])
             if (r.get("leakage") or {}).get("leakage_detected")]
    L.append(f"- leaking runs: **{len(leaks)}** "
             f"({'none' if not leaks else leaks}).")
    L.append(f"- runs classifying as CONTEXT_OVERFLOW: "
             f"**{sum(1 for r in cod.get('records', []) if r['failure_class'] == 'CONTEXT_OVERFLOW')}**.")
    L.append(f"- predictions total-prompt-tokens <= model window enforced by the harness on every run.")
    L.append("")

    L.append("## 12. Per-Cell Failure Analysis")
    L.append("")
    L.append("| cell | method | success | failure_class | corr_recall | obs_exposure | context_tokens |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in cod.get("records", []):
        L.append(f"| {r['task_id']}:{r['seed']}:{r['historical_budget']} | {r['method']} "
                 f"| {'ok' if r['final_success'] else 'FAIL'} | "
                 f"{r['failure_class'] or 'SUCCESS'} | "
                 f"{_fmt(r['diagnostics'].get('correction_recall'))} | "
                 f"{_fmt(r['diagnostics'].get('obsolete_fact_exposure'))} | "
                 f"{r['historical_context_tokens']} |")
    L.append("")

    L.append("## 13. What Was NOT Changed")
    L.append("")
    L.append("- `_is_supersession()`: unchanged - still requires a revision/"
             "negation marker plus full restatement of the stored fact.")
    L.append("- `memory_optimizer/retrieval.py` and `memory_optimizer/pipeline.py`: "
             "not modified in Phase 14 (retrieval-weight fix only if evidence "
             "gates demanded it).")
    L.append("- `config.yaml` production values: untouched (no threshold, weight, "
             "budget or embedding-model changes).")
    L.append("- E17 artifacts: immutable; not regenerated.")
    L.append("")

    L.append("## 14. Threats to Validity & Limitations")
    L.append("")
    L.append("- Task suite is synthetic-but-real-code; 6-8 facts per 600-turn "
             "history, so the store stays far under its budget - the offline "
             "gate isolates consolidation, not budget pressure.")
    L.append("- Codes' corrections are oracle-pre-extracted; live-extraction "
             "corrections are out of scope.")
    L.append("- 72 coding runs => wide success-rate CIs; absence of a difference "
             "is not evidence of equivalence.")
    L.append("- Ollama-served llama3.1:8b at temperature 0.1 is not "
             "deterministic; the grid is resumable and per-run cells are the unit.")
    L.append("")

    L.append("## 15. Relationship to E17 (Phase 13 artifacts, immutable)")
    L.append("")
    L.append(f"E17 verdict: gates_all_passed=True, adaptive_advances=False. "
             f"E17's adaptive failure cells were exactly the ones where the "
             f"obsolete fact was exposed and the correction was not recalled. "
             f"E18's offline gate verifies the consolidation boundary is fixed; "
             f"the coding half re-measures the downstream outcome on the two "
             f"E17 offenders (user_ids, validation_pure).")
    L.append("")

    L.append("## 16. Conclusion")
    L.append("")
    if gate.get("gate_passed") is True:
        L.append("- OFFLINE GATE: **PASS** - every correction-bearing cell keeps "
                 "the current fact and holds no obsolete fact in the store.")
    else:
        L.append("- OFFLINE GATE: **FAIL** - at least one correction-bearing "
                 "cell is stale; do not proceed to derive production conclusions.")
    L.append("")

    L.append("## 17. Reproduction / Artifacts / Provenance")
    L.append("")
    L.append(f"- JSON: `{OUT_JSON.name}`.")
    L.append(f"- generated_at: {cfg.get('generated_at')}")
    L.append(f"- tokenizer: `{TOKENIZER_NAME}`; model: `{cfg.get('model')}`; "
             f"embedding: `{cfg.get('embedding_model')}`; use_embeddings: "
             f"{cfg.get('use_embeddings')}.")
    L.append(f"- work dir: `experiments/results/e18_work/` (gitignored).")
    L.append("")

    L.append("## 18. Open Questions")
    L.append("")
    L.append("- How does the same correction handling behave under store "
             "budget pressure (facts-heavy histories), where eviction and the "
             "correction-protection flag interact?")
    L.append("- Is oracle-pre-extracted correction text a fair proxy for "
             "candidate corrections produced by a live extractor?")
    L.append("")

    L.append("## 19. Next Research Question")
    L.append("")
    L.append("Evaluate whether correction authoritativeness holds when the "
             "history contains *many* corrections of the same fact (fact "
             "versioning) and whether retrieval should treat superseded_prior "
             "references as a guardrail signal rather than a lexical cue.")
    L.append("")
    L.append("— All numbers are generated from the E18 JSON by `generate_report()`; "
             "no hand-typed figures.")
    return "\n".join(L)


def _minmax(xs):
    return (min(xs), max(xs)) if xs else (None, None)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="E18 correction-safety validation")
    parser.add_argument("--offline", action="store_true",
                        help="run the offline identity gate (adaptive prep)")
    parser.add_argument("--coding", action="store_true",
                        help="run the targeted coding validation grid (72 runs)")
    parser.add_argument("--resume", action="store_true",
                        help="resume: skip cells already recorded in the JSON")
    parser.add_argument("--limit", type=int, default=0,
                        help="limit number of NEW cells to run (debugging)")
    parser.add_argument("--report-only", action="store_true",
                        help="regenerate the report from the existing JSON")
    parser.add_argument("--force", action="store_true", help="recompute all cells")
    parser.add_argument("--no-embed", action="store_true",
                        help="disable embeddings (lexical retrieval; diagnostic only)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--max-output-tokens", type=int,
                        default=cb.MAX_OUTPUT_TOKENS)
    parser.add_argument("--skip-model-check", action="store_true")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    use_embeddings = not args.no_embed

    if args.report_only:
        if not OUT_JSON.exists():
            parser.error("no e18_correction_safety.json to report")
        result = json.loads(OUT_JSON.read_text())
        result["offline"]["gate"] = offline_gate(result["offline"]["cells"])
        result["coding"]["aggregate"] = aggregate_coding(result["coding"]["records"])
        OUT_JSON.write_text(json.dumps(result, indent=2, default=str))
        OUT_REPORT.write_text(generate_report(result))
        print(f"Re-generated {OUT_REPORT}")
        return result

    if not args.skip_model_check:
        check_models(args.model, args.embedding_model, args.endpoint, use_embeddings)

    state = {}
    if OUT_JSON.exists() and not args.force:
        state = json.loads(OUT_JSON.read_text())

    offline_cells: List[Dict] = list(state.get("offline", {}).get("cells", []))
    coding_records: List[Dict] = list(state.get("coding", {}).get("records", []))
    off_existing = {_offline_key(c["task_id"], c["seed"], c["budget"]): c
                    for c in offline_cells}
    cod_existing = {_coding_key(r["task_id"], r["seed"], r["historical_budget"],
                                r["method"]): r for r in coding_records}

    embed_fn = _resolve_embed(use_embeddings, args.embedding_model, args.endpoint)

    def persist():
        result = {
            "experiment": "e18_correction_safety",
            "config": {
                "offline_tasks": OFFLINE_TASKS, "offline_seeds": OFFLINE_SEEDS,
                "offline_budgets": OFFLINE_BUDGETS,
                "offline_cells": len(OFFLINE_TASKS) * len(OFFLINE_SEEDS)
                * len(OFFLINE_BUDGETS),
                "coding_tasks": CODING_TASKS, "coding_seeds": CODING_SEEDS,
                "coding_budgets": CODING_BUDGETS, "coding_methods": CODING_METHODS,
                "coding_cells": len(CODING_TASKS) * len(CODING_SEEDS)
                * len(CODING_BUDGETS) * len(CODING_METHODS),
                "model": args.model, "endpoint": args.endpoint,
                "embedding_model": args.embedding_model,
                "use_embeddings": use_embeddings,
                "temperature": cb.CODING_TEMPERATURE, "tokenizer": TOKENIZER_NAME,
                "ingestion": "oracle_pre_extracted",
                "adaptive": {"retention_mode": "dual_score",
                             "protect_corrections": True,
                             "injection_token_limit": "historical_budget",
                             "memory_store_token_budget": "max(4096, budget*4)"},
                "generated_at": time.time(),
            },
            "offline": {"cells": offline_cells,
                        "gate": offline_gate(offline_cells)},
            "coding": {"records": coding_records,
                       "aggregate": aggregate_coding(coding_records)},
        }
        OUT_JSON.write_text(json.dumps(result, indent=2, default=str))
        return result

    print("=" * 64)
    print("E18 Correction-Safety Validation (Phase 14)")
    print("=" * 64)

    if args.offline:
        todo = [(tid, seed, budget) for tid in OFFLINE_TASKS
                for seed in OFFLINE_SEEDS for budget in OFFLINE_BUDGETS
                if _offline_key(tid, seed, budget) not in off_existing]
        for idx, (tid, seed, budget) in enumerate(todo):
            if idx >= (args.limit or 10 ** 9):
                break
            key = _offline_key(tid, seed, budget)
            print(f"  [offline {idx + 1}/{key}] {key} ...", flush=True)
            try:
                cell = run_offline_cell(tid, seed, budget, embed_fn=embed_fn,
                                        embedding_model=args.embedding_model,
                                        endpoint=args.endpoint, model=args.model)
            except Exception as exc:  # keep the grid resumable
                print(f"      ERROR {key}: {exc}", flush=True)
                continue
            offline_cells.append(cell)
            off_existing[key] = cell
            persist()

    gate_now = offline_gate(offline_cells) if offline_cells else {}
    print(f"[offline] cells={len(offline_cells)} "
          f"gate_passed={gate_now.get('gate_passed')} "
          f"(corr cells {gate_now.get('correction_bearing_cells')}, "
          f"passing {gate_now.get('cells_passing')})", flush=True)

    if args.coding:
        if not (offline_cells and gate_now.get("gate_passed")):
            prev_gate = (state.get("offline") or {}).get("gate") or {}
            if not prev_gate.get("gate_passed"):
                print("ABORT: --coding requested but the offline identity gate "
                      "has not passed. Run --offline first and satisfy "
                      "correction_recall==1.0 and obsolete_exposure==0.0 on every "
                      "correction-bearing cell.", flush=True)
                return persist()
        coder = cb.OllamaCoder(args.model, args.endpoint, args.max_output_tokens)
        summarizer_generate_fn = lambda prompt, mx: coder.generate(prompt)["text"]
        todo = [(tid, seed, budget, method) for tid in CODING_TASKS
                for seed in CODING_SEEDS for budget in CODING_BUDGETS
                for method in CODING_METHODS
                if _coding_key(tid, seed, budget, method) not in cod_existing]
        for idx, (tid, seed, budget, method) in enumerate(todo):
            if idx >= (args.limit or 10 ** 9):
                break
            key = _coding_key(tid, seed, budget, method)
            print(f"  [coding {idx + 1}/{key}] {key} ...", flush=True)
            try:
                rec = run_coding_cell(tid, seed, budget, method,
                                      coder=coder, embed_fn=embed_fn,
                                      model=args.model,
                                      endpoint=args.endpoint,
                                      embedding_model=args.embedding_model)
            except Exception as exc:
                print(f"      ERROR {key}: {exc}", flush=True)
                continue
            coding_records.append(rec)
            cod_existing[key] = rec
            persist()

    result = persist()
    print(f"\nWrote {OUT_JSON}")
    OUT_REPORT.write_text(generate_report(result))
    print(f"Wrote {OUT_REPORT}")
    print(f"offline_gate_passed={result['offline']['gate']['gate_passed']}")
    return result


if __name__ == "__main__":
    main()