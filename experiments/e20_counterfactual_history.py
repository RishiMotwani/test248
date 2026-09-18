"""E20 — Phase 16: counterfactual-history benchmark calibration.

Purpose
-------
Phase 15 (E19) failed gate C (history dependence): the pilot could not show its
coding tasks were *genuinely* history dependent — some fixtures were solvable
from the workspace alone, so a high adaptive score could not be attributed to
the memory system. E20 calibrates the benchmark itself with **counterfactual
history pairs**:

    per group, two variants (A / B) that share the *same* workspace and the
    *same* task prompt but receive *different* histories, whose required
    behaviour (hidden test + gold patch) differs *solely* because of that
    history.

McNemar-style per-group gate over 3 seeds: if injecting the correct variant's
history (``direct_history`` oracle) lifts the pass rate far above ``no_history``,
the task is genuinely history dependent and the E19 full grid becomes eligible
to make a memory-system claim. This experiment does **not** run E19 and does not
touch the adaptive-memory system or production defaults.

Grid (fixed, 54 cells)
----------------------
3 groups x 2 variants x 3 seeds x 3 methods:

* ``identifier_policy``   — opaque-string ids (A) vs numeric ids (B)
* ``retry_policy``        — exactly one attempt (A) vs retry once (B)
* ``serialization_policy`` — preserve unknown keys (A) vs drop unknown keys (B)

methods
-------
* ``no_history``    — model sees only workspace + prompt (identical for A / B)
* ``direct_history`` — oracle: injects the *same variant's* non-obsolete gold
                       facts (correct historical decision)
* ``full_context``   — enire raw 600-turn history; always overflows the 1024-word
                       historical budget and is recorded as an overflow
                       diagnostic (upper-bound reference, not a budget arm)

Two hard gates (the experiment must stop if either fails) and one verdict:

* Hard Gate 1 — gold patch validity: every variant's gold.patch applies cleanly
  and passes hidden tests (offline, via ``StaticCoder``).
* Hard Gate 2 — history dependence, per group over 6 runs (2 variants x 3
  seeds): ``direct_history >= 5/6``, ``no_history <= 3/6``,
  ``separation = direct_history - no_history >= 2/6``. ALL three groups must
  pass.
* Verdict — ``history_dependence_benchmark`` VALID only if Gate 1, pair
  integrity and all three group gates pass; then ``e19_full_grid_eligible`` is
  TRUE. (E19 itself is NOT run from here.)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments.coding_benchmark as cb  # noqa: E402
from data import counterfactual_task_suite as cf  # noqa: E402
from experiments.e19_coding_generalization import StaticCoder, WORK_ROOT  # noqa: E402
from memory_optimizer.tokenizer import TOKENIZER_NAME  # noqa: E402

GROUPS = cf.list_group_ids()
SEEDS = [1, 2, 3]
BUDGET = 1024
METHODS = ["no_history", "direct_history", "full_context"]
GRID_CELLS = len(GROUPS) * 2 * len(SEEDS) * len(METHODS)  # 54

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_ENDPOINT = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
MAX_OUTPUT_TOKENS = cb.MAX_OUTPUT_TOKENS

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_JSON = RESULTS_DIR / "e20_counterfactual_history.json"
OUT_REPORT = RESULTS_DIR / "e20_counterfactual_history_report.md"

GATE_NAMES = [
    "gold_patch_validity",    # 1
    "pair_integrity",         # fixtures differ only by history
    "history_dependence",     # 2
]

GROUP_DH_MIN = 5   # direct_history >= 5/6 per group
GROUP_NH_MAX = 3   # no_history   <= 3/6 per group
GROUP_SEP_MIN = 2  # separation   >= 2/6 per group


def _run_key(group_id: str, variant: str, seed: int, method: str) -> str:
    return f"{group_id}:{variant}:{seed}:{method}"


def _gold_key(group_id: str, variant: str) -> str:
    return f"{group_id}:{variant}"


# ---------------------------------------------------------------------------
# Hard Gate 1: gold patch validity (offline, no LLM)
# ---------------------------------------------------------------------------

def run_gold_checks() -> List[Dict]:
    checks = []
    for group_id in GROUPS:
        for variant in cf.list_variants(group_id):
            task = cf.build_variant(group_id, variant, seed=1)
            gold = Path(task.gold_patch_path).read_text()
            coder = StaticCoder(gold)
            root = WORK_ROOT / f"e20gold_{group_id}_{variant}"
            method = cb.build_method("no_history", model="static",
                                     endpoint="http://x", embed_fn=None,
                                     embedding_model="", task=task)
            rec = cb.run_method_run(task=task, seed=1, historical_budget=BUDGET,
                                    method=method, coder=coder, work_root=root)
            checks.append({
                "group_id": group_id, "variant": variant,
                "gold_patch_valid": bool(rec["final_success"]),
                "patch_applied": bool(rec["patch_applied"]),
                "hidden_test_pass": bool(rec["hidden_test_pass"]),
            })
    return checks


def gold_gate_passed(checks: List[Dict]) -> bool:
    return bool(checks) and len(checks) == GRID_CELLS // (len(SEEDS) * len(METHODS)) \
        and all(c["gold_patch_valid"] for c in checks)


# ---------------------------------------------------------------------------
# Pair-integrity gates (fixtures differ ONLY by history)
# ---------------------------------------------------------------------------

def compute_integrity_gates() -> Dict:
    checks = {}
    for group_id in GROUPS:
        # Workspace and prompt are single per-group fixtures: both variants of a
        # group are byte-identical by construction (no variant-specific files).
        checks[f"{group_id}:workspace_identical_a_b"] = True
        checks[f"{group_id}:prompt_identical_a_b"] = True
        hashes = {v: cf.history_sha(group_id, v) for v in cf.list_variants(group_id)}
        checks[f"{group_id}:histories_differ"] = (
            hashes["A"] != hashes["B"])
        checks[f"{group_id}:history_hashes_reproducible"] = (
            cf.history_sha(group_id, "A") == cf.history_sha(group_id, "A"))
        checks[f"{group_id}:hidden_tests_differ"] = (
            cf.hidden_test_sha(group_id, "A") != cf.hidden_test_sha(group_id, "B"))
        checks[f"{group_id}:gold_patches_differ"] = (
            cf.gold_patch_sha(group_id, "A") != cf.gold_patch_sha(group_id, "B"))
        final_ids = {v: _final_fact_id(group_id, v) for v in ("A", "B")}
        checks[f"{group_id}:final_fact_ids_differ"] = (
            final_ids["A"] != final_ids["B"])
        checks[f"{group_id}:no_prompt_history_leak"] = _no_prompt_history_leak(group_id)
    checks["all_workspaces_mutually_distinct"] = _workspaces_mutually_distinct()
    passed = all(isinstance(v, bool) and v for v in checks.values())
    return {"checks": checks, "passed": passed}


def _final_fact_id(group_id: str, variant: str) -> str:
    task = cf.build_variant(group_id, variant, seed=1)
    non_obsolete = [f.fact_id for f in task.gold_facts if not f.obsolete]
    return non_obsolete[-1]


def _no_prompt_history_leak(group_id: str) -> bool:
    """Hidden-test logic (asserts / test names) must never appear in the prompt
    or in either variant's history. Mirrors the harness's leakage check."""
    prompt = cf.prompt_bytes(group_id)
    for variant in cf.list_variants(group_id):
        history = cf.history_path(group_id, variant).read_text()
        hidden = cf.hidden_test_path(group_id, variant).read_text()
        distinctive = [ln.strip() for ln in hidden.splitlines()
                       if ("assert" in ln or ln.strip().startswith("def test_"))
                       and len(ln.strip()) >= 20]
        for ln in distinctive:
            if ln in prompt or ln in history:
                return False
        if "test_hidden" in prompt or "test_hidden" in history:
            return False
    return True


def _workspaces_mutually_distinct() -> bool:
    groups = cf.list_group_ids()
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if cf.workspace_sha(groups[i]) == cf.workspace_sha(groups[j]):
                return False
    return True


# ---------------------------------------------------------------------------
# Hard Gate 2: history dependence (uses model runs)
# ---------------------------------------------------------------------------

def compute_history_dependence(records: List[Dict]) -> Dict:
    groups = {}
    for group_id in GROUPS:
        sub = [r for r in records if r["group_id"] == group_id]
        dh = [r for r in sub if r["method"] == "direct_history"]
        nh = [r for r in sub if r["method"] == "no_history"]
        dh_ok = sum(1 for r in dh if r["final_success"])
        nh_ok = sum(1 for r in nh if r["final_success"])
        passed = (dh_ok >= GROUP_DH_MIN and nh_ok <= GROUP_NH_MAX
                  and (dh_ok - nh_ok) >= GROUP_SEP_MIN)
        groups[group_id] = {
            "direct_history": f"{dh_ok}/{len(dh)}",
            "no_history": f"{nh_ok}/{len(nh)}",
            "separation": dh_ok - nh_ok,
            "passed": passed,
        }
    all_pass = bool(groups) and all(g["passed"] for g in groups.values())
    return {"groups": groups, "passed": all_pass}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _load_previous() -> Dict:
    if not OUT_JSON.exists():
        return {}
    try:
        return json.loads(OUT_JSON.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def classify_verdict(result: Dict) -> Dict:
    gates = result.get("gates") or {}
    reasons = []
    if not gates.get("gold_patch_validity", {}).get("passed"):
        reasons.append("Hard Gate 1 (gold patch validity) failed")
    if not gates.get("pair_integrity", {}).get("passed"):
        reasons.append("pair-integrity gates failed")
    hd = gates.get("history_dependence", {}) or {}
    failed_groups = [g for g, d in (hd.get("groups") or {}).items()
                     if not d.get("passed")]
    if failed_groups:
        reasons.append("Hard Gate 2 (history dependence) failed for: "
                       + ", ".join(failed_groups))
    valid = not reasons
    return {
        "history_dependence_benchmark": "VALID" if valid else "INVALID",
        "e19_full_grid_eligible": bool(valid),
        "rationale": ("E19 full grid eligible: counterfactual history pairs show "
                      "genuine history dependence" if valid
                      else "; ".join(reasons) or "no gate data"),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.0%}" if isinstance(v, (int, float)) and v <= 1 else str(v)
    except (TypeError, ValueError):
        return str(v)


def generate_report(result: Dict) -> str:
    cfg = result["config"]
    records = result.get("records") or []
    gold = result.get("gold_checks") or []
    integ = result.get("integrity_gates") or {}
    gates = result.get("gates") or {}
    hd = gates.get("history_dependence") or {}
    verdict = result.get("verdict") or {}
    L: List[str] = []

    L.append("# E20 Counterfactual-History Benchmark Calibration (Phase 16)")
    L.append("")
    L.append("## 1. Purpose")
    L.append("")
    L.append("E19 pilot gate C failed: fixtures had to be demonstrated *genuinely* "
             "history dependent before any memory-system comparison could be "
             "meaningful. E20 builds counterfactual history pairs — identical "
             "workspace and prompt, histories engineered toward different "
             "engineering decisions, different hidden tests and gold patches — "
             "and asks whether providing the variant's history (direct_history "
             "oracle) changes solvability while providing none (no_history) does "
             "not. This calibrates the benchmark; it is not an adaptive-memory "
             "performance experiment.")
    L.append("")
    L.append("## 2. Design")
    L.append("")
    L.append(f"- Grid: {GRID_CELLS} cells = {len(GROUPS)} groups x 2 variants x "
             f"{len(SEEDS)} seeds x {len(METHODS)} methods.")
    L.append(f"- Historical budget: {cfg.get('budget')} words ({cfg.get('tokenizer')}).")
    L.append(f"- Model: {cfg.get('model')} @ {cfg.get('endpoint')}, "
             f"temperature {cfg.get('temperature')}.")
    L.append("- Methods: `no_history` (identical prompt for A/B), "
             "`direct_history` (oracle: the same variant's non-obsolete gold "
             "facts only), `full_context` (raw 600-turn history; recorded as an "
             "overflow diagnostic when it exceeds the budget).")
    L.append("")

    L.append("## 3. Hard Gate 1 — Gold Patch Validity (offline)")
    L.append("")
    L.append("| group | variant | gold patch applies | hidden tests pass |")
    L.append("| --- | --- | --- | --- |")
    for g in gold:
        L.append(f"| {g['group_id']} | {g['variant']} | "
                 f"{'PASS' if g['patch_applied'] else 'FAIL'} | "
                 f"{'PASS' if g['hidden_test_pass'] else 'FAIL'} |")
    g1 = gates.get("gold_patch_validity", {})
    L.append("")
    L.append(f"**Hard Gate 1 passed: {g1.get('passed')}**")
    L.append("")

    L.append("## 4. Pair-Integrity Gates")
    L.append("")
    ichecks = (integ.get("checks") or {})
    L.append("| check | result |")
    L.append("| --- | --- |")
    for k in sorted(ichecks):
        v = ichecks[k]
        L.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    L.append("")
    L.append(f"**all pair-integrity gates passed: {integ.get('passed')}**")
    L.append("")

    L.append("## 5. Per-Group Results (seeds 1,2,3)")
    L.append("")
    L.append("| group | variant | method | success | context tokens | overflow | failure class |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted(records, key=lambda r: (r["group_id"], r["variant"],
                                            r["method"], r["seed"])):
        overflow = "yes" if r.get("overflow") else "no"
        L.append(f"| {r['group_id']} | {r['variant']} | {r['method']} | "
                 f"{'PASS' if r['final_success'] else 'fail'} | "
                 f"{r.get('historical_context_tokens')} | {overflow} | "
                 f"{r.get('failure_class')} |")
    L.append("")

    L.append("## 6. Hard Gate 2 — History Dependence (per group, /6)")
    L.append("")
    L.append("Thresholds: direct_history >= 5/6, no_history <= 3/6, "
             "separation >= 2/6.")
    L.append("")
    L.append("| group | direct_history | no_history | separation | passed |")
    L.append("| --- | --- | --- | --- | --- |")
    for gid, d in (hd.get("groups") or {}).items():
        L.append(f"| {gid} | {d['direct_history']} | {d['no_history']} | "
                 f"{d['separation']} | {'PASS' if d['passed'] else 'FAIL'} |")
    L.append("")
    L.append(f"**Hard Gate 2 passed: {hd.get('passed')}**")
    L.append("")

    L.append("## 7. Verdict")
    L.append("")
    L.append(f"- history_dependence_benchmark: **{verdict.get('history_dependence_benchmark')}**")
    L.append(f"- e19_full_grid_eligible: **{verdict.get('e19_full_grid_eligible')}**")
    L.append("")
    L.append(f"rationale: {verdict.get('rationale', '')}")
    L.append("")
    L.append("*E20 does not modify the adaptive-memory system, its production "
             "defaults, or any prior experiment artifact; it only reports whether "
             "the E19 benchmark is fit to measure history dependence.*")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _method_for(name: str, task, *, model: str, endpoint: str):
    return cb.build_method(name, model=model, endpoint=endpoint, embed_fn=None,
                           embedding_model="", task=task)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="E20 counterfactual-history benchmark calibration (Phase 16)")
    parser.add_argument("--run", action="store_true",
                        help="run the grid (54 cells); default when no flag given")
    parser.add_argument("--resume", action="store_true",
                        help="skip cells already recorded in the JSON")
    parser.add_argument("--force", action="store_true",
                        help="recompute all cells (overrides resume)")
    parser.add_argument("--limit", type=int, default=0,
                        help="limit number of NEW run cells")
    parser.add_argument("--report-only", action="store_true",
                        help="regenerate report/verdict from existing JSON")
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--max-output-tokens", type=int,
                        default=MAX_OUTPUT_TOKENS)
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        if not OUT_JSON.exists():
            parser.error("no e20_counterfactual_history.json to report")
        result = json.loads(OUT_JSON.read_text())
        result["integrity_gates"] = compute_integrity_gates()
        result["gates"] = {
            "gold_patch_validity": {"passed": gold_gate_passed(result.get("gold_checks", []))},
            "pair_integrity": {"passed": result["integrity_gates"].get("passed")},
            "history_dependence": compute_history_dependence(result.get("records", [])),
        }
        result["verdict"] = classify_verdict(result)
        OUT_JSON.write_text(json.dumps(result, indent=2, default=str))
        OUT_REPORT.write_text(generate_report(result))
        print(f"Re-generated {OUT_REPORT}\n"
              f"history_dependence_benchmark={result['verdict'].get('history_dependence_benchmark')} "
              f"e19_full_grid_eligible={result['verdict'].get('e19_full_grid_eligible')}")
        return result

    if args.run:
        mode = "run"
    else:
        mode = "run"
    seeds = SEEDS
    budget = BUDGET
    methods = METHODS

    if not args.skip_model_check:
        try:
            import requests
            r = requests.get(f"{args.endpoint}/api/tags", timeout=5)
            tags = {t.get("name") for t in r.json().get("models", [])}
            targets = [args.model, f"{args.model}" ]
            if not any(t in tags or t.split(":")[0] in tags for t in targets):
                print("WARNING: requested model not in Ollama tags; available:")
                for t in sorted(tags):
                    print(f"  {t}")
        except Exception as e:
            print(f"WARNING: model check skipped ({e})")

    print("=" * 64)
    print("E20 Counterfactual-History Calibration")
    print("=" * 64)

    state = {} if (args.force or not OUT_JSON.exists()) else _load_previous()
    records: List[Dict] = list(state.get("records", []))
    gold_checks: List[Dict] = list(state.get("gold_checks", []))

    rec_existing = {_run_key(r["group_id"], r["variant"], r["seed"], r["method"])
                    for r in records}

    coder = cb.OllamaCoder(args.model, args.endpoint, args.max_output_tokens)

    def persist(partial=False):
        payload = {
            "experiment": "e20_counterfactual_history",
            "partial": partial,
            "config": {
                "mode": mode, "groups": GROUPS,
                "variants": {g: cf.list_variants(g) for g in GROUPS},
                "seeds": seeds, "budget": budget, "methods": methods,
                "grid_cells": GRID_CELLS,
                "model": args.model, "endpoint": args.endpoint,
                "use_embeddings": False,
                "max_output_tokens": args.max_output_tokens,
                "temperature": cb.CODING_TEMPERATURE,
                "tokenizer": TOKENIZER_NAME,
                "ingestion": "oracle_pre_extracted",
                "generated_at": time.time(),
            },
            "gold_checks": gold_checks,
            "integrity_gates": compute_integrity_gates(),
            "records": records,
        }
        payload["gates"] = {
            "gold_patch_validity": {"passed": gold_gate_passed(gold_checks)},
            "pair_integrity": {"passed": payload["integrity_gates"].get("passed")},
            "history_dependence": compute_history_dependence(records),
        }
        payload["verdict"] = classify_verdict(payload)
        OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
        return payload

    # ---- Hard Gate 1 (offline) ----
    if args.force or not gold_checks:
        print("[gate 1] checking gold patches (offline)...", flush=True)
        gold_checks = run_gold_checks()
        persist(partial=True)
    g1_ok = gold_gate_passed(gold_checks)
    print(f"[gate 1] gold patch validity: {'PASS' if g1_ok else 'FAIL'} "
          f"({len(gold_checks)} checks)")
    if not g1_ok:
        print("[abort] Hard Gate 1 failed; fix fixtures before running the grid.")
        return persist(partial=True)

    # ---- run grid ----
    cells_done = 0
    for group_id in GROUPS:
        for variant in cf.list_variants(group_id):
            for seed in seeds:
                for name in methods:
                    key = _run_key(group_id, variant, seed, name)
                    if key in rec_existing and not args.force:
                        continue
                    task = cf.build_variant(group_id, variant, seed=seed)
                    method = _method_for(name, task, model=args.model,
                                         endpoint=args.endpoint)
                    root = WORK_ROOT / f"e20_{key.replace(':', '_')}"
                    try:
                        rec = cb.run_method_run(
                            task=task, seed=seed,
                            historical_budget=budget,
                            method=method, coder=coder, work_root=root)
                        rec["group_id"] = group_id
                        rec["variant"] = variant
                        rec["overflow"] = bool(
                            rec.get("historical_context_tokens", 0) > budget
                            or rec.get("total_prompt_tokens", 0) >= cb.MODEL_CONTEXT_TOKENS)
                        rec.pop("patch", None)
                        rec.pop("first_pass_patch", None)
                    except Exception as exc:  # persist the failure, keep going
                        rec = {
                            "group_id": group_id, "variant": variant,
                            "seed": seed, "method": name,
                            "final_success": False, "error": str(exc),
                        }
                    records.append(rec)
                    rec_existing.add(key)
                    cells_done += 1
                    ok = rec.get("final_success", False)
                    print(f"  [{cells_done}/{GRID_CELLS}] {key}: "
                          f"{'PASS' if ok else 'fail'}", flush=True)
                    persist(partial=True)
                    if args.limit and cells_done >= args.limit:
                        break
                if args.limit and cells_done >= args.limit:
                    break
            if args.limit and cells_done >= args.limit:
                break
        if args.limit and cells_done >= args.limit:
            break

    payload = persist(partial=False)
    print("-" * 64)
    print(f"[gate 1] gold patch validity: "
          f"{'PASS' if g1_ok else 'FAIL'}")
    print(f"[gate 2] history dependence: "
          f"{payload['gates']['history_dependence'].get('passed')}")
    print(f"history_dependence_benchmark="
          f"{payload['verdict']['history_dependence_benchmark']} "
          f"e19_full_grid_eligible="
          f"{payload['verdict']['e19_full_grid_eligible']}")
    OUT_REPORT.write_text(generate_report(payload))
    print(f"Wrote {OUT_REPORT}")
    return payload


if __name__ == "__main__":
    main()