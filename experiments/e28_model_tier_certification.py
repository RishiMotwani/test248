"""E28 — model-tier certification of the frozen E27 calibration surface (Phase 23).

Purpose
-------
E27 (Phase 22) calibrated a single-policy counterfactual measurement surface and
was stopped at two failed pilot gates at ``llama3.1:8b``:
``history_dependence`` (every group) and ``contradiction_state`` (vanilla RAG
obsolete-fact exposure 2/3 < 3/4). E28 asks one question: is that failure a
property of the *benchmark surface* or of the *model tier* the surface was
measured at? It does so by re-running the byte-identical, frozen 48-cell E27
procedure at exactly one other tier, ``qwen2.5:7b``, with the exact same
fixtures, budget, methods, embeddings and locked gate thresholds. E28 never
redesigns the surface, never re-runs E26/E19, never declares a winner and never
runs a head-to-head grid.

Delegation
----------
This module is a thin runner: it rebinds E27's output artifacts to the
``e28_model_tier_certification*`` files and delegates the whole 48-cell grid +
gate computation to ``e27_calibration.main`` (the same technique used by the
E22/E23 E19 wrappers). No benchmark logic or gate thresholds are duplicated
here; ``compute_pilot_gates`` and ``compute_verdict`` are E27's, unchanged.
After the run the E27-shaped payload is wrapped into the E28 result shape (with
``model``, ``source_calibration``, ``model_comparison`` and the E28 verdict) and
an E28 report is rendered.

Usage::

    python experiments/e28_model_tier_certification.py --dry-run
    python experiments/e28_model_tier_certification.py --run [--resume|--force]
    python experiments/e28_model_tier_certification.py --report-only
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import coding_benchmark as cb  # noqa: E402
from experiments import e27_calibration as e27  # noqa: E402
from experiments.e17_coding_capability import check_models  # noqa: E402
from data.e27_calibration_suite import as_coding_task, build_variant  # noqa: E402

CANDIDATE_MODEL = "qwen2.5:7b"
SOURCE_MODEL = e27.DEFAULT_MODEL  # "llama3.1:8b"

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_JSON = RESULTS_DIR / "e28_model_tier_certification.json"
OUT_REPORT = RESULTS_DIR / "e28_model_tier_certification_report.md"
WORK_ROOT = RESULTS_DIR / "e28_work"
SOURCE_JSON = e27.OUT_JSON
SOURCE_REPORT = e27.OUT_REPORT

PROBE_GROUP = "route_contract"
PROBE_VARIANT = "A"
PROBE_SEED = 1

GATE_NAMES = ["history_dependence", "contradiction_state", "budget_binding",
              "context_difference", "no_leakage"]

SECTIONS = [
    "Purpose",
    "Why E27 Was Frozen",
    "Model-Tier Protocol",
    "History-Dependence Gate",
    "Contradiction-State Gate",
    "Budget-Binding Gate",
    "Context-Difference Gate",
    "Leakage Gate",
    "Llama3.1:8b vs Qwen2.5:7b Gate Comparison",
    "Per-Group Results",
    "Retry-Contract Analysis",
    "Calibration Verdict",
    "What E28 Establishes",
    "What E28 Does Not Establish",
    "Eligibility for a Future Head-to-Head",
]


def load_source() -> Dict:
    if not SOURCE_JSON.exists():
        raise SystemExit(
            f"source calibration artifact missing: {SOURCE_JSON} "
            "(E27 must have completed and been committed first)")
    return json.loads(SOURCE_JSON.read_text(encoding="utf-8"))


def model_comparison(source: Dict, candidate: Dict) -> Dict:
    src_gates = (((source or {}).get("pilot_gates") or {}).get("gates") or {})
    cand_gates = (((candidate or {}).get("pilot_gates") or {}).get("gates")
                  or {})
    src_v = (source or {}).get("verdict") or {}
    cand_v = candidate.get("verdict") or {}
    gates = {}
    for name in GATE_NAMES:
        gates[name] = {
            "source": src_gates.get(name),
            "candidate": cand_gates.get(name),
            "passed_source": bool((src_gates.get(name) or {}).get("passed")),
            "passed_candidate": bool(
                (cand_gates.get(name) or {}).get("passed")),
        }
    agg_src = {}
    for m in e27.METHODS:
        agg_src[m] = ((source or {}).get("aggregate") or {}).get(
            m, {}).get("success_rate")
    return {
        "source_model": SOURCE_MODEL,
        "candidate_model": CANDIDATE_MODEL,
        "surface": "identical frozen 48-cell E27 calibration grid",
        "source_calibration_relative": "experiments/results/e27_calibration.json",
        "gates": gates,
        "aggregate": {"source": agg_src},
        "verdict": {
            "source": {k: src_v.get(k) for k in
                       ("calibration_valid", "full_head_to_head_eligible")},
            "candidate": {k: cand_v.get(k) for k in
                          ("calibration_valid",
                           "full_head_to_head_eligible")},
        },
        "no_model_ranking": True,
    }


def wrap_payload(payload: Dict, source: Dict) -> Dict:
    """Wrap an E27-shaped pilot payload into the E28 result shape."""
    verdict_src = payload.get("verdict") or {}
    config = dict(payload.get("config") or {})
    config["model"] = CANDIDATE_MODEL
    config["protocol"] = f"{payload.get('experiment')}@{CANDIDATE_MODEL}"
    gates_passed = {
        name: bool(((payload.get("pilot_gates") or {}).get("gates") or {})
                   .get(name, {}).get("passed"))
        for name in GATE_NAMES
    }
    sourced = payload.get("experiment") or "e27_calibration"
    calibration_valid = bool(verdict_src.get("calibration_valid"))
    h2h = bool(verdict_src.get(
        "full_head_to_head_eligible",
        verdict_src.get("head_to_head_eligible")))
    rationale = list(verdict_src.get("rationale") or [])
    if payload.get("experiment") != "e28_model_tier_certification":
        rationale.append(
            "executed at qwen2.5:7b under the frozen E27 protocol")
    return {
        "experiment": "e28_model_tier_certification",
        "model": CANDIDATE_MODEL,
        "source_calibration": {
            "experiment": sourced,
            "model": SOURCE_MODEL,
            "json_relative": "experiments/results/e27_calibration.json",
            "report_relative": "experiments/results/e27_calibration_report.md",
            "verdict": {
                "calibration_valid": verdict_src.get("calibration_valid"),
                "full_head_to_head_eligible": h2h,
            },
        },
        "config": config,
        "records": payload.get("records") or [],
        "fixture_gates": payload.get("fixture_gates"),
        "pilot_gates": payload.get("pilot_gates"),
        "aggregate": payload.get("aggregate"),
        "diagnostics": payload.get("diagnostics"),
        "model_comparison": model_comparison(source or {}, payload),
        "verdict": {
            "calibration_valid": calibration_valid,
            "model": CANDIDATE_MODEL,
            "source_calibration": ("e27_calibration@"
                                   f"{SOURCE_MODEL}"),
            "gates_passed": gates_passed,
            "head_to_head_eligible": h2h,
            "rationale": rationale,
        },
    }


# ---------------------------------------------------------------------------
# Report rendering (E28: sections 1-15, Tables A-E)
# ---------------------------------------------------------------------------

def _fmt(x, nd=3) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _gate_detail(gate: Dict) -> List[str]:
    L: List[str] = []
    if "per_group" in gate:
        for gid, g in gate["per_group"].items():
            L.append(f"  - {gid}: " + "; ".join(
                f"{k}={_fmt(g[k])}" for k in g if k != "passed"))
    if "methods" in gate:
        for m, g in gate["methods"].items():
            L.append(f"  - {m}: " + "; ".join(
                f"{k}={_fmt(g[k])}" for k in g if k != "passed"))
    if "leaked_cells" in gate:
        L.append(f"  - leaked_cells={_fmt(gate['leaked_cells'])}")
    if "pairs" in gate:
        L.append(f"  - pairs={_fmt(gate.get('pairs'))} "
                 f"differ={_fmt(gate.get('differ'))} "
                 f"fraction={_fmt(gate.get('fraction'))}")
    return L


def generate_report(result: Dict) -> str:
    cfg = result.get("config") or {}
    records = result.get("records") or []
    fixtures = result.get("fixture_gates") or {}
    gates = result.get("pilot_gates") or {}
    agg = result.get("aggregate") or {}
    comp = result.get("model_comparison") or {}
    verdict = result.get("verdict") or {}
    src_v = ((result.get("source_calibration") or {}).get("verdict")) or {}
    groups = cfg.get("groups") or e27.GROUPS
    variants = cfg.get("variants") or e27.VARIANTS
    seeds = cfg.get("seeds") or e27.SEEDS
    budget = (cfg.get("budgets") or e27.BUDGETS)[0]
    gate_map = gates.get("gates") or {}
    per_group = (gate_map.get("history_dependence") or {}).get("per_group")
    contra = (gate_map.get("contradiction_state") or {}).get("methods") or {}
    bind = (gate_map.get("budget_binding") or {}).get("methods") or {}
    ctx_diff = gate_map.get("context_difference") or {}
    leakage = gate_map.get("no_leakage") or {}
    calibrated = bool(verdict.get("calibration_valid"))

    def heading(n: int) -> str:
        return f"## {n}. {SECTIONS[n - 1]}"

    L: List[str] = []
    L.append("# E28 Model-Tier Certification - Qwen2.5:7b @ Frozen E27 Surface (Phase 23)")
    L.append("")
    L.append(heading(1))
    L.append("")
    L.append("E28 certifies whether the frozen E27 calibration surface behaves "
             "as the memory system expects at a *second* LLM tier "
             f"(`{CANDIDATE_MODEL}`), measured with the exact same 48-cell "
             "procedure, fixtures, budget, methods, embeddings and locked gate "
             "thresholds that produced E27's verdict at "
             f"`{SOURCE_MODEL}`. It does NOT redesign the surface, does NOT "
             "re-run E26/E19, does NOT declare a winner and does NOT run a "
             "head-to-head grid.")
    L.append("")
    L.append(heading(2))
    L.append("")
    L.append(f"E27 was stopped at two failed pilot gates and its "
             f"`calibration_valid=False` verdict at `{SOURCE_MODEL}` is on "
             "record — the artifacts are byte-immutable and are never re-run. "
             "E28 therefore re-uses the E27 runner exactly (same thresholds in "
             "`e27_calibration`, delegated through a thin output-rebinding "
             "wrapper, so the E27 suite, runner, tests and result files stay "
             "untouched). The only dimension changed is the serving model "
             f"(`{CANDIDATE_MODEL}`).")
    L.append("")
    L.append(heading(3))
    L.append("")
    L.append("- grid: identical frozen 48-cell geometry "
             "(3 groups x 2 variants x seeds {1, 2} x budget 96 x 4 methods)")
    L.append(f"- groups: {', '.join(groups)}; variants: {', '.join(variants)}; "
             f"seeds: {seeds}; budget: {budget}")
    L.append(f"- methods: {', '.join(e27.METHODS)}")
    L.append(f"- model: `{CANDIDATE_MODEL}`; embedding: "
             f"`{cfg.get('embedding_model')}`; endpoint: `{cfg.get('endpoint')}`; "
             f"temperature: {cfg.get('temperature')}")
    L.append("- offline fixture gates are re-run before any LLM call "
             "(route_contract/serialization_contract/retry_contract "
             "west/history/gold geometry must hold)")
    L.append("- gate thresholds are E27's, unchanged: history dependence "
             f"(direct>=3/4, no_history<=1/4, diff>=2 cells), contradiction "
             "state (adaptive corr_recall>=0.90, obs_exposure<=0.10; vanilla "
             "obs_exposure>=0.75), budget binding (>=75% of recall-method cells "
             f"at >=0.8x budget), context difference (>=80% of the 12 pairs), "
             "no leakage (0 cells).")
    L.append(f"- fraction run: {len(records)} cell(s) recorded.")
    L.append("")
    L.append(heading(4))
    L.append("")
    if "history_dependence" in gate_map:
        g = gate_map["history_dependence"]
        L.append(f"- status: {'PASS' if g['passed'] else 'FAIL'}"
                 " (locked: direct_history >= 3/4, no_history <= 1/4, "
                 "per-group difference >= 2 cells)")
        L.extend(_gate_detail(g))
    L.append("")
    L.append(heading(5))
    L.append("")
    if "contradiction_state" in gate_map:
        g = gate_map["contradiction_state"]
        L.append(f"- status: {'PASS' if g['passed'] else 'FAIL'}"
                 " (locked: adaptive corr_recall >= 0.90 and "
                 "obs_exposure <= 0.10; vanilla obs_exposure >= 0.75)")
        L.extend(_gate_detail(g))
    L.append("")
    L.append("**Table D. Contradiction-state detail "
             f"(`{CANDIDATE_MODEL}`).**")
    L.append("")
    L.append("| method | corr_recall | obs_exposure | 12-cell pass |")
    L.append("|---|---|---|---|")
    for m in e27.RECALL_METHODS:
        g = contra.get(m) or {}
        L.append(f"| {m} | {_fmt(g.get('correction_recall'))} | "
                 f"{_fmt(g.get('obsolete_fact_exposure'))} | "
                 f"{'PASS' if g.get('passed') else 'FAIL'} |")
    L.append("")
    L.append(heading(6))
    L.append("")
    if "budget_binding" in gate_map:
        g = gate_map["budget_binding"]
        L.append(f"- status: {'PASS' if g['passed'] else 'FAIL'}"
                 " (locked: >=75% of each recall-method cell at "
                 f"budget_ratio >= 0.8 x {budget} = {0.8 * budget:.1f} tokens)")
        L.extend(_gate_detail(g))
    L.append("")
    L.append("**Table E. Budget / context-pair structure "
             f"(`{CANDIDATE_MODEL}`).**")
    L.append("")
    L.append("| recall method | cells | bound_cells | bound fraction | threshold |")
    L.append("|---|---|---|---|---|")
    for m in e27.RECALL_METHODS:
        g = bind.get(m) or {}
        L.append(f"| {m} | {_fmt(g.get('cells'))} | "
                 f"{_fmt(g.get('bound_cells'))} | "
                 f"{_fmt(g.get('fraction'))} | >= 0.75 |")
    pairs = ctx_diff.get("pairs") or 0
    L.append(f"\nadaptive-vs-vanilla context_sha pairs: "
             f"{ctx_diff.get('differ', 0)} / {pairs} differ "
             f"(threshold >= 80%).")
    L.append("")
    L.append(heading(7))
    L.append("")
    if "context_difference" in gate_map:
        g = gate_map["context_difference"]
        L.append(f"- status: {'PASS' if g['passed'] else 'FAIL'}"
                 " (locked: >=80% of the 12 adaptive-vs-vanilla paired cells "
                 "differ in context_sha)")
        L.extend(_gate_detail(g))
    L.append("")
    L.append(heading(8))
    L.append("")
    if "no_leakage" in gate_map:
        g = gate_map["no_leakage"]
        L.append(f"- status: {'PASS' if g['passed'] else 'FAIL'}"
                 " (locked: 0 leaked cells)")
        L.extend(_gate_detail(g))
    L.append("")
    L.append(heading(9))
    L.append("")
    L.append("E28 compares the gate verdicts of the identical surface at the two "
             "tiers. A divergence isolates the model tier as the causal factor; "
             "a match keeps the surface-level explanation intact. No model is "
             "ranked.")
    L.append("")
    L.append("**Table A. Gate-by-gate comparison "
             f"(`{SOURCE_MODEL}` vs `{CANDIDATE_MODEL}`).**")
    L.append("")
    L.append("| gate | E27 @ llama3.1:8b | E28 @ qwen2.5:7b | outcome |")
    L.append("|---|---|---|---|")
    for name in GATE_NAMES:
        g = (comp.get("gates") or {}).get(name) or {}
        spass = g.get("passed_source")
        cpass = g.get("passed_candidate")
        state = ("both pass" if spass and cpass
                 else "both fail" if not spass and not cpass
                 else "tier-dependent")
        L.append(f"| {name} | {'PASS' if spass else 'FAIL'} | "
                 f"{'PASS' if cpass else 'FAIL'} | {state} |")
    src_cal = (comp.get("verdict") or {}).get("source") or {}
    cand_cal = (comp.get("verdict") or {}).get("candidate") or {}
    L.append(f"\nsource calibration_valid: {_fmt(src_cal.get('calibration_valid'))}; "
             f"candidate calibration_valid: "
             f"{_fmt(cand_cal.get('calibration_valid'))}.")
    L.append("")
    L.append(heading(10))
    L.append("")
    L.append("**Table B. Per-method success and faithfulness diagnostics "
             f"(`{CANDIDATE_MODEL}`; E27 source success in brackets).**")
    L.append("")
    L.append("| method | runs | E27 success | E28 success | E28 corr_recall | "
             "E28 obs_exposure | E28 mean budget ratio |")
    L.append("|---|---|---|---|---|---|---|")
    src_succ = ((comp.get("aggregate") or {}).get("source") or {})
    for m in e27.METHODS:
        row = agg.get(m) or {}
        L.append(f"| {m} | {_fmt(row.get('runs'))} | "
                 f"{_fmt(src_succ.get(m))} | "
                 f"{_fmt(row.get('success_rate'))} | "
                 f"{_fmt(row.get('correction_recall'))} | "
                 f"{_fmt(row.get('obsolete_fact_exposure'))} | "
                 f"{_fmt(row.get('mean_budget_ratio'))} |")
    L.append("")
    L.append(f"**Table C. Per-group history dependence "
             f"(`{CANDIDATE_MODEL}`).**")
    L.append("")
    L.append("| group | direct_history success | no_history success | diff cells | pass |")
    L.append("|---|---|---|---|---|")
    for gid in groups:
        g = (per_group or {}).get(gid) or {}
        L.append(f"| {gid} | {_fmt(g.get('direct_history_success'))} | "
                 f"{_fmt(g.get('no_history_success'))} | "
                 f"{_fmt(g.get('diff_cells'))} | "
                 f"{'PASS' if g.get('passed') else 'FAIL'} |")
    L.append("")
    L.append(heading(11))
    L.append("")
    rt = (per_group or {}).get("retry_contract") or {}
    L.append(f"On the E27@`{SOURCE_MODEL}` run the retry_contract group failed "
             "the strongest (direct_history success 0/4 at the earlier tier). "
             f"At `{CANDIDATE_MODEL}` retry_contract direct_history success is "
             f"{_fmt(rt.get('direct_history_success'))} and no_history success "
             f"is {_fmt(rt.get('no_history_success'))} "
             f"(diff_cells={_fmt(rt.get('diff_cells'))}); the group "
             f"{'PASSES' if rt.get('passed') else 'FAILS'} the history-dependence "
             "gate. This is the key cell-by-cell probe for whether the earlier "
             "failure was a tier artifact of the coding model rather than a "
             "surface defect.")
    L.append("")
    L.append(heading(12))
    L.append("")
    L.append("**Calibration verdict (E28 @ qwen2.5:7b, frozen E27 protocol).**")
    L.append("")
    L.append(f"- calibration_valid: **{calibrated}**")
    L.append(f"- model: `{CANDIDATE_MODEL}`; source calibration: "
             f"e27_calibration@`{SOURCE_MODEL}`")
    L.append(f"- gates_passed: {verdict.get('gates_passed')}")
    L.append(f"- head_to_head_eligible: **{verdict.get('head_to_head_eligible')}**")
    for r in (verdict.get("rationale") or []):
        L.append(f"- rationale: {r}")
    L.append("")
    L.append(heading(13))
    L.append("")
    if calibrated:
        L.append(f"The frozen E27 calibration surface is **valid** at "
                 f"`{CANDIDATE_MODEL}`: every offline fixture gate and every "
                 "pilot gate passes on a complete 48-cell grid. Because the "
                 f"identical surface failed at `{SOURCE_MODEL}`, the E27 "
                 "failure is model-tier dependent: it is attributable to the "
                 "llama3.1:8b serving tier / model behaviour rather than to a "
                 "defect of the calibrated surface itself.")
    else:
        L.append(f"The frozen E27 calibration surface is **not valid** at "
                 f"`{CANDIDATE_MODEL}`: the locked pilot gates did not pass "
                 "under the identical protocol. The E27 failure therefore does "
                 "NOT isolate to the llama3.1:8b tier alone; the surface "
                 "remains the leading explanation. No redesign is performed "
                 "here - this is a measurement, not a fix.")
    L.append("")
    L.append(heading(14))
    L.append("")
    L.append("- No adaptive-vs-vanilla result and no 'winner'; "
             "`adaptive_advances=false` (Phase 18 / E19) remains in force.")
    L.append("- No claim about any model tier other than "
             f"`{SOURCE_MODEL}` and `{CANDIDATE_MODEL}`.")
    L.append("- No claim about throughput, latency or serving cost.")
    L.append("- No claim that the surface is valid at any tier on evidence "
             "from one grid; gate thresholds are pre-registered and locked.")
    L.append("")
    L.append(heading(15))
    L.append("")
    if calibrated:
        L.append(f"`head_to_head_eligible = True` at `{CANDIDATE_MODEL}`: the "
                 "surface measured correctly at this tier, so a head-to-head "
                 "grid at this tier may be planned - but ONLY as a separate, "
                 "explicitly commissioned phase. E28 does not run it and never "
                 "compares methods as competitors outside an authorised "
                 "head-to-head.")
    else:
        L.append(f"`head_to_head_eligible = False` at `{CANDIDATE_MODEL}`: the "
                 "frozen surface does not behave as expected at either tier. "
                 "No head-to-head may be planned; stop and bring the two-tier "
                 "gate evidence to the architect for the surface-level "
                 "redesign decision.")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Dry run (context construction only; no LLM)
# ---------------------------------------------------------------------------

def dry_run(*, endpoint: str, embedding_model: str, embed_fn=None) -> Dict:
    """Build all four contexts for one representative variant, no LLM calls."""
    if embed_fn is None:
        embed_fn = e27._resolve_embed(embedding_model, endpoint)
    task = build_variant(PROBE_GROUP, PROBE_VARIANT, seed=PROBE_SEED)
    ctask = as_coding_task(task)
    budget = e27.BUDGETS[0]
    history = cb.HistoryBundle(ctask.task_id, ctask.history,
                               ctask.history_text)
    out: Dict = {"probe": f"{PROBE_GROUP}_{PROBE_VARIANT}", "seed": PROBE_SEED,
                 "budget": budget, "methods": {}}
    for name in e27.METHODS:
        method = cb.build_method(name, model=CANDIDATE_MODEL, endpoint=endpoint,
                                 embed_fn=embed_fn,
                                 embedding_model=embedding_model, task=ctask)
        prepared = method.prepare(history, budget, seed=PROBE_SEED)
        context = method.retrieve(prepared, ctask.task_prompt, budget)
        tokens = (cb.count_tokens(context) if context.strip() else 0)
        out["methods"][name] = {
            "historical_context_tokens": tokens,
            "context_sha": e27._sha16(context) if context.strip() else None,
            "memory_ids": sorted(prepared.memory_ids),
        }
    print(f"E28 dry-run (no LLM): method -> tokens / context_sha")
    for name, m in out["methods"].items():
        print(f"  {name:>15}: {m['historical_context_tokens']} tokens "
              f"sha={m['context_sha']}")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def preflight(endpoint: str, embedding_model: str,
              skip_model_check: bool = False) -> None:
    if not SOURCE_JSON.exists():
        raise SystemExit(
            f"source calibration artifact missing: {SOURCE_JSON}")
    if not skip_model_check:
        check_models(CANDIDATE_MODEL, embedding_model, endpoint,
                     use_embeddings=True)
    print("=" * 64)
    print("E28 preflight")
    print(f"  model:           {CANDIDATE_MODEL}")
    print(f"  embedding_model: {embedding_model}")
    print(f"  endpoint:        {endpoint}")
    print(f"  source:          {SOURCE_JSON.name} (from {SOURCE_MODEL})")
    print(f"  outputs:         {OUT_JSON.name}, {OUT_REPORT.name}")
    print("E28 preflight: ok")
    print("=" * 64)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="E28 model-tier certification (Phase 23)")
    parser.add_argument("--run", action="store_true",
                        help="delegate the frozen E27 grid to e27.main at "
                             "qwen2.5:7b, then wrap into the E28 result")
    parser.add_argument("--dry-run", action="store_true",
                        help="construct the four contexts only; no LLM")
    parser.add_argument("--report-only", action="store_true",
                        help="re-wrap and re-render from the existing E28 JSON")
    parser.add_argument("--resume", action="store_true",
                        help="skip cells already present (default for --run)")
    parser.add_argument("--force", action="store_true",
                        help="recompute all 48 cells")
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--endpoint", default=e27.DEFAULT_ENDPOINT)
    parser.add_argument("--embedding-model",
                        default=e27.DEFAULT_EMBED_MODEL)
    args = parser.parse_args(argv)

    if args.dry_run:
        dry_run(endpoint=args.endpoint,
                embedding_model=args.embedding_model)
        return 0

    if args.report_only:
        if not OUT_JSON.exists():
            parser.error(f"no existing E28 result at {OUT_JSON}")
        payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        result = wrap_payload(payload, load_source())
        OUT_JSON.write_text(json.dumps(result, indent=2, default=str))
        OUT_REPORT.write_text(generate_report(result))
        print("calibration_valid=",
              result["verdict"]["calibration_valid"],
              "head_to_head_eligible=",
              result["verdict"]["head_to_head_eligible"])
        return 0

    if not args.run:
        parser.error("pass --run, --dry-run or --report-only")

    preflight(args.endpoint, args.embedding_model, args.skip_model_check)

    del_args = []
    if args.force:
        del_args.append("--force")
    else:
        del_args.append("--resume")
    if args.skip_model_check:
        del_args.append("--skip-model-check")

    original_json = e27.OUT_JSON
    original_report = e27.OUT_REPORT
    original_work = e27.WORK_ROOT
    try:
        e27.OUT_JSON = OUT_JSON
        e27.OUT_REPORT = OUT_REPORT
        e27.WORK_ROOT = WORK_ROOT
        payload = e27.main(
            ["--pilot", "--model", CANDIDATE_MODEL,
             "--endpoint", args.endpoint,
             "--embedding-model", args.embedding_model] + del_args)
        result = wrap_payload(payload, load_source())
        OUT_JSON.write_text(json.dumps(result, indent=2, default=str))
        OUT_REPORT.write_text(generate_report(result))
    finally:
        e27.OUT_JSON = original_json
        e27.OUT_REPORT = original_report
        e27.WORK_ROOT = original_work

    v = result["verdict"]
    print("calibration_valid=", v["calibration_valid"])
    print("head_to_head_eligible=", v["head_to_head_eligible"])
    print("gates_passed=", v["gates_passed"])
    return 0


if __name__ == "__main__":
    sys.exit(main())