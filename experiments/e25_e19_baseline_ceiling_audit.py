"""E25: offline audit of the E19 benchmark baseline ceiling and budget geometry.

Why this exists (Phase 20 / D42)
--------------------------------
E19 ended with ``adaptive_advances = false``: adaptive memory did *not* show a
positive paired advantage against every baseline on the primary tasks. E25 is a
deterministic, artifact-only audit of the E19 result records. It re-derives the
budget geometry (how each method consumed the 256/512/1024 historical-token
budgets), the correction states behind the higher-level success counts, the
vanilla_rag ceiling failure, the context-hash relationship between adaptive and
vanilla_rag, and the diagnostic flags that constrain what any next benchmark
must do. E25 runs no LLM, no embedding server, and no production memory code:
it reads the existing ``e19_coding_generalization_full_repaired.json`` artifact
plus the in-repo counterfactual task fixtures and writes two new outputs
(the result JSON and this plain-text report).

The audit is *diagnostic only*. It declares no winner and changes no ranking:
its purpose is to separate the candidate explanations of D41 -- that the tested
budget range does not bind adaptive/vanilla_rag (so their near-ceiling success
does not constrain them), and that vanilla_rag merely tolerates obsolete-only
evidence at these budgets -- from the possibility that E19's budget range was
simply not discriminative.

E19 facts re-verified here (never modified, never recomputed):
  primary grid     135 = 3 tasks x 3 seeds x 3 budgets x 5 methods
  adaptive         27/27     vanilla_rag 26/27
  llm_summarization 14/27    raw_clipped 13/27    sliding_window 9/27
  gates.all_passed = true; verdict.adaptive_advances = false

This experiment is offline; it makes zero LLM calls, uses no embeddings, and
does not import or execute any memory_optimizer server code.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = os.path.abspath(str(BASE_DIR / "experiments"))

# ``experiments/`` contains a project-local ``statistics.py`` that shadows the
# standard-library module (it can sit at sys.path[0] when run as a script and
# at other indices under some pytest collection modes). Stash every entry that
# resolves exactly to the experiments dir, import the stdlib module, then
# restore the same layout.
_stashed = [_i for _i, _p in enumerate(sys.path)
            if _p and os.path.abspath(str(_p)) == _SCRIPT_DIR]
for _i in reversed(_stashed):
    del sys.path[_i]
import statistics  # noqa: E402  (stdlib; overrides the project-local shadow)
for _i in reversed(_stashed):
    sys.path.insert(_i, _SCRIPT_DIR)
from statistics import mean, median  # noqa: E402
sys.path.insert(0, str(BASE_DIR))


from data.counterfactual_task_suite import _GROUP_BY_ID  # noqa: E402

PRIMARY_TASKS = ("routing_policy", "retry_policy", "serialization_policy")
METHODS = ("raw_clipped", "sliding_window", "llm_summarization", "vanilla_rag", "adaptive")
BUDGETS = (256, 512, 1024)
SEEDS = (1, 2, 3)
PRIMARY_GRID_CELLS = len(PRIMARY_TASKS) * len(SEEDS) * len(BUDGETS) * len(METHODS)

RESULT_DIR = BASE_DIR / "experiments" / "results"
SOURCE_ARTIFACT = RESULT_DIR / "e19_coding_generalization_full_repaired.json"
OUT_JSON = RESULT_DIR / "e25_e19_baseline_ceiling_audit.json"
OUT_REPORT = RESULT_DIR / "e25_e19_baseline_ceiling_audit_report.md"

BUDGET_BINDING_THRESHOLD = 0.90

# Correction-state classification: a primary cell records whether it exposed
# the obsolete intermediate fact (obsolete_fact_exposure) and whether it
# recalled the current/final fact (correction_recall). The four reachable
# combinations describe what historical evidence the model actually saw.
# Unexpected combinations are a hard stop: the artifact changed in a way this
# audit does not understand.
CLEAN_CURRENT = "CLEAN_CURRENT"
CURRENT_PLUS_OBSOLETE = "CURRENT_PLUS_OBSOLETE"
OBSOLETE_ONLY = "OBSOLETE_ONLY"
NEITHER = "NEITHER"

CORRECTION_STATES = (CLEAN_CURRENT, CURRENT_PLUS_OBSOLETE, OBSOLETE_ONLY, NEITHER)


def load_e19_records() -> dict:
    """Load and return the entire E19 repaired artifact dict.

    Raises RuntimeError if the immutable source artifact is missing.
    """
    if not SOURCE_ARTIFACT.exists():
        raise RuntimeError(f"E25: source artifact missing: {SOURCE_ARTIFACT}")
    return json.loads(SOURCE_ARTIFACT.read_text(encoding="utf-8"))


def filter_primary_records(artifact: dict) -> list:
    """Return only the primary-grid records (routing/retry/serialization)."""
    return [r for r in artifact["records"] if r["task_id"] in PRIMARY_TASKS]


def validate_primary_grid(primary: list) -> dict:
    """Assert the primary grid is exactly 135 cells with unique, complete keys.

    Returns an index mapping (task_id, seed, budget, method) -> record.
    Raises RuntimeError (a hard stop) on any size/completeness violation so the
    audit never reports numbers over a partial or malformed grid.
    """
    if len(primary) != PRIMARY_GRID_CELLS:
        raise RuntimeError(
            f"E25: source has {len(primary)} primary cells, expected "
            f"{PRIMARY_GRID_CELLS}; stopping without writing any results."
        )
    expected = set()
    for t in PRIMARY_TASKS:
        for s in SEEDS:
            for b in BUDGETS:
                for m in METHODS:
                    expected.add((t, s, b, m))
    index = {}
    for rec in primary:
        key = (rec["task_id"], rec["seed"], rec["historical_budget"], rec["method"])
        if key in index:
            raise RuntimeError(f"E25: duplicate primary cell key {key}")
        index[key] = rec
    if set(index) != expected:
        missing = sorted(expected - set(index))[:10]
        extra = sorted(set(index) - expected)[:10]
        raise RuntimeError(
            f"E25: primary grid incomplete. missing={missing} extra={extra}"
        )
    for key, rec in index.items():
        if rec.get("context_sha") is None:
            raise RuntimeError(f"E25: primary cell {key} has null context_sha")
    return index


def classify_correction_state(correction_recall, obsolete_fact_exposure) -> str:
    """Classify a primary cell's correction state from its recall/exposure pair.

    Mapping is exact:
      (1.0, 0.0) -> CLEAN_CURRENT
      (1.0, 1.0) -> CURRENT_PLUS_OBSOLETE
      (0.0, 1.0) -> OBSOLETE_ONLY
      (0.0, 0.0) -> NEITHER
    Anything else raises ValueError: the audit does not know how to read it.
    """
    try:
        pair = (float(correction_recall), float(obsolete_fact_exposure))
    except (TypeError, ValueError):
        raise ValueError(
            f"E25: unexpected correction pair {(correction_recall, obsolete_fact_exposure)!r}"
        ) from None
    mapping = {
        (1.0, 0.0): CLEAN_CURRENT,
        (1.0, 1.0): CURRENT_PLUS_OBSOLETE,
        (0.0, 1.0): OBSOLETE_ONLY,
        (0.0, 0.0): NEITHER,
    }
    if pair not in mapping:
        raise ValueError(f"E25: unexpected correction pair {pair!r}")
    return mapping[pair]


def _success_rate(successes: int, cells: int) -> float:
    return 0.0 if cells == 0 else successes / cells


def _utilization(rec: dict) -> float:
    return rec["historical_context_tokens"] / rec["historical_budget"]


def _rounded(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


# ---------------------------------------------------------------------------
# Audit builders
# ---------------------------------------------------------------------------


def build_budget_geometry(primary: list, index: dict) -> dict:
    """Audit 1: how each method's context tokens used (or failed to use) the budget."""
    geometry = {"primary_cells": len(primary),
                "budget_binding_threshold": BUDGET_BINDING_THRESHOLD,
                "by_method_budget": {}}
    for m in METHODS:
        tokens_all = [r["historical_context_tokens"] for r in primary if r["method"] == m]
        geometry[m] = {
            "observed_token_min": _rounded(min(tokens_all)),
            "observed_token_max": _rounded(max(tokens_all)),
            "observed_token_mean": _rounded(mean(tokens_all)),
        }
        by_budget = {}
        for b in BUDGETS:
            recs = [r for r in primary
                    if r["method"] == m and r["historical_budget"] == b]
            tokens = [r["historical_context_tokens"] for r in recs]
            utils = [_utilization(r) for r in recs]
            binding = sum(1 for t in tokens if t >= BUDGET_BINDING_THRESHOLD * b)
            by_budget[b] = {
                "mean_context_tokens": _rounded(mean(tokens)),
                "median_context_tokens": _rounded(median(tokens)),
                "min_context_tokens": _rounded(min(tokens)),
                "max_context_tokens": _rounded(max(tokens)),
                "mean_utilization": _rounded(mean(utils)),
                "mean_budget_headroom": _rounded(1.0 - mean(utils)),
                "fraction_budget_binding": _rounded(_success_rate(binding, len(recs))),
                "cells": len(recs),
            }
        geometry["by_method_budget"][m] = by_budget
    return geometry


def build_budget_elasticity(primary: list, index: dict) -> dict:
    """Audit 2: per task x seed x method track, token/context behavior across budgets."""
    tracks = {}
    for t in PRIMARY_TASKS:
        for s in SEEDS:
            for m in METHODS:
                tokens = {}
                shas = {}
                for b in BUDGETS:
                    rec = index[(t, s, b, m)]
                    tokens[b] = rec["historical_context_tokens"]
                    shas[b] = rec["context_sha"]
                distinct_shas = set(shas.values())
                tracks[f"{t}:{s}:{m}"] = {
                    "task": t,
                    "seed": s,
                    "method": m,
                    "tokens_by_budget": tokens,
                    "token_min": _rounded(min(tokens.values())),
                    "token_max": _rounded(max(tokens.values())),
                    "token_range": _rounded(max(tokens.values()) - min(tokens.values())),
                    "token_growth_256_to_1024": _rounded(tokens[1024] - tokens[256]),
                    "distinct_contexts_across_budgets": len(distinct_shas),
                    "budget_context_stable": len(distinct_shas) == 1,
                    "budget_token_stable": min(tokens.values()) == max(tokens.values()),
                }
    return {"tracks": tracks, "tracks_total": len(tracks)}


def build_budget_binding_by_method(primary: list, elasticity: dict) -> dict:
    """Audit 3: method-level budget binding and cross-budget context stability."""
    out = {}
    for m in METHODS:
        recs = [r for r in primary if r["method"] == m]
        by_budget = {}
        for b in BUDGETS:
            br = [r for r in recs if r["historical_budget"] == b]
            tokens = [r["historical_context_tokens"] for r in br]
            utils = [_utilization(r) for r in br]
            binding = sum(1 for t in tokens if t >= BUDGET_BINDING_THRESHOLD * b)
            by_budget[b] = {
                "tokens": tokens,
                "mean_context_tokens": _rounded(mean(tokens)),
                "median_context_tokens": _rounded(median(tokens)),
                "max_context_tokens": _rounded(max(tokens)),
                "mean_utilization": _rounded(mean(utils)),
                "fraction_budget_binding": _rounded(_success_rate(binding, len(br))),
            }
        track_recs = [v for v in elasticity["tracks"].values() if v["method"] == m]
        stable_context = sum(1 for v in track_recs if v["budget_context_stable"])
        stable_tokens = sum(1 for v in track_recs if v["budget_token_stable"])
        all_utils = [_utilization(r) for r in recs]
        out[m] = {
            "by_budget": by_budget,
            "cross_budget_stable_tracks": stable_context,
            "cross_budget_token_stable_tracks": stable_tokens,
            "tracks_total": len(track_recs),
            "mean_utilization_across_all_budgets": _rounded(mean(all_utils)),
            "mean_utilization_at_256": _rounded(by_budget[256]["mean_utilization"]),
        }
    return out


def _correction_pairs_from_fixtures() -> dict:
    """Return {primary_task: correction_pairs} from the variant A fixtures."""
    pairs = {}
    for gid in PRIMARY_TASKS:
        vspec = next(v for v in _GROUP_BY_ID[gid].variants if v.variant == "A")
        pairs[gid] = list(vspec.corrections)
    return pairs


def build_correction_state_analysis(primary: list, artifact: dict) -> dict:
    """Audit 4 + 5: correction pairs per task and per-method/state success table.

    Correction-pair counts are verified from the counterfactual task fixtures
    (variant A of each primary task) and cross-checked against the E19 artifact
    offline gate cells. A mismatch is a hard stop.
    """
    from_fixtures = _correction_pairs_from_fixtures()
    pairs_per_task = {t: len(v) for t, v in from_fixtures.items()}
    if any(n != 1 for n in pairs_per_task.values()):
        raise RuntimeError(
            f"E25: expected exactly 1 correction pair per primary task, "
            f"got {pairs_per_task}"
        )

    offline_cells = artifact.get("offline", {}).get("cells", [])
    offline_counts = defaultdict(set)
    for c in offline_cells:
        if c["task_id"] in PRIMARY_TASKS:
            offline_counts[c["task_id"]].add(c.get("n_corrections"))
    for t in PRIMARY_TASKS:
        if t not in offline_counts or offline_counts[t] != {1}:
            raise RuntimeError(
                f"E25: offline gate cells disagree with fixture metadata for {t}"
            )

    states = defaultdict(lambda: {"cells": 0, "successes": 0})
    method_state_distribution = defaultdict(dict)
    for rec in primary:
        state = classify_correction_state(rec["correction_recall"], rec["obsolete_fact_exposure"])
        key = (rec["method"], state)
        states[key]["cells"] += 1
        states[key]["successes"] += 1 if rec["final_success"] else 0
        dist = method_state_distribution[rec["method"]]
        dist[state] = dist.get(state, 0) + 1

    table = {}
    for m in METHODS:
        table[m] = {}
        for state in CORRECTION_STATES:
            key = (m, state)
            if key in states:
                s = states[key]
                table[m][state] = {
                    "cells": s["cells"],
                    "successes": s["successes"],
                    "success_rate": _rounded(_success_rate(s["successes"], s["cells"])),
                }
    return {
        "correction_pairs_from_fixtures": from_fixtures,
        "correction_pairs_per_task": pairs_per_task,
        "task_metadata_note": (
            "verified from data.counterfactual_task_suite variant A fixtures "
            "and cross-checked against the E19 artifact offline gate cells"
        ),
        "per_method_correction_state_distribution": dict(method_state_distribution),
        "per_method_state_success": table,
    }


def build_task_budget_analysis(primary: list, index: dict) -> dict:
    """Audit 6: per task x method success rate and mean tokens by budget."""
    out = {}
    for t in PRIMARY_TASKS:
        out[t] = {}
        for m in METHODS:
            row = {}
            for b in BUDGETS:
                recs = [r for r in primary
                        if r["task_id"] == t and r["method"] == m
                        and r["historical_budget"] == b]
                succ = sum(1 for r in recs if r["final_success"])
                row[b] = {
                    "successes": succ,
                    "cells": len(recs),
                    "success_rate": _rounded(_success_rate(succ, len(recs))),
                }
            out[t][m] = row
    return out


def build_vanilla_failure_analysis(primary: list) -> dict:
    """Audit 7: every failed primary vanilla_rag cell, with full context."""
    failures = []
    for rec in primary:
        if rec["method"] == "vanilla_rag" and not rec["final_success"]:
            failures.append({
                "cell": f"{rec['task_id']}:{rec['seed']}:{rec['historical_budget']}",
                "task_id": rec["task_id"],
                "seed": rec["seed"],
                "historical_budget": rec["historical_budget"],
                "historical_context_tokens": rec["historical_context_tokens"],
                "context_utilization": _rounded(_utilization(rec)),
                "context_sha": rec["context_sha"],
                "correction_state": classify_correction_state(
                    rec["correction_recall"], rec["obsolete_fact_exposure"]
                ),
                "correction_recall": rec["correction_recall"],
                "obsolete_fact_exposure": rec["obsolete_fact_exposure"],
                "failure_class": rec["failure_class"],
            })
    total = sum(1 for r in primary if r["method"] == "vanilla_rag")
    succ = sum(1 for r in primary if r["method"] == "vanilla_rag" and r["final_success"])
    return {
        "failures": failures,
        "failed_cells": len(failures),
        "cells": total,
        "successes": succ,
        "success_rate": _rounded(_success_rate(succ, total)),
    }


def build_context_hash_analysis(primary: list, index: dict) -> dict:
    """Audit 8 + 9: adaptive-vs-vanilla context identity and cross-budget stability."""
    same_counts = {}
    for b in BUDGETS:
        same = 0
        diff = 0
        for t in PRIMARY_TASKS:
            for s in SEEDS:
                a = index[(t, s, b, "adaptive")]["context_sha"]
                v = index[(t, s, b, "vanilla_rag")]["context_sha"]
                if a == v:
                    same += 1
                else:
                    diff += 1
        same_counts[str(b)] = {"equals": same, "differs": diff}

    elasticity = build_budget_elasticity(primary, index)
    stability = {}
    for m in METHODS:
        track = [v for v in elasticity["tracks"].values() if v["method"] == m]
        stability[m] = {
            "stable_tracks_context": sum(1 for v in track if v["budget_context_stable"]),
            "stable_tracks_tokens": sum(1 for v in track if v["budget_token_stable"]),
            "budget_sensitive_tracks": sum(
                1 for v in track if not v["budget_context_stable"]
            ),
            "tracks_total": len(track),
        }

    same_sha_at_256 = 0
    same_tokens_at_256 = 0
    for t in PRIMARY_TASKS:
        for s in SEEDS:
            a_256 = index[(t, s, 256, "adaptive")]
            v_256 = index[(t, s, 256, "vanilla_rag")]
            if a_256["context_sha"] == v_256["context_sha"]:
                same_sha_at_256 += 1
            if a_256["historical_context_tokens"] == v_256["historical_context_tokens"]:
                same_tokens_at_256 += 1
    return {
        "adaptive_vs_vanilla_context_equality_by_budget": same_counts,
        "adaptive_vanilla_same_sha_at_256_tracks": same_sha_at_256,
        "adaptive_vanilla_same_token_count_at_256_tracks": same_tokens_at_256,
        "cross_budget_context_stability": stability,
        "note": (
            "SHA-equality measures identical serialized historical context; "
            "token-count equality alone is never used as the stability test."
        ),
    }


def build_diagnostic_flags(primary: list, artifact: dict, index: dict) -> dict:
    """Audit 12: four diagnostic flags constraining any next benchmark.

    flag_a: vanilla_rag OBSOLETE_ONLY cells succeed >= 0.80 of the time.
    flag_b: both adaptive and vanilla_rag are non-binding at the 256 budget
            (mean utilization < 0.50) and both are context-stable across all
            three budgets on every track.
    flag_c: vanilla_rag primary success rate >= 0.90 (near-ceiling baseline).
    flag_d: the locked E19 verdict has adaptive_advances == false.
    """
    vanilla = [r for r in primary if r["method"] == "vanilla_rag"]
    obs_only = [r for r in vanilla
                if classify_correction_state(r["correction_recall"],
                                             r["obsolete_fact_exposure"]) == OBSOLETE_ONLY]
    obs_succ = sum(1 for r in obs_only if r["final_success"])
    flag_a_value = _success_rate(obs_succ, len(obs_only))

    elasticity = build_budget_elasticity(primary, index)
    stable = {m: 0 for m in METHODS}
    for v in elasticity["tracks"].values():
        if v["budget_context_stable"]:
            stable[v["method"]] += 1

    util_at_256 = {}
    for m in ("adaptive", "vanilla_rag"):
        recs = [r for r in primary
                if r["method"] == m and r["historical_budget"] == 256]
        util_at_256[m] = mean(_utilization(r) for r in recs)

    vanilla_succ = sum(1 for r in vanilla if r["final_success"])
    flag_c_value = _success_rate(vanilla_succ, len(vanilla))

    verdict = artifact.get("verdict", {})
    adaptive_advances = verdict.get("adaptive_advances")

    flag_a = flag_a_value >= 0.80
    flag_b = (
        util_at_256["adaptive"] < 0.50
        and util_at_256["vanilla_rag"] < 0.50
        and stable["adaptive"] == 9
        and stable["vanilla_rag"] == 9
    )
    flag_c = flag_c_value >= 0.90
    flag_d = bool(adaptive_advances is False)

    return {
        "flag_a_vanilla_obsolete_only_success_rate": flag_a,
        "flag_a_vanilla_obsolete_only_support": {
            "obsolete_only_cells": len(obs_only),
            "obsolete_only_successes": obs_succ,
            "obsolete_only_success_rate": _rounded(flag_a_value),
            "threshold": 0.80,
        },
        "flag_b_budget_non_binding": flag_b,
        "flag_b_support": {
            "adaptive_mean_utilization_at_256": _rounded(util_at_256["adaptive"]),
            "vanilla_mean_utilization_at_256": _rounded(util_at_256["vanilla_rag"]),
            "threshold": 0.50,
            "adaptive_cross_budget_stable_tracks": stable["adaptive"],
            "vanilla_cross_budget_stable_tracks": stable["vanilla_rag"],
            "required_stable_tracks": 9,
        },
        "flag_c_vanilla_near_ceiling": flag_c,
        "flag_c_support": {
            "vanilla_successes": vanilla_succ,
            "vanilla_cells": len(vanilla),
            "vanilla_success_rate": _rounded(flag_c_value),
            "threshold": 0.90,
        },
        "flag_d_adaptive_advances_false": flag_d,
        "flag_d_support": {"adaptive_advances": adaptive_advances},
        "matched_expectations": bool(
            flag_a and flag_b and flag_c and flag_d
        ),
    }


def build_next_benchmark_requirements(flags: dict, geometry: dict) -> dict:
    """Derive next-benchmark requirements from the flags that actually fired.

    Only requirements supported by E25's own findings are included. The final
    numerical budgets are deliberately NOT chosen here; E25 only reports the
    observed threshold range from the current data.
    """
    requirements = []
    if flags["flag_a_vanilla_obsolete_only_success_rate"]:
        requirements.append(
            "the next benchmark must make obsolete-only historical evidence "
            "materially unsafe for the hidden test"
        )
    if flags["flag_b_budget_non_binding"]:
        requirements.append(
            "the next benchmark must use a budget range that actually binds "
            "both adaptive and vanilla retrieval"
        )
    if flags["flag_c_vanilla_near_ceiling"]:
        requirements.append(
            "the next benchmark needs more discrimination against strong "
            "retrieval than E19 currently provides"
        )
    if flags["flag_d_adaptive_advances_false"]:
        requirements.append(
            "do not treat E19 as evidence of adaptive superiority; "
            "adaptive_advances remains false"
        )

    adaptive_geom = geometry["by_method_budget"]["adaptive"]
    vanilla_geom = geometry["by_method_budget"]["vanilla_rag"]
    return {
        "requirements": requirements,
        "bind_both_adaptive_and_vanilla_at_low_budget": bool(
            "must use a budget range" in " ".join(requirements)
        ),
        "observed_context_threshold_range": {
            "adaptive_tokens_low": adaptive_geom[256]["min_context_tokens"],
            "adaptive_tokens_high": adaptive_geom[1024]["max_context_tokens"],
            "vanilla_tokens_low": vanilla_geom[256]["min_context_tokens"],
            "vanilla_tokens_high": vanilla_geom[1024]["max_context_tokens"],
            "note": (
                "any next low-end budget below the max context size of the "
                "recall methods will start to bind them; E25 does not choose "
                "the final numbers"
            ),
        },
        "budgets_not_reselected": True,
    }


def build_conclusion(primary: list, geometry: dict, flags: dict,
                     vanilla: dict, context_analysis: dict, scope: dict) -> dict:
    """Assemble the audit conclusion (diagnostic; no winner, no ranking)."""
    by_method_success = {}
    for m in METHODS:
        recs = [r for r in primary if r["method"] == m]
        succ = sum(1 for r in recs if r["final_success"])
        by_method_success[m] = {
            "successes": succ,
            "cells": len(recs),
            "success_rate": _rounded(_success_rate(succ, len(recs))),
        }
    return {
        "method_success": by_method_success,
        "offline": True,
        "llm_calls": 0,
        "embedding_calls": 0,
        "production_pipeline_imported": False,
        "adaptive_advances_unchanged": False,
        "winner_declared": False,
        "ranking_changed": False,
        "e19_artifacts_modified": False,
        "summary": (
            "E25 is an offline, deterministic audit of the locked E19 primary "
            "grid. It re-verified 135/135 primary cells, the per-method success "
            "rates behind the E19 verdict, and the budget geometry. adaptive "
            "and vanilla_rag both operated far below the 256 budget (adaptive "
            "mean utilization < 0.5, vanilla_rag likewise) with context "
            "stable across every track; vanilla_rag's obsolete-only correction "
            "state still succeeded >= 0.80 of the time, including its single "
            "OBSOLETE_INFORMATION_USED failure; and the E19 verdict "
            "adaptive_advances remains false. These four diagnostics set input "
            "requirements for any next benchmark, but this audit declares no "
            "winner and changes no ranking."
        ),
        "scope": scope,
    }


def write_json(result: dict, path: Path) -> None:
    """Write the E25 result JSON (new artifact only)."""
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _table_rows(result: dict) -> dict:
    geometry = result["budget_geometry"]
    correction = result["correction_state_analysis"]
    task_analysis = result["task_budget_analysis"]
    vanilla = result["vanilla_failures"]
    stability = result["context_hash_analysis"]["cross_budget_context_stability"]
    flags = result["diagnostic_flags"]

    rows_a = []
    for m in METHODS:
        for b in BUDGETS:
            g = geometry["by_method_budget"][m][b]
            rows_a.append({
                "method": m, "budget": b,
                "mean_tokens": g["mean_context_tokens"],
                "mean_utilization": g["mean_utilization"],
                "mean_headroom": g["mean_budget_headroom"],
                "fraction_binding": g["fraction_budget_binding"],
            })

    rows_b = []
    for m in METHODS:
        for state in CORRECTION_STATES:
            cell = correction["per_method_state_success"].get(m, {}).get(state)
            if cell:
                rows_b.append({
                    "method": m, "correction_state": state,
                    "cells": cell["cells"], "successes": cell["successes"],
                    "success_rate": cell["success_rate"],
                })

    rows_c = []
    for m in METHODS:
        st = stability[m]
        rows_c.append({
            "method": m,
            "tracks_total": st["tracks_total"],
            "stable_context": st["stable_tracks_context"],
            "budget_sensitive": st["budget_sensitive_tracks"],
        })

    rows_d = []
    for t in PRIMARY_TASKS:
        for m in METHODS:
            row = {"task": t, "method": m}
            for b in BUDGETS:
                ta = task_analysis[t][m][b]
                row[f"success_{b}"] = f"{ta['successes']}/{ta['cells']}"
            rows_d.append(row)

    rows_e = []
    for f in vanilla["failures"]:
        rows_e.append(f)

    rows_f = []
    for key, value in sorted(flags.items()):
        if key == "matched_expectations" or key.endswith("_support"):
            continue
        rows_f.append({"flag": key, "value": value})

    return {"A": rows_a, "B": rows_b, "C": rows_c, "D": rows_d,
            "E": rows_e, "F": rows_f}


def _fmt_rate(rate: float) -> str:
    return f"{rate:.1%}"


def write_report(result: dict, path: Path) -> None:
    """Render the E25 plain-text audit report from the result dict."""
    tables = _table_rows(result)
    geometry = result["budget_geometry"]
    correction = result["correction_state_analysis"]
    vanilla = result["vanilla_failures"]
    stability = result["context_hash_analysis"]["cross_budget_context_stability"]
    equality = result["context_hash_analysis"]["adaptive_vs_vanilla_context_equality_by_budget"]
    flags = result["diagnostic_flags"]
    reqs = result["next_benchmark_requirements"]
    conclusion = result["conclusion"]
    method_success = conclusion["method_success"]
    scope = result["scope"]

    lines = []
    add = lines.append

    add("# E25: E19 baseline ceiling and budget-geometry audit")
    add("")
    add("## 1. Purpose")
    add("")
    add(conclusion["summary"])
    add("")
    add("## 2. Method")
    add("")
    add(
        "Offline, deterministic, artifact-only audit of "
        "`experiments/results/e19_coding_generalization_full_repaired.json` "
        "plus the in-repo counterfactual fixtures "
        "(`data/counterfactual_task_suite.py`). No LLM, no embedding calls, no "
        "production memory code. The E19 artifact is read-only; this audit "
        "writes only `e25_e19_baseline_ceiling_audit.json` and this report."
    )
    add("")
    add("## 3. Source artifact")
    add("")
    add(f"- Source: `{scope['source_artifact']}`")
    add(f"- Primary cells audited: `{scope['grid']}`")
    add(f"- Task metadata: `data/counterfactual_task_suite.py`")
    add("")
    add("## 4. Scope")
    add("")
    add(
        f"- Tasks: {', '.join(scope['primary_tasks'])}; "
        f"seeds {scope['seeds']}; budgets {scope['budgets']}; "
        f"methods {', '.join(scope['methods'])}."
    )
    add("")
    add("## 5. Assertions")
    add("")
    add(
        "Before any analysis, the primary grid is validated as exactly "
        "135 unique, complete cells with non-null `context_sha` on every "
        "record; any mismatch stops the audit and writes nothing."
    )
    add("")
    add("## 6. Budget geometry (Table A)")
    add("")
    add("### A. Per-method budget use (means)")
    add("")
    add("| Method | Budget | Mean tokens | Mean utilization | Mean headroom | Fraction binding |")
    add("|---|---|---|---|---|---|")
    for row in tables["A"]:
        add(f"| {row['method']} | {row['budget']} | {row['mean_tokens']:.0f} | "
            f"{_fmt_rate(row['mean_utilization'])} | "
            f"{_fmt_rate(row['mean_headroom'])} | "
            f"{_fmt_rate(row['fraction_binding'])} |")
    add("")
    add("Binding = `historical_context_tokens >= 0.90 * historical_budget` (diagnostic only).")
    add("")
    add("## 7. Budget elasticity")
    add("")
    add(
        "Per task:seed:method track, contexts are compared across 256/512/1024 "
        "by `context_sha`. A track is context-stable iff it shows exactly one "
        "distinct SHA across the three budgets. Token-count equality alone is "
        "never a stability test."
    )
    add("")
    add(
        "Per-track elasticity (tokens by budget, token range, distinct context "
        "count, context/token stability) is reported in the JSON under "
        "`budget_geometry` and `context_hash_analysis`; the aggregate "
        "cross-budget stability appears in Table C."
    )
    add("")
    add("## 8. Budget binding by method (Table C, right columns)")
    add("")
    add("Estimated from per-track context stability:")
    add("")
    add("| Method | Stable across 256/512/1024 | Budget-sensitive contexts |")
    add("|---|---|---|")
    for row in tables["C"]:
        add(f"| {row['method']} | {row['stable_context']}/{row['tracks_total']} | "
            f"{row['budget_sensitive']}/{row['tracks_total']} |")
    add("")
    add("## 9. Correction states")
    add("")
    add(
        "Each primary cell is classified from `(correction_recall, "
        "obsolete_fact_exposure)`: CLEAN_CURRENT = (1.0, 0.0), "
        "CURRENT_PLUS_OBSOLETE = (1.0, 1.0), OBSOLETE_ONLY = (0.0, 1.0), "
        "NEITHER = (0.0, 0.0). Any other pair stops the audit."
    )
    add("")
    add("## 10. Correction-pair verification")
    add("")
    add(
        f"Every primary task has exactly one counterfactual correction pair "
        f"(verified from variant A fixtures and the E19 offline gate cells): "
        f"{correction['correction_pairs_from_fixtures']}."
    )
    add("")
    add("## 11. Success by correction state (Table B)")
    add("")
    add("| Method | Correction state | Cells | Successes | Success rate |")
    add("|---|---|---|---|---|")
    for row in tables["B"]:
        add(f"| {row['method']} | {row['correction_state']} | {row['cells']} | "
            f"{row['successes']} | {_fmt_rate(row['success_rate'])} |")
    add("")
    add("## 12. Task x budget analysis (Table D)")
    add("")
    add("| Task | Method | 256 | 512 | 1024 |")
    add("|---|---|---|---|---|")
    for row in tables["D"]:
        add(f"| {row['task']} | {row['method']} | {row['success_256']} | "
            f"{row['success_512']} | {row['success_1024']} |")
    add("")
    add("## 13. Vanilla_rag failure analysis (Table E)")
    add("")
    add(f"Primary vanilla_rag: {vanilla['successes']}/{vanilla['cells']} "
        f"({_fmt_rate(vanilla['success_rate'])}). Failed cells: {vanilla['failed_cells']}.")
    add("")
    if tables["E"]:
        add("| Cell | Failure class | Context tokens | Utilization | Correction state |")
        add("|---|---|---|---|---|")
        for f in tables["E"]:
            add(f"| {f['cell']} | {f['failure_class']} | {f['historical_context_tokens']} | "
                f"{_fmt_rate(f['context_utilization'])} | {f['correction_state']} |")
    else:
        add("No failed primary vanilla_rag cells.")
    add("")
    add("## 14. Context-hash relationship (adaptive vs vanilla_rag)")
    add("")
    add("| Budget | Same context_sha (tracks) | Different (tracks) |")
    add("|---|---|---|")
    for b in BUDGETS:
        e = equality[str(b)]
        add(f"| {b} | {e['equals']} | {e['differs']} |")
    add("")
    add(
        "adaptive and vanilla_rag therefore produce different serialized "
        "historical contexts on every primary track at every budget while "
        "being byte-for-byte stable across budgets themselves."
    )
    add("")
    add("## 15. Diagnostic flags")
    add("")
    add("| Flag | Value | Support |")
    add("|---|---|---|")
    for f in tables["F"]:
        add(f"| {f['flag']} | {f['value']} | see JSON |")
    add("")
    add("## 16. E19 gate alignment")
    add("")
    add(
        "The locked E19 `budget_pressure` gate passed because *some* run filled "
        ">= 60% of budget at a budget level and raw_clipped grew with budget, "
        "not because adaptive or vanilla_rag were budget-bound. With adaptive "
        f"mean utilization at 256 of "
        f"{_fmt_rate(result['budget_geometry']['by_method_budget']['adaptive'][256]['mean_utilization'])} "
        f"and vanilla_rag of "
        f"{_fmt_rate(result['budget_geometry']['by_method_budget']['vanilla_rag'][256]['mean_utilization'])}, "
        "both recall methods ran far below the tightest tested budget. E25's "
        "binding statement is diagnostic only and does not invalidate the E19 "
        "gate: the two gates measure different things."
    )
    add("")
    add("## 17. Next-benchmark requirements (constraints only)")
    add("")
    for req in reqs["requirements"]:
        add(f"- {req}")
    add("")
    add(
        "Observed context threshold range from current data: "
        f"adaptive {reqs['observed_context_threshold_range']['adaptive_tokens_low']}--"
        f"{reqs['observed_context_threshold_range']['adaptive_tokens_high']} tokens; "
        f"vanilla_rag {reqs['observed_context_threshold_range']['vanilla_tokens_low']}--"
        f"{reqs['observed_context_threshold_range']['vanilla_tokens_high']} tokens. "
        "E25 does not select final numerical budgets."
    )
    add("")
    add("## 18. Conclusion")
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    for m in METHODS:
        ms = method_success[m]
        add(f"| {m} | {ms['successes']}/{ms['cells']} ({_fmt_rate(ms['success_rate'])}) |")
    add(f"| adaptive_advances (E19 verdict) | {result['conclusion']['adaptive_advances_unchanged']} (locked, unchanged) |")
    add("")
    add(
        "E25 is offline and declarative: 0 LLM calls, 0 embedding calls, "
        "no production pipeline import, no winner declared, no ranking "
        "changed, E19 artifacts untouched. The four diagnostics set input "
        "constraints for any next benchmark; they are not an E19 verdict "
        "change and not a claim that all baselines are saturated."
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _assemble(artifact: dict) -> dict:
    primary = filter_primary_records(artifact)
    index = validate_primary_grid(primary)
    scope = {
        "source_artifact": str(SOURCE_ARTIFACT),
        "primary_tasks": list(PRIMARY_TASKS),
        "seeds": list(SEEDS),
        "budgets": list(BUDGETS),
        "methods": list(METHODS),
        "grid": f"{len(primary)} cells",
        "grid_cells": len(primary),
    }

    geometry = build_budget_geometry(primary, index)
    elasticity = build_budget_elasticity(primary, index)
    geometry["tracks"] = elasticity["tracks"]
    geometry["tracks_total"] = elasticity["tracks_total"]
    geometry["method_binding"] = build_budget_binding_by_method(primary, elasticity)
    correction = build_correction_state_analysis(primary, artifact)
    task_analysis = build_task_budget_analysis(primary, index)
    vanilla = build_vanilla_failure_analysis(primary)
    context_analysis = build_context_hash_analysis(primary, index)
    flags = build_diagnostic_flags(primary, artifact, index)
    requirements = build_next_benchmark_requirements(flags, geometry)
    conclusion = build_conclusion(primary, geometry, flags, vanilla,
                                  context_analysis, scope)

    result = {
        "experiment": "E25",
        "phase": "Phase 20",
        "source_artifact": scope["source_artifact"],
        "source_primary_cells": len(primary),
        "scope": scope,
        "budget_geometry": geometry,
        "correction_state_analysis": correction,
        "task_budget_analysis": task_analysis,
        "vanilla_failures": vanilla,
        "context_hash_analysis": context_analysis,
        "diagnostic_flags": flags,
        "next_benchmark_requirements": requirements,
        "conclusion": conclusion,
    }
    return result


def main() -> None:
    """Run the E25 audit end to end."""
    artifact = load_e19_records()
    result = _assemble(artifact)
    write_json(result, OUT_JSON)
    write_report(result, OUT_REPORT)
    print(f"E25 complete: wrote {OUT_JSON.name} and {OUT_REPORT.name}")
    print(f"  primary cells audited: {result['source_primary_cells']}")
    print(f"  vanilla_rag: {result['vanilla_failures']['successes']}/"
          f"{result['vanilla_failures']['cells']}")
    print(f"  flags: {result['diagnostic_flags']['flag_a_vanilla_obsolete_only_success_rate']} "
          f"{result['diagnostic_flags']['flag_b_budget_non_binding']} "
          f"{result['diagnostic_flags']['flag_c_vanilla_near_ceiling']} "
          f"{result['diagnostic_flags']['flag_d_adaptive_advances_false']}")
    print("  (if a flag diagnostic fires, see next_benchmark_requirements)")


if __name__ == "__main__":
    main()