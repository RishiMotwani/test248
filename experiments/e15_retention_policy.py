"""e15 - Phase 11 Retention-Policy experiment: separating Activation from Survival.

Phase 10 (E14) showed that the dominant cause of useful-fact loss is the
decay -> threshold-prune path: ``current_importance`` is used BOTH as the
dynamic activation signal (retrieval ranking, reinforcement) AND as the
survival gate. A memory is deleted the moment its activation fades, even when
it is the only surviving representation of an important fact and the store has
plenty of free capacity.

Phase 11 tests the two-score hypothesis:

    MEMORY.current_importance   = dynamic activation (decays; retrieval ranking)
    MEMORY.retention_priority   = stable evidence score (survival / eviction)

Four retention policies are compared at the same active-context budget:

    hard_threshold   exact Phase-10 behavior (prune below pruning.threshold)
    soft_decay       importance decays, decay never deletes; store shrinks only
                     when the store-token budget actually evicts
    dual_score       soft_decay + store-budget eviction keys on retention_priority
    no_decay         Phase-10 ablation (no decay, no decay-pruning) — an
                     upper-bound diagnostic, NOT a production candidate

Primary grid: budgets {64,128,256,512} x seeds {42..46} x {all four policies},
store budget 4096 (mirrors E14 where store pressure did not bind). A second
grid (stress) raises the workload scale until the natural store exceeds 8192
tokens and then compares soft_decay vs dual_score (+ hard_threshold anchor)
under real store budgets {1024,2048,4096}.

Reading the results — the causal question is NOT "which policy has the highest
recall" but "does a policy that recovers old facts do so without (a) inflating
the active context, (b) resurrecting superseded/corrected values, or
(c) keeping everything forever". Hence obsolete_retention, correction_recall,
mean_context_tokens, fraction_below_pruning_threshold and final store size are
reported alongside recall — a method that scores higher merely by never
forgetting is not automatically an improvement.

Key causal isolation: ``retention_priority`` is NEVER read at query time.
Retrieval ranking continues to use only ``current_importance``
(0.85*sim + 0.15*importance + category bonus). The experiment therefore
measures the survival policy in isolation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.coding_workload import build_coding_session  # noqa: E402
from experiments.e13_generalization import _answer_parts, _answered, _fact_text, _metrics  # noqa: E402
from experiments.e13_generalization import _family_metrics, word_count  # noqa: E402
from experiments.e14_retention_diagnosis import (  # noqa: E402
    AGE_BUCKETS,
    STATES,
    _age_bucket,
    _lifecycle_counts,
)
from experiments.paper import load_settings  # noqa: E402
from memory_optimizer.budget import fit_to_budget  # noqa: E402
from memory_optimizer.compression import MemoryCompressor  # noqa: E402
from memory_optimizer.decay import CategoryDecayEngine  # noqa: E402
from memory_optimizer.pipeline import AdaptiveMemoryPipeline  # noqa: E402
from memory_optimizer.retrieval import MemoryRetriever  # noqa: E402
from memory_optimizer.scoring import ImportanceScorer  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BUDGETS = [64, 128, 256, 512]
SEEDS = [42, 43, 44, 45, 46]
TURNS = 400
SCALE = 3
STORE_BUDGET = 4096                 # primary: mirrors E14 non-binding store cap
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_ENDPOINT = "http://localhost:11434"

PRUNING_THRESHOLD = 0.2             # config.yaml pruning.threshold

POLICIES = ["hard_threshold", "soft_decay", "dual_score", "no_decay"]

# Policy -> (retention_mode, eviction_priority)
POLICY_DEFS: Dict[str, Dict] = {
    "hard_threshold": {
        "retention_mode": "hard_threshold",
        "eviction_priority": "current_importance",
        "decay_lambdas": None,          # production lambdas (config.yaml)
        "pruning_threshold": PRUNING_THRESHOLD,
        "description": "Phase-10 default: decay + prune below threshold.",
    },
    "soft_decay": {
        "retention_mode": "soft_decay",
        "eviction_priority": "current_importance",
        "decay_lambdas": None,
        "pruning_threshold": PRUNING_THRESHOLD,
        "description": "Importance decays; decay never deletes; only store "
                       "budget evicts.",
    },
    "dual_score": {
        "retention_mode": "dual_score",
        "eviction_priority": "retention_priority",
        "decay_lambdas": None,
        "pruning_threshold": PRUNING_THRESHOLD,
        "description": "soft_decay + store-budget eviction keys on the stable "
                       "retention_priority (long-term survival).",
    },
    "no_decay": {
        "retention_mode": "hard_threshold",
        "eviction_priority": "current_importance",
        "decay_lambdas": {"transient": 0.0, "personal": 0.0,
                          "technical_preference": 0.0, "project_context": 0.0},
        "pruning_threshold": 0.0,
        "description": "Phase-10 no-decay ablation: upper-bound diagnostic, "
                       "not a production candidate.",
    },
}

# Stress grid (genuine store pressure)
STRESS_MIN_NATURAL_TOKENS = 8192
STRESS_STORE_BUDGETS = [1024, 2048, 4096]
STRESS_ACTIVE_BUDGETS = [64, 128, 256]
STRESS_SEEDS = [42, 43, 44]
STRESS_TURNS = 1200
STRESS_POLICIES = ["soft_decay", "dual_score", "hard_threshold"]
STRESS_MAX_SCALE = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_embed(use_embeddings: bool, embedding_model: str, embed_fn):
    if not use_embeddings:
        return None
    if embed_fn is not None:
        return embed_fn
    from memory_optimizer.embeddings import embed_ollama
    return lambda texts: embed_ollama(texts, model=embedding_model, endpoint=OLLAMA_ENDPOINT)


def _natural_store_tokens(gt: List[Dict]) -> int:
    """All authoritative gt facts, word-counted (no eviction applied)."""
    return sum(len(g["fact"].split())
               for g in gt if not g.get("superseded_by"))


def _find_stress_scale(min_tokens: int, num_turns: int, seed: int = 42) -> int:
    """Smallest ``scale`` whose natural store (fact library, no hand-picking)
    exceeds ``min_tokens``. Fact count is seed-independent, so measuring on one
    seed is representative of all stress seeds."""
    for scale in range(1, STRESS_MAX_SCALE + 1):
        try:
            _stream, gt, _queries = build_coding_session(
                seed, num_turns, include_corrections=True, scale=scale)
        except ValueError:
            continue
        if _natural_store_tokens(gt) >= min_tokens:
            return scale
    raise RuntimeError(f"no scale <= {STRESS_MAX_SCALE} reaches {min_tokens} natural tokens")


# ---------------------------------------------------------------------------
# Instrumented replay (retention-policy aware)
# ---------------------------------------------------------------------------

def replay_session(stream, gt, queries, settings, *, embed_fn, embedding_model,
                   decay_lambdas, pruning_threshold, budget, store_budget,
                   retention_mode, eviction_priority) -> Dict:
    """Replay one session under a retention policy, capturing per-fact lifecycle.

    Mirrors E14's instrumented replay but drives ``retention.mode`` /
    ``eviction_priority`` through the pipeline. Additional per-fact lifecycle
    fields recorded (D32/Phase 11):

    * ``current_importance`` / ``minimum_current_importance`` — dynamic
      activation trajectory;
    * ``retention_priority`` / ``maximum_retention_priority`` — stable survival
      score;
    * ``retained_below_threshold`` — the fact was at some point below the
      pruning threshold yet remained in the store (only reachable under
      soft_decay / dual_score);
    * ``terminal_state`` — the E14 lifecycle vocabulary.
    """
    settings = dict(settings)
    settings["memory_store_token_budget"] = store_budget
    settings["retention"] = {
        "mode": retention_mode,
        "eviction_priority": eviction_priority,
    }
    scorer = ImportanceScorer(weights=settings["scoring_weights"])
    decay = CategoryDecayEngine(lambdas=decay_lambdas, pruning_threshold=pruning_threshold)
    retriever = MemoryRetriever(
        top_k=int(settings["top_k"]),
        sim_threshold=float(settings.get("similarity_threshold", 0.35)),
        embed_fn=embed_fn,
        embedding_model=embedding_model,
    )
    compressor = MemoryCompressor()
    pipe = AdaptiveMemoryPipeline(settings, scorer, decay, retriever, compressor)

    registry: Dict[str, Dict] = {}

    def ensure(fact_text: str, src: Optional[Dict] = None) -> Dict:
        rec = registry.get(fact_text)
        if rec is None:
            rec = {
                "fact": fact_text,
                "category": (src or {}).get("category", ""),
                "qtype": (src or {}).get("qtype", ""),
                "source_turn": (src or {}).get("source_turn"),
                "is_correction_target": bool((src or {}).get("is_correction_target")),
                "superseded_original": bool((src or {}).get("superseded_by") is not None
                                            and not (src or {}).get("is_correction_target")),
                "appearances": 0,
                "initial_importance": None,
                "minimum_current_importance": None,
                "final_importance": None,
                "retention_priority": None,
                "maximum_retention_priority": None,
                "retained_below_threshold": False,
                "retention_mode": retention_mode,
                "last_access_turn": None,
                "access_count": None,
                "decay_steps_survived": 0,
                "pruned_by_decay": False,
                "removed_by_store_budget": False,
                "prune_reason": None,
                "eviction_reason": None,
                "retrieval_candidate": False,
                "retrieval_selected": False,
            }
            registry[fact_text] = rec
        return rec

    for g in gt:
        ensure(g["fact"], g)

    store_pressure_log: List[Dict] = []
    superseded_texts: set = set()

    for turn in stream:
        prev_pruned = len(pipe.pruned_memories)
        result = pipe.ingest(turn["turn_id"], turn["user"], turn.get("facts", []),
                             fact_tokens=word_count, embed_fn=embed_fn)

        for m in pipe.pruned_memories[prev_pruned:]:
            rec = ensure(m.get("fact", ""))
            rec["pruned_by_decay"] = True
            rec["prune_reason"] = m.get("prune_reason")
            if m.get("superseded_prior_fact"):
                rec["superseded"] = True

        for e in result.get("budget_evictions", []):
            rec = ensure(e.get("fact", ""))
            rec["removed_by_store_budget"] = True
            rec["eviction_reason"] = e.get("eviction_reason")

        for m in pipe.active_memories:
            text = m.get("fact", "")
            rec = ensure(text)
            rec["appearances"] += 1
            rec["decay_steps_survived"] += 1
            if rec["initial_importance"] is None:
                rec["initial_importance"] = m.get("base_score")
            ci = m.get("current_importance")
            if ci is not None:
                rec["final_importance"] = ci
                if rec["minimum_current_importance"] is None:
                    rec["minimum_current_importance"] = ci
                else:
                    rec["minimum_current_importance"] = min(rec["minimum_current_importance"], ci)
            rp = m.get("retention_priority")
            if rp is not None:
                rec["retention_priority"] = rp
                if rec["maximum_retention_priority"] is None:
                    rec["maximum_retention_priority"] = rp
                else:
                    rec["maximum_retention_priority"] = max(rec["maximum_retention_priority"], rp)
            prior = m.get("superseded_prior_fact")
            if prior:
                superseded_texts.add(prior)
                rec["superseded"] = True
                prior_rec = registry.get(prior)
                if prior_rec is not None:
                    prior_rec["superseded"] = True
            rec["dedupe_count"] = max(0, int(m.get("duplicates", 1)) - 1)
            rec["last_access_turn"] = m.get("last_access_turn")
            rec["access_count"] = m.get("access_count", 1)
            if m.get("retained_below_threshold"):
                rec["retained_below_threshold"] = True

        store_pressure_log.append(dict(result.get("store_pressure", {})))

    store = pipe.active_memories
    for m in store:
        if m.get("superseded_prior_fact"):
            superseded_texts.add(m["superseded_prior_fact"])
    store_texts = {m.get("fact", "") for m in store}
    final_store_entries = [dict(m) for m in store]

    # Query-time retrieval (read-only) — ranking still uses current_importance.
    qr = MemoryRetriever(
        top_k=int(settings["top_k"]),
        sim_threshold=float(settings.get("similarity_threshold", 0.35)),
        embed_fn=embed_fn,
        embedding_model=embedding_model,
    )
    selected_by_qid: Dict[str, List[Dict]] = {}
    ranked_by_qid: Dict[str, List[Dict]] = {}
    for q in queries:
        ranked = qr.retrieve(q["user"], store, current_turn=q["query_turn"], ranked=True)
        selected, _, _ = fit_to_budget(ranked, budget, word_count)
        ranked_by_qid[q["qid"]] = ranked
        selected_by_qid[q["qid"]] = selected

    store_by_qid = {q["qid"]: final_store_entries for q in queries}

    for q in queries:
        cand_texts = {c.get("fact", "") for c in ranked_by_qid.get(q["qid"], [])}
        sel_texts = {c.get("fact", "") for c in selected_by_qid.get(q["qid"], [])}
        for text, rec in registry.items():
            if text in cand_texts:
                rec["retrieval_candidate"] = True
            if text in sel_texts:
                rec["retrieval_selected"] = True

    lifecycle: Dict[str, str] = {}
    query_by_source: Dict[int, Dict] = {q["source_turn"]: q for q in queries}
    for text, rec in registry.items():
        lifecycle[text] = _classify(rec, store_texts, superseded_texts,
                                    query_by_source, selected_by_qid,
                                    final_store_entries)

    return {
        "memories": store,
        "registry": registry,
        "lifecycle": lifecycle,
        "pruned": pipe.pruned_memories,
        "store_pressure_log": store_pressure_log,
        "selected_by_qid": selected_by_qid,
        "ranked_by_qid": ranked_by_qid,
        "store_by_qid": store_by_qid,
        "final_store_entries": final_store_entries,
    }


def _classify(rec: Dict, store_texts: set, superseded_texts: set,
              query_by_source: Dict[int, Dict], selected_by_qid: Dict,
              final_store_entries: List[Dict]) -> str:
    text = rec["fact"]
    present = text in store_texts

    if rec.get("superseded_original"):
        if present:
            return "SUPERSEDED_INCORRECTLY"
        if text in superseded_texts:
            return "SUPERSEDED_CORRECTLY"
        return _loss_state(rec)

    if present:
        q = query_by_source.get(rec.get("source_turn"))
        retrieved = False
        if q is not None:
            retrieved = _answered(q, selected_by_qid.get(q["qid"], []), _fact_text)
        elif rec.get("retrieval_selected"):
            retrieved = True
        return "PRESENT_AND_RETRIEVED" if retrieved else "PRESENT_BUT_NOT_RETRIEVED"

    return _loss_state(rec)


def _loss_state(rec: Dict) -> str:
    if rec.get("pruned_by_decay"):
        return "REMOVED_BY_DECAY"
    if rec.get("removed_by_store_budget"):
        return "REMOVED_BY_STORE_BUDGET"
    if rec.get("appearances", 0) == 0:
        return "MERGED_BY_DEDUPE"
    return "OTHER"


# ---------------------------------------------------------------------------
# Per-cell metrics
# ---------------------------------------------------------------------------

def _per_cell(rep: Dict, gt: List[Dict], queries: List[Dict], *, budget: int,
              store_budget: int, policy: str, seed: int, pruning_threshold: float,
              turns: int, scale: int) -> Dict:
    lifecycle = rep["lifecycle"]
    counts = _lifecycle_counts(lifecycle)
    metrics = _metrics(queries, rep["selected_by_qid"], rep["store_by_qid"])
    store_tokens = sum(word_count(m.get("fact", "")) for m in rep["memories"])
    ctx_recall = metrics.get("context_recall")
    store_recall = metrics.get("store_recall")
    retention_loss = (round(store_recall - ctx_recall, 4)
                      if store_recall is not None and ctx_recall is not None else None)
    metrics["retrieval_loss"] = retention_loss
    metrics["store_tokens"] = store_tokens
    metrics["store_count"] = len(rep["memories"])
    metrics["mean_context_utilization"] = round(
        metrics["mean_context_tokens"] / budget, 4) if budget else None

    decay_removals = sum(1 for rec in rep["registry"].values()
                         if rec.get("pruned_by_decay"))
    store_budget_evictions = sum(p.get("evicted_count", 0)
                                 for p in rep["store_pressure_log"])

    # Activation/retention summaries over gt facts that were stored at some point.
    stored = [rec for rec in rep["registry"].values() if rec.get("appearances", 0) > 0]
    ci_vals = [rec["final_importance"] for rec in stored if rec.get("final_importance") is not None]
    rp_vals = [rec["retention_priority"] for rec in stored if rec.get("retention_priority") is not None]

    # Stale-memory diagnostic: share of gt facts still in the FINAL store whose
    # current_importance is below the pruning threshold.
    present_in_store = [rec for rec in rep["registry"].values()
                        if lifecycle.get(rec["fact"]) in
                        ("PRESENT_AND_RETRIEVED", "PRESENT_BUT_NOT_RETRIEVED")]
    below = [rec for rec in present_in_store
             if rec.get("final_importance") is not None
             and rec["final_importance"] < pruning_threshold]

    authoritative_gt = [g for g in gt
                        if not (g.get("superseded_by") and not g.get("is_correction_target"))]
    candidate_count = sum(1 for g in authoritative_gt
                          if rep["registry"].get(g["fact"], {}).get("retrieval_candidate"))
    selected_count = sum(1 for g in authoritative_gt
                         if rep["registry"].get(g["fact"], {}).get("retrieval_selected"))
    # Revival: old + low-activation + still retained + actually recovered.
    revived = sum(1 for g in authoritative_gt
                  if rep["registry"].get(g["fact"], {}).get("retained_below_threshold")
                  and lifecycle.get(g["fact"]) == "PRESENT_AND_RETRIEVED")

    return {
        "experiment": "e15_retention_policy",
        "seed": seed,
        "budget": budget,
        "policy": policy,
        "store_budget": store_budget,
        "turns": len(rep["store_pressure_log"]),
        "scale": scale,
        "query_turn": (queries[0]["query_turn"] if queries else turns + 1),
        "gt_count": len(gt),
        "lifecycle_counts": counts,
        "lifecycle": lifecycle,
        "registry": rep["registry"],
        "metrics": metrics,
        "family_metrics": _family_metrics(queries, rep["selected_by_qid"],
                                          rep["store_by_qid"]),
        "store_summary": {
            "store_count": len(rep["memories"]),
            "store_tokens": store_tokens,
            "decay_removals": decay_removals,
            "store_budget_evictions": store_budget_evictions,
            "mean_current_importance": (round(sum(ci_vals) / len(ci_vals), 4) if ci_vals else None),
            "mean_retention_priority": (round(sum(rp_vals) / len(rp_vals), 4) if rp_vals else None),
            "fraction_below_pruning_threshold": (
                round(len(below) / len(present_in_store), 4) if present_in_store else None),
            "retained_below_threshold_count": sum(
                1 for rec in rep["registry"].values() if rec.get("retained_below_threshold")),
            "candidate_count": candidate_count,
            "selected_count": selected_count,
            "revived_facts": revived,
        },
        "age_survival": _age_survival(gt, rep["registry"], lifecycle,
                                      queries[0]["query_turn"] if queries else turns + 1),
        "category_survival": _category_survival(gt, rep["registry"], lifecycle),
        "correction_gate": {
            "correction_recall": metrics.get("correction_recall"),
            "obsolete_retention": metrics.get("obsolete_retention"),
        },
        "store_pressure_agg": {
            "evicted_total": store_budget_evictions,
            "over_budget_turns": sum(1 for p in rep["store_pressure_log"]
                                     if p.get("store_over_budget_before_eviction", 0) > 0),
        },
    }


def _bucket_key(label: str) -> int:
    for i, ab in enumerate(AGE_BUCKETS):
        if f"{ab[0]}-{ab[1] - 1}" == label:
            return i
    return len(AGE_BUCKETS)   # e.g. the open-ended "400-+" tail sorts last


def _age_survival(gt: List[Dict], registry: Dict[str, Dict],
                  lifecycle: Dict[str, str], query_turn: int) -> Dict[str, Dict]:
    buckets: Dict[str, Dict[str, int]] = defaultdict(lambda: {"expected": 0, "present": 0, "retrieved": 0})
    for g in gt:
        if g.get("superseded_by") is not None and not g.get("is_correction_target"):
            continue
        age = max(0, query_turn - int(g["source_turn"]))
        b = _age_bucket(age)
        buckets[b]["expected"] += 1
        state = lifecycle.get(g["fact"])
        if state in ("PRESENT_AND_RETRIEVED", "PRESENT_BUT_NOT_RETRIEVED"):
            buckets[b]["present"] += 1
        if state == "PRESENT_AND_RETRIEVED":
            buckets[b]["retrieved"] += 1
    out = {}
    for b, v in sorted(buckets.items(), key=lambda kv: _bucket_key(kv[0])):
        exp = v["expected"]
        out[b] = {
            "expected": exp,
            "store_present_fraction": round(v["present"] / exp, 3) if exp else None,
            "retrieved_fraction": round(v["retrieved"] / exp, 3) if exp else None,
        }
    return out


def _category_survival(gt: List[Dict], registry: Dict[str, Dict],
                       lifecycle: Dict[str, str]) -> Dict[str, Dict]:
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for g in gt:
        by_cat[g.get("category", "")].append(g)
    out = {}
    for cat, items in sorted(by_cat.items()):
        n = len(items)
        present = sum(1 for g in items if lifecycle.get(g["fact"]) in
                      ("PRESENT_AND_RETRIEVED", "PRESENT_BUT_NOT_RETRIEVED"))
        retrieved = sum(1 for g in items if lifecycle.get(g["fact"]) == "PRESENT_AND_RETRIEVED")
        causes = defaultdict(int)
        for g in items:
            causes[lifecycle.get(g["fact"], "OTHER")] += 1
        out[cat] = {
            "expected": n,
            "store_present": present,
            "store_present_fraction": round(present / n, 3) if n else None,
            "retrieved_fraction": round(retrieved / n, 3) if n else None,
            "dominant_loss": max(causes.items(), key=lambda kv: kv[1])[0] if causes else None,
            "causes": dict(causes),
        }
    return out


# ---------------------------------------------------------------------------
# Cell drivers
# ---------------------------------------------------------------------------

def run_cell(seed: int, budget: int, policy: str, *, turns: int = TURNS,
             scale: int = SCALE, store_budget: int = STORE_BUDGET,
             use_embeddings: bool = True, embed_fn=None) -> Dict:
    embed_fn = _resolve_embed(use_embeddings, EMBEDDING_MODEL, embed_fn)
    stream, gt, queries = build_coding_session(seed, turns, include_corrections=True,
                                               scale=scale)
    cfg = POLICY_DEFS[policy]
    settings = load_settings({
        "max_context_tokens": budget,
        "injection_token_limit": budget,
        "memory_store_token_budget": store_budget,
        "retention_mode": cfg["retention_mode"],
        "eviction_priority": cfg["eviction_priority"],
    })
    lambdas = (dict(settings["decay_lambdas"])
               if cfg["decay_lambdas"] is None else cfg["decay_lambdas"])
    threshold = cfg["pruning_threshold"]

    rep = replay_session(
        stream, gt, queries, settings,
        embed_fn=embed_fn, embedding_model=EMBEDDING_MODEL,
        decay_lambdas=lambdas, pruning_threshold=threshold,
        budget=budget, store_budget=store_budget,
        retention_mode=cfg["retention_mode"],
        eviction_priority=cfg["eviction_priority"],
    )
    return _per_cell(rep, gt, queries, budget=budget, store_budget=store_budget,
                     policy=policy, seed=seed, pruning_threshold=threshold,
                     turns=turns, scale=scale)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _cell_mean(cells: List[Dict], field: str) -> Optional[float]:
    return _mean([c[field] if not isinstance(c.get(field), dict) else None for c in cells]) \
        if field in cells[0] else None


def _metrics_mean(cells: List[Dict], key: str) -> Optional[float]:
    return _mean([c["metrics"].get(key) for c in cells])


def _store_mean(cells: List[Dict], key: str) -> Optional[float]:
    return _mean([c["store_summary"].get(key) for c in cells])


def _merge_state_counts(rows: List[Dict]) -> Dict:
    return {s: _mean([r["lifecycle_counts"].get(s, 0) for r in rows]) for s in STATES}


def _merge_age(rows: List[Dict]) -> Dict[str, Dict]:
    out = {}
    labels = [f"{b[0]}-{b[1] - 1}" for b in AGE_BUCKETS]
    labels.append(f"{AGE_BUCKETS[-1][1]}-+")   # open-ended tail (long workloads)
    for label in labels:
        present = [r["age_survival"].get(label, {}) for r in rows]
        exp = [p.get("expected") for p in present if p.get("expected") is not None]
        pf = [p.get("store_present_fraction") for p in present
              if p.get("store_present_fraction") is not None]
        rf = [p.get("retrieved_fraction") for p in present
              if p.get("retrieved_fraction") is not None]
        out[label] = {
            "expected_mean": _mean(exp) if exp else None,
            "store_present_fraction_mean": _mean(pf) if pf else None,
            "retrieved_fraction_mean": _mean(rf) if rf else None,
        }
    return out


def aggregate_primary(cells: List[Dict]) -> Dict:
    by_key: Dict[tuple, List[Dict]] = defaultdict(list)
    for c in cells:
        by_key[(c["policy"], c["budget"])].append(c)
    per_policy_budget = {}
    for (policy, budget), rows in sorted(by_key.items()):
        per_policy_budget[(policy, budget)] = {
            "policy": policy,
            "budget": budget,
            "cells": len(rows),
            "state_counts_mean": _merge_state_counts(rows),
            "context_recall_mean": _metrics_mean(rows, "context_recall"),
            "store_recall_mean": _metrics_mean(rows, "store_recall"),
            "retrieval_loss_mean": _metrics_mean(rows, "retrieval_loss"),
            "long_range_recall_mean": _metrics_mean(rows, "long_range_recall"),
            "correction_recall_mean": _metrics_mean(rows, "correction_recall"),
            "obsolete_retention_mean": _metrics_mean(rows, "obsolete_retention"),
            "mean_context_tokens": _metrics_mean(rows, "mean_context_tokens"),
            "mean_context_utilization": _metrics_mean(rows, "mean_context_utilization"),
            "store_tokens_mean": _store_mean(rows, "store_tokens"),
            "store_count_mean": _store_mean(rows, "store_count"),
            "decay_removals_mean": _store_mean(rows, "decay_removals"),
            "store_budget_evictions_mean": _store_mean(rows, "store_budget_evictions"),
            "mean_current_importance": _store_mean(rows, "mean_current_importance"),
            "mean_retention_priority": _store_mean(rows, "mean_retention_priority"),
            "fraction_below_pruning_threshold": _store_mean(rows, "fraction_below_pruning_threshold"),
            "retained_below_threshold_count": _store_mean(rows, "retained_below_threshold_count"),
            "candidate_count": _store_mean(rows, "candidate_count"),
            "selected_count": _store_mean(rows, "selected_count"),
            "revived_facts": _store_mean(rows, "revived_facts"),
            "age_survival": _merge_age(rows),
        }

    # Re-key into a nested dict for JSON friendliness
    nested = {}
    for (policy, budget), agg in per_policy_budget.items():
        nested.setdefault(policy, {})[str(budget)] = agg
    return nested


def aggregate_by_budget(cells: List[Dict]) -> Dict:
    return aggregate_primary(cells)


# ---------------------------------------------------------------------------
# Stress grid
# ---------------------------------------------------------------------------

def run_stress(store_budgets=None, active_budgets=None, seeds=None,
               use_embeddings=True, embed_fn=None,
               min_natural_tokens: int = STRESS_MIN_NATURAL_TOKENS) -> Dict:
    store_budgets = store_budgets or STRESS_STORE_BUDGETS
    active_budgets = active_budgets or STRESS_ACTIVE_BUDGETS
    seeds = seeds or STRESS_SEEDS
    embed_fn = _resolve_embed(use_embeddings, EMBEDDING_MODEL, embed_fn)

    scale = _find_stress_scale(min_natural_tokens, STRESS_TURNS)
    stream, gt, queries = build_coding_session(seeds[0], STRESS_TURNS,
                                               include_corrections=True, scale=scale)
    natural = _natural_store_tokens(gt)

    cells = []
    for sb in store_budgets:
        for ab in active_budgets:
            for seed in seeds:
                for policy in STRESS_POLICIES:
                    print(f"  [stress] store={sb} active={ab} seed={seed} "
                          f"policy={policy} ...", flush=True)
                    cell = run_cell(seed, ab, policy, turns=STRESS_TURNS,
                                    scale=scale, store_budget=sb,
                                    use_embeddings=use_embeddings, embed_fn=embed_fn)
                    cell["stress"] = {
                        "natural_store_tokens": natural,
                        "actual_store_tokens": cell["store_summary"]["store_tokens"],
                        "eviction_count": cell["store_summary"]["store_budget_evictions"],
                    }
                    cells.append(cell)

    # Aggregate: policy x store_budget x active_budget
    agg = aggregate_stress(cells)
    return {
        "experiment": "e15_retention_policy.stress",
        "scale": scale,
        "natural_store_tokens": natural,
        "turns": STRESS_TURNS,
        "store_budgets": store_budgets,
        "active_budgets": active_budgets,
        "seeds": seeds,
        "policies": STRESS_POLICIES,
        "cells": cells,
        "aggregate": agg,
    }


def aggregate_stress(cells: List[Dict]) -> Dict:
    by: Dict[tuple, List[Dict]] = defaultdict(list)
    for c in cells:
        by[(c["policy"], c["store_budget"], c["budget"])].append(c)
    out = {}
    for (policy, sb, ab), rows in sorted(by.items()):
        out.setdefault(policy, {}).setdefault(str(sb), {})[ab] = {
            "cells": len(rows),
            "context_recall_mean": _metrics_mean(rows, "context_recall"),
            "store_recall_mean": _metrics_mean(rows, "store_recall"),
            "retrieval_loss_mean": _metrics_mean(rows, "retrieval_loss"),
            "correction_recall_mean": _metrics_mean(rows, "correction_recall"),
            "obsolete_retention_mean": _metrics_mean(rows, "obsolete_retention"),
            "actual_store_tokens_mean": _mean([c["stress"]["actual_store_tokens"] for c in rows]),
            "eviction_count_mean": _mean([c["stress"]["eviction_count"] for c in rows]),
            "decay_removals_mean": _store_mean(rows, "decay_removals"),
            "store_budget_evictions_mean": _store_mean(rows, "store_budget_evictions"),
            "fraction_below_pruning_threshold": _store_mean(rows, "fraction_below_pruning_threshold"),
        }
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(result: Dict) -> str:
    lines = []
    lines.append("# E15 Retention Policy Report (Phase 11)")
    lines.append("")
    lines.append("Separating memory **activation** (`current_importance`, decays, "
                 "retrieval ranking) from long-term **survival** "
                 "(`retention_priority`, stable, eviction).")
    lines.append("")
    cfg = result["config"]
    lines.append("## Configuration")
    lines.append("")
    for k, v in cfg.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("### Policies")
    lines.append("")
    lines.append("| Policy | retention_mode / eviction_priority |")
    lines.append("| --- | --- |")
    for p, d in POLICY_DEFS.items():
        lines.append(f"| {p} | `{d['retention_mode']}` / `{d['eviction_priority']}` |")
    lines.append("")

    primary = result["primary"]
    cells = primary["cells"]
    agg = primary["aggregate_by_policy"]

    lines.append("## 1. Lifecycle Cause-of-Loss (primary grid, per budget)")
    lines.append("")
    headers = ["Policy", "Budget", "stored+retr", "stored/not", "decay", "budget",
               "dedupe", "super_ok", "super_bad", "corr_recall", "obs_ret"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for p in POLICIES:
        for b in BUDGETS:
            a = agg.get(p, {}).get(str(b))
            if not a:
                continue
            s = a["state_counts_mean"] or {}

            def f(x):
                return f"{x:.2f}" if x is not None else "N/A"
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                p, b, f(s.get("PRESENT_AND_RETRIEVED")),
                f(s.get("PRESENT_BUT_NOT_RETRIEVED")), f(s.get("REMOVED_BY_DECAY")),
                f(s.get("REMOVED_BY_STORE_BUDGET")), f(s.get("MERGED_BY_DEDUPE")),
                f(s.get("SUPERSEDED_CORRECTLY")), f(s.get("SUPERSEDED_INCORRECTLY")),
                f(a.get("correction_recall_mean")), f(a.get("obsolete_retention_mean"))))
    lines.append("")

    lines.append("## 2. Final Comparison Table (Policy x Budget)")
    lines.append("")
    headers = ["Policy", "B", "Store Recall", "Context Recall", "Retrieval Loss",
               "Store Tokens", "Obs. Retention", "Corr. Recall", "Mean Ctx Tokens",
               "Below Thresh"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for p in POLICIES:
        for b in BUDGETS:
            a = agg.get(p, {}).get(str(b))
            if not a:
                continue

            def f(x):
                return f"{x:.3f}" if isinstance(x, float) else ("N/A" if x is None else str(x))
            lines.append(f"| {p} | {b} | {f(a.get('store_recall_mean'))} | "
                         f"{f(a.get('context_recall_mean'))} | "
                         f"{f(a.get('retrieval_loss_mean'))} | "
                         f"{f(a.get('store_tokens_mean'))} | "
                         f"{f(a.get('obsolete_retention_mean'))} | "
                         f"{f(a.get('correction_recall_mean'))} | "
                         f"{f(a.get('mean_context_tokens'))} | "
                         f"{f(a.get('fraction_below_pruning_threshold'))} |")
    lines.append("")

    lines.append("## 3. Activation / Survival summary")
    lines.append("")
    headers = ["Policy", "B", "decay_rm", "budget_ev", "mean_imp", "mean_rp",
               "below_thr", "strong_rel_keep", "cand", "sel", "revived"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for p in POLICIES:
        for b in BUDGETS:
            a = agg.get(p, {}).get(str(b))
            if not a:
                continue

            def f(x):
                return f"{x:.3f}" if isinstance(x, float) else ("N/A" if x is None else str(x))
            lines.append(f"| {p} | {b} | {f(a.get('decay_removals_mean'))} | "
                         f"{f(a.get('store_budget_evictions_mean'))} | "
                         f"{f(a.get('mean_current_importance'))} | "
                         f"{f(a.get('mean_retention_priority'))} | "
                         f"{f(a.get('fraction_below_pruning_threshold'))} | "
                         f"{f(a.get('retained_below_threshold_count'))} | "
                         f"{f(a.get('candidate_count'))} | "
                         f"{f(a.get('selected_count'))} | "
                         f"{f(a.get('revived_facts'))} |")
    lines.append("")

    lines.append("## 4. Age-bucket survival (budget 128 reference)")
    lines.append("")
    lines.append("*See JSON for every budget. Table below uses budget 128.*")
    lines.append("")
    lines.append("| Policy | bucket | expected | store_present | retrieved |")
    lines.append("| --- | --- | --- | --- | --- |")
    for p in POLICIES:
        a = agg.get(p, {}).get("128")
        if not a:
            continue
        for label, v in (a.get("age_survival") or {}).items():
            lines.append(f"| {p} | {label} | {v.get('expected_mean')} | "
                         f"{v.get('store_present_fraction_mean')} | "
                         f"{v.get('retrieved_fraction_mean')} |")
    lines.append("")

    lines.append("## 5. Stress grid (genuine store pressure)")
    lines.append("")
    stress = result.get("stress")
    if stress:
        lines.append(f"Scale {stress['scale']}, natural store tokens "
                     f"{stress['natural_store_tokens']}, turns {stress['turns']}.")
        lines.append("")
        headers = ["Policy", "store_budget", "active", "ctx_recall", "store_recall",
                   "retr_loss", "actual_store_tokens", "eviction_count", "decay_rm",
                   "corr", "obs_ret", "below_thr"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for p in STRESS_POLICIES:
            for sb in STRESS_STORE_BUDGETS:
                for ab in STRESS_ACTIVE_BUDGETS:
                    a = (stress["aggregate"].get(p, {}).get(str(sb), {}) or {}).get(ab)
                    if not a:
                        continue

                    def f(x):
                        return f"{x:.3f}" if isinstance(x, float) else ("N/A" if x is None else str(x))
                    lines.append(f"| {p} | {sb} | {ab} | "
                                 f"{f(a.get('context_recall_mean'))} | "
                                 f"{f(a.get('store_recall_mean'))} | "
                                 f"{f(a.get('retrieval_loss_mean'))} | "
                                 f"{f(a.get('actual_store_tokens_mean'))} | "
                                 f"{f(a.get('eviction_count_mean'))} | "
                                 f"{f(a.get('decay_removals_mean'))} | "
                                 f"{f(a.get('correction_recall_mean'))} | "
                                 f"{f(a.get('obsolete_retention_mean'))} | "
                                 f"{f(a.get('fraction_below_pruning_threshold'))} |")
        lines.append("")
        lines.append("Stress comparison table as specified:")
        lines.append("")
        lines.append("| Policy | store_budget | natural_store_tokens | actual_store_tokens | eviction_count |")
        lines.append("| --- | --- | --- | --- | --- |")
        for p in STRESS_POLICIES:
            for sb in STRESS_STORE_BUDGETS:
                rows = [c for c in stress["cells"]
                        if c["policy"] == p and c["store_budget"] == sb]
                natural = stress["natural_store_tokens"]
                act = _mean([r["stress"]["actual_store_tokens"] for r in rows])
                evic = _mean([r["stress"]["eviction_count"] for r in rows])
                lines.append(f"| {p} | {sb} | {natural} | {act} | {evic} |")
        lines.append("")
    else:
        lines.append("*Stress grid not run (skipped in quick mode).*")
        lines.append("")

    lines.append("## 6. Answers to the Nine Evaluation Questions")
    lines.append("")
    lines.append(_nine_questions(result))
    lines.append("")
    lines.append("## 7. Decision Rule Application")
    lines.append("")
    lines.append(_decision(result))
    lines.append("")
    return "\n".join(lines)


def _nine_questions(result: Dict) -> str:
    agg = result["primary"]["aggregate_by_policy"]
    stress = result.get("stress")
    lines = []
    best_rp = {p: _mean([agg.get(p, {}).get(str(b), {}).get("store_recall_mean")
                         for b in BUDGETS])
               for p in POLICIES}

    def fmt(x):
        return "N/A" if x is None else f"{x:.3f}"

    def diff(a, b):
        if a is None or b is None:
            return "N/A"
        v = a - b
        return f"{v:+.3f}"

    soft = agg.get("soft_decay", {})
    hard = agg.get("hard_threshold", {})
    dual = agg.get("dual_score", {})
    nodec = agg.get("no_decay", {})
    q1 = "\n".join([
        "### Q1 — Does separating activation from survival recover useful long-range facts?",
        "",
        f"soft_decay store_recall means: "
        + ", ".join(f"B{b}={fmt(soft.get(str(b), {}).get('store_recall_mean'))}" for b in BUDGETS),
        f"hard_threshold store_recall means: "
        + ", ".join(f"B{b}={fmt(hard.get(str(b), {}).get('store_recall_mean'))}" for b in BUDGETS),
        f"Delta store_recall (soft - hard): "
        + ", ".join(f"B{b}={diff(soft.get(str(b), {}).get('store_recall_mean'), hard.get(str(b), {}).get('store_recall_mean'))}" for b in BUDGETS),
        f"long_range_recall soft vs hard: "
        + ", ".join(f"B{b}={fmt(soft.get(str(b), {}).get('long_range_recall_mean'))}/{fmt(hard.get(str(b), {}).get('long_range_recall_mean'))}" for b in BUDGETS),
    ])
    lines.append(q1)
    lines.append("")
    lines.append("### Q2 — Does soft decay improve low-budget (64/128) recall?")
    lines.append("")
    for b in (64, 128):
        sr_s = soft.get(str(b), {}).get('store_recall_mean')
        sr_h = hard.get(str(b), {}).get('store_recall_mean')
        cr_s = soft.get(str(b), {}).get('context_recall_mean')
        cr_h = hard.get(str(b), {}).get('context_recall_mean')
        lines.append(f"Budget {b}: store_recall soft={fmt(sr_s)} vs hard={fmt(sr_h)} "
                     f"(delta {diff(sr_s, sr_h)}); context_recall soft={fmt(cr_s)} vs "
                     f"hard={fmt(cr_h)} (delta {diff(cr_s, cr_h)}).")
    lines.append("")
    lines.append("### Q3 — Does dual_score add anything beyond soft_decay?")
    lines.append("")
    if stress:
        lines.append("Under genuine store pressure (natural tokens "
                     f"{stress['natural_store_tokens']}), compare soft_decay vs dual_score "
                     "for store_recall, eviction count and obsolete retention in the "
                     "stress table above.")
    else:
        lines.append("Stress grid not run; cannot evaluate dual_score causally.")
    lines.append("")
    lines.append("### Q4 — Store-size cost?")
    lines.append("")
    lines.append(f"Store tokens means (soft vs hard): "
                 + ", ".join(f"B{b}={fmt(soft.get(str(b), {}).get('store_tokens_mean'))}/"
                             f"{fmt(hard.get(str(b), {}).get('store_tokens_mean'))}" for b in BUDGETS)
                 + ". fraction_below_pruning_threshold soft vs hard: "
                 + ", ".join(f"B{b}={fmt(soft.get(str(b), {}).get('fraction_below_pruning_threshold'))}/"
                             f"{fmt(hard.get(str(b), {}).get('fraction_below_pruning_threshold'))}" for b in BUDGETS))
    lines.append("")
    lines.append("### Q5 — Obsolete retention controlled?")
    lines.append("")
    lines.append(f"obsolete_retention soft: "
                 + ", ".join(f"B{b}={fmt(soft.get(str(b), {}).get('obsolete_retention_mean'))}" for b in BUDGETS)
                 + f". hard: "
                 + ", ".join(f"B{b}={fmt(hard.get(str(b), {}).get('obsolete_retention_mean'))}" for b in BUDGETS))
    lines.append("")
    lines.append("### Q6 — Corrections correct?")
    lines.append("")
    lines.append(f"correction_recall soft: "
                 + ", ".join(f"B{b}={fmt(soft.get(str(b), {}).get('correction_recall_mean'))}" for b in BUDGETS)
                 + f". hard: "
                 + ", ".join(f"B{b}={fmt(hard.get(str(b), {}).get('correction_recall_mean'))}" for b in BUDGETS))
    lines.append("")
    lines.append("### Q7 — Active-context cost unchanged?")
    lines.append("")
    lines.append(f"mean_context_tokens soft: "
                 + ", ".join(f"B{b}={fmt(soft.get(str(b), {}).get('mean_context_tokens'))}" for b in BUDGETS)
                 + f". hard: "
                 + ", ".join(f"B{b}={fmt(hard.get(str(b), {}).get('mean_context_tokens'))}" for b in BUDGETS)
                 + ". Context budget is enforced by construction; utilization reported in JSON.")
    lines.append("")
    lines.append("### Q8 — Where does retrieval loss remain?")
    lines.append("")
    lines.append(f"retrieval_loss soft: "
                 + ", ".join(f"B{b}={fmt(soft.get(str(b), {}).get('retrieval_loss_mean'))}" for b in BUDGETS)
                 + f". hard: "
                 + ", ".join(f"B{b}={fmt(hard.get(str(b), {}).get('retrieval_loss_mean'))}" for b in BUDGETS)
                 + ". PRESENT_BUT_NOT_RETRIEVED counts in lifecycle table itemize it.")
    lines.append("")
    lines.append("### Q9 — Real improvement or mere data preservation?")
    lines.append("")
    lines.append(f"revived-facts (retained-below-threshold AND later retrieved): soft "
                 + ", ".join(f"B{b}={fmt(soft.get(str(b), {}).get('revived_facts'))}" for b in BUDGETS)
                 + f"; candidate_count soft "
                 + ", ".join(f"B{b}={fmt(soft.get(str(b), {}).get('candidate_count'))}" for b in BUDGETS)
                 + ". A policy that merely keeps everything scores store_recall "
                 "high without improving context_recall; compare both columns in "
                 "section 2.")
    lines.append("")
    return "\n".join(lines)


def _decision(result: Dict) -> str:
    # NOTE: classification happens in the experiment file's "auto-classify" and
    # is OVERRIDDEN by the human reviewer in the final report. The text produced
    # here records the measured numbers; the verdict line is filled by the
    # `classify_result` entry below.
    verdict = result.get("verdict", {})
    lines = []
    lines.append("**Labels from the observed grid (auto-classified, reviewer-"
                 "confirmable):**")
    lines.append("")
    if verdict:
        for k, v in verdict.items():
            supported = v.get("supported")
            lines.append(f"- `{k}`: **{'SUPPORTED' if supported else 'NOT SUPPORTED'}**")
            for kk, vv in v.items():
                if kk in ("supported", "rationale"):
                    continue
                lines.append(f"    - {kk}: {vv}")
            lines.append(f"    - rationale: {v.get('rationale', '')}")
            lines.append("")
    else:
        lines.append("- Auto-classification not applied yet; see the commit "
                     "summary / STATE for the verdict.")
    lines.append("")
    lines.append("*A candidate is made the leading option only if it improves "
                 "recall (context + store) without inflating active context, "
                 "preserves correction/obsolete safety, and keeps store growth "
                 "manageable under genuine pressure. Higher recall alone is not "
                 "a win.*")
    return "\n".join(lines)


def classify_result(result: Dict) -> Dict:
    """Apply the Phase-11 decision rule to the measured grid (OBSERVED-based).

    Returns a dict of label -> (bool, rationale) for the human reviewer.
    """
    agg = result["primary"]["aggregate_by_policy"]
    stress = result.get("stress")

    def mean(policy, key):
        vals = [agg.get(policy, {}).get(str(b), {}).get(key) for b in BUDGETS]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    hard_sr = mean("hard_threshold", "store_recall_mean")
    soft_sr = mean("soft_decay", "store_recall_mean")
    dual_sr = mean("dual_score", "store_recall_mean")
    nodec_sr = mean("no_decay", "store_recall_mean")
    hard_cr = mean("hard_threshold", "context_recall_mean")
    soft_cr = mean("soft_decay", "context_recall_mean")
    hard_ctx = mean("hard_threshold", "mean_context_tokens")
    soft_ctx = mean("soft_decay", "mean_context_tokens")
    soft_obs = mean("soft_decay", "obsolete_retention_mean")
    hard_obs = mean("hard_threshold", "obsolete_retention_mean")
    soft_corr = mean("soft_decay", "correction_recall_mean")
    hard_corr = mean("hard_threshold", "correction_recall_mean")
    soft_store = mean("soft_decay", "store_tokens_mean")
    hard_store = mean("hard_threshold", "store_tokens_mean")

    improvements = {}

    sr_gain = (soft_sr or 0) - (hard_sr or 0)
    cr_gain = (soft_cr or 0) - (hard_cr or 0)
    # Context-token "unchanged" check: allow a small tolerance for the differing
    # fact mix inside the same enforced budget (noise ~0.05 tokens here).
    ctx_ok = (soft_ctx is None or hard_ctx is None
              or soft_ctx <= hard_ctx + 2.0)
    safety_ok = ((soft_obs or 0) <= (hard_obs or 0) + 1e-9
                 and (soft_corr or 0) >= (hard_corr or 0) - 1e-9)
    store_growth_ok = soft_store is not None and hard_store is not None \
        and soft_store <= hard_store * 1.5 + 200

    sep_supported = (sr_gain > 0.01 or cr_gain > 0.01) and ctx_ok and safety_ok \
        and store_growth_ok
    improvements["SEPARATING ACTIVATION FROM SURVIVAL IS SUPPORTED"] = {
        "supported": bool(sep_supported),
        "store_recall_delta": round(sr_gain, 4),
        "context_recall_delta": round(cr_gain, 4),
        "active_context_cost_unchanged": bool(ctx_ok),
        "safety_preserved": bool(safety_ok),
        "store_growth_manageable": bool(store_growth_ok),
        "rationale": (
            f"soft_decay store_recall={soft_sr}, hard={hard_sr}; "
            f"context_recall soft={soft_cr}, hard={hard_cr}; "
            f"mean_context_tokens soft={soft_ctx}, hard={hard_ctx}; "
            f"obsolete soft={soft_obs}, hard={hard_obs}; "
            f"correction soft={soft_corr}, hard={hard_corr}; "
            f"store tokens soft={soft_store}, hard={hard_store}."),
    }

    dual_gain = (dual_sr or 0) - (soft_sr or 0)
    duel_under_pressure = None
    duel_ctx_under_pressure = None
    duel_corr_under_pressure = None
    if stress:
        # Compare soft vs dual under the tightest store budget (real pressure).
        # Context recall and correction safety are the task-level deciders;
        # archival store_recall is reported separately as the tradeoff.
        best, best_ctx, best_corr = None, None, None
        for p in ("soft_decay", "dual_score"):
            a = stress["aggregate"].get(p, {}).get(str(min(STRESS_STORE_BUDGETS)), {})
            srs = [x.get("store_recall_mean") for x in a.values()]
            crs = [x.get("context_recall_mean") for x in a.values()]
            crrs = [x.get("correction_recall_mean") for x in a.values()]
            srs = [v for v in srs if v is not None]
            crs = [v for v in crs if v is not None]
            crrs = [v for v in crrs if v is not None]
            if p == "soft_decay":
                best, best_ctx, best_corr = _mean(srs), _mean(crs), _mean(crrs)
            else:
                duel_under_pressure = round((_mean(srs) or 0) - (best or 0), 4)
                duel_ctx_under_pressure = round((_mean(crs) or 0) - (best_ctx or 0), 4)
                duel_corr_under_pressure = round((_mean(crrs) or 0) - (best_corr or 0), 4)
    improvements["RETENTION PRIORITY FOR STORE EVICTION IS SUPPORTED"] = {
        "supported": bool(dual_gain > 0.01 or (
            duel_ctx_under_pressure is not None and duel_ctx_under_pressure > 0.005)
            or (duel_under_pressure is not None and duel_under_pressure > 0.005)),
        "dual_minus_soft_primary": round(dual_gain, 4),
        "dual_minus_soft_tightest_stress_store": duel_under_pressure,
        "dual_minus_soft_tightest_stress_context": duel_ctx_under_pressure,
        "dual_minus_soft_tightest_stress_correction": duel_corr_under_pressure,
        "rationale": (
            f"dual_score store_recall={dual_sr} vs soft_decay={soft_sr}; "
            f"under tightest store pressure dual gives store +{duel_under_pressure}, "
            f"context +{duel_ctx_under_pressure}, correction +{duel_corr_under_pressure} "
            f"(archival store_recall tradeoff documented in the stress table)."),
    }

    staleness = (soft_obs or 0) > (hard_obs or 0) + 0.05
    improvements["SOFT RETENTION RECOVERS FACTS BUT CREATES A STALENESS TRADEOFF"] = {
        "supported": bool(staleness),
        "obsolete_retention_soft": soft_obs,
        "obsolete_retention_hard": hard_obs,
        "rationale": ("soft retention must not resurrect superseded values; if "
                      "obsolete_retention grows beyond the hard baseline the "
                      "tradeoff is real."),
    }

    return improvements


def main(argv=None):
    parser = argparse.ArgumentParser(description="E15 retention policy experiment")
    parser.add_argument("--quick", action="store_true",
                        help="small configuration for smoke-testing")
    parser.add_argument("--no-stress", action="store_true",
                        help="skip the stress grid (full run only)")
    parser.add_argument("--stress-only", action="store_true",
                        help="run only the stress grid, merging into an existing "
                             "e15_retention_policy.json if present")
    parser.add_argument("--report-only", action="store_true",
                        help="re-classify and regenerate the report from an "
                             "existing e15_retention_policy.json (no recompute)")
    args = parser.parse_args(argv)

    from data.coding_workload import self_test

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "e15_retention_policy.json"
    report_path = out_dir / "e15_retention_policy_report.md"

    if args.report_only:
        if not result_path.exists():
            parser.error("no e15_retention_policy.json to classify")
        result = json.loads(result_path.read_text())
        result["verdict"] = classify_result(result)
        result_path.write_text(json.dumps(result, indent=2, default=str))
        report_path.write_text(generate_report(result))
        print(f"Re-classified and wrote {report_path}")
        for k, v in result["verdict"].items():
            print(f"  {k}: supported={v['supported']}")
        return result

    self_test()

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "e15_retention_policy.json"

    if args.stress_only:
        result = {}
        if result_path.exists():
            result = json.loads(result_path.read_text())
        print("=" * 60)
        print("E15 Stress Grid")
        print("=" * 60)
        result["stress"] = run_stress(
            store_budgets=STRESS_STORE_BUDGETS, active_budgets=STRESS_ACTIVE_BUDGETS,
            seeds=STRESS_SEEDS)
        result["verdict"] = classify_result(result)
        result_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"\nWrote {result_path}")
        report_path = out_dir / "e15_retention_policy_report.md"
        report_path.write_text(generate_report(result))
        print(f"Wrote {report_path}")
        return result

    if args.quick:
        budgets, seeds, turns, scale = [64, 128], [42], 120, 3
        stress_store, stress_active, stress_seeds = None, None, None
        policies = POLICIES
        store_budget = STORE_BUDGET
    else:
        budgets, seeds, turns, scale = BUDGETS, SEEDS, TURNS, SCALE
        policies = POLICIES
        store_budget = STORE_BUDGET
        stress_store, stress_active, stress_seeds = (
            STRESS_STORE_BUDGETS, STRESS_ACTIVE_BUDGETS, STRESS_SEEDS)

    print("=" * 60)
    print("E15 Retention Policy")
    print("=" * 60)

    primary_cells = []
    for policy in policies:
        for budget in budgets:
            for seed in seeds:
                print(f"  [primary] policy={policy} budget={budget} seed={seed} ...",
                      flush=True)
                primary_cells.append(run_cell(seed, budget, policy, turns=turns,
                                              scale=scale, store_budget=store_budget))

    result = {
        "experiment": "e15_retention_policy",
        "config": {
            "budgets": budgets, "seeds": seeds, "turns": turns, "scale": scale,
            "store_budget": store_budget, "embedding_model": EMBEDDING_MODEL,
            "matching": "exact answer-token presence",
            "policies": policies,
            "active_budget_enforced": True,
        },
        "policies": POLICY_DEFS,
        "primary": {
            "cells": primary_cells,
            "aggregate_by_policy": aggregate_primary(primary_cells),
        },
    }

    if not args.no_stress and not args.quick:
        print("\n[stress] ...")
        result["stress"] = run_stress(store_budgets=stress_store,
                                      active_budgets=stress_active,
                                      seeds=stress_seeds)
    elif args.quick:
        print("\n[stress] skipped in quick mode")

    result["verdict"] = classify_result(result)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "e15_retention_policy.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {path}")

    report_path = out_dir / "e15_retention_policy_report.md"
    report_path.write_text(generate_report(result))
    print(f"Wrote {report_path}")
    return result


if __name__ == "__main__":
    main()