"""e14 - Phase 10 Retention Diagnosis.

Pure diagnosis: identify exactly WHY useful task facts are disappearing from the
long-term memory store (before query-time retrieval). No memory-policy changes.
Only instrumentation, experimental toggles that isolate existing mechanisms, and
benchmark/report fixes are permitted.

The experiment replays a deterministic coding session through the production
``AdaptiveMemoryPipeline`` turn by turn and records, for every ground-truth fact,
which internal mechanism (if any) removed it:

    NEVER_STORED  REMOVED_BY_DECAY  REMOVED_BY_STORE_BUDGET  MERGED_BY_DEDUPE
    SUPERSEDED_CORRECTLY  SUPERSEDED_INCORRECTLY
    PRESENT_BUT_NOT_RETRIEVED  PRESENT_AND_RETRIEVED  OTHER

It also reports:
  * dominant cause-of-loss table overall and per budget
  * category survival
  * retention-time curves (8 age buckets)
  * Ablation A: no decay (isolates decay)
  * Ablation B: no pruning threshold (decay applied, nothing deleted)
  * Ablation C: score-component diagnosis (relevance/utility/recency/frequency)
  * Ablation D: write-time salience vs query-time similarity
  * correction/obsolete safety gate for every run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.coding_workload import build_coding_session  # noqa: E402
from experiments.e13_generalization import (  # noqa: E402
    _answer_parts,
    _answered,
    _fact_text,
    _family_metrics,
    _metrics,
    word_count,
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
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_ENDPOINT = "http://localhost:11434"

ABLATION_SEEDS = [42, 43, 44]

# Terminal lifecycle states (mutually exclusive)
STATES = [
    "PRESENT_AND_RETRIEVED",
    "PRESENT_BUT_NOT_RETRIEVED",
    "REMOVED_BY_DECAY",
    "REMOVED_BY_STORE_BUDGET",
    "MERGED_BY_DEDUPE",
    "SUPERSEDED_CORRECTLY",
    "SUPERSEDED_INCORRECTLY",
    "NEVER_STORED",
    "OTHER",
]

# Age buckets (turns since write), inclusive lower bound, exclusive upper
AGE_BUCKETS = [(0, 50), (50, 100), (100, 150), (150, 200),
               (200, 250), (250, 300), (300, 350), (350, 400)]


def _age_bucket(age: int) -> str:
    for lo, hi in AGE_BUCKETS:
        if lo <= age < hi:
            return f"{lo}-{hi - 1}"
    return f"{AGE_BUCKETS[-1][1]}-+"


def _resolve_embed(use_embeddings: bool, embedding_model: str, embed_fn):
    if not use_embeddings:
        return None
    if embed_fn is not None:
        return embed_fn
    from memory_optimizer.embeddings import embed_ollama
    return lambda texts: embed_ollama(texts, model=embedding_model, endpoint=OLLAMA_ENDPOINT)


# ---------------------------------------------------------------------------
# Instrumented replay
# ---------------------------------------------------------------------------

def _instrumented_replay(stream, gt, queries, settings, *, embed_fn,
                         embedding_model, decay_lambdas, pruning_threshold,
                         budget, store_budget) -> Dict:
    """Replay one session, capturing per-fact survival diagnostics.

    The only production-facing instrumentation relied on here is
    ``pipeline.last_store_pressure`` / ``result["budget_evictions"]`` (Phase 10)
    and the ``write_time_salience`` field stamped at ingest. Everything else is
    observation: snapshots of the store taken between turns. No policy change.
    """
    settings = dict(settings)
    settings["memory_store_token_budget"] = store_budget
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

    # Per-fact registry keyed by exact fact text.
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
                "max_importance": None,
                "final_importance": None,
                "dedupe_count": 0,
                "superseded": False,
                "last_access_turn": None,
                "access_count": None,
                "decay_steps_survived": 0,
                "pruned_by_decay": False,
                "removed_by_store_budget": False,
                "prune_reason": None,
                "eviction_reason": None,
                "write_time_salience": None,
                "retrieval_candidate": False,
                "retrieval_selected": False,
                "retrieval_sim": None,
                "rank": None,
                "confidence": None,
            }
            registry[fact_text] = rec
        return rec

    # Seed registry with every ground-truth fact (including superseded originals).
    for g in gt:
        ensure(g["fact"], g)

    store_pressure_log: List[Dict] = []
    superseded_texts: set = set()

    for turn in stream:
        prev_pruned = len(pipe.pruned_memories)
        result = pipe.ingest(turn["turn_id"], turn["user"], turn.get("facts", []),
                             fact_tokens=word_count, embed_fn=embed_fn)

        # --- decay removals -------------------------------------------------
        for m in pipe.pruned_memories[prev_pruned:]:
            rec = ensure(m.get("fact", ""))
            rec["pruned_by_decay"] = True
            rec["prune_reason"] = m.get("prune_reason")
            rec["superseded"] = True if m.get("superseded_prior_fact") else rec["superseded"]

        # --- hard budget evictions -----------------------------------------
        for e in result.get("budget_evictions", []):
            rec = ensure(e.get("fact", ""))
            rec["removed_by_store_budget"] = True
            rec["eviction_reason"] = e.get("eviction_reason")

        # --- live store snapshot -------------------------------------------
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
                rec["max_importance"] = ci if rec["max_importance"] is None else max(rec["max_importance"], ci)
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
            rec["confidence"] = m.get("confidence")
            if rec["write_time_salience"] is None:
                rec["write_time_salience"] = m.get("write_time_salience")

        store_pressure_log.append(dict(result.get("store_pressure", {})))

    # --- final store -------------------------------------------------------
    store = pipe.active_memories
    for m in store:
        if m.get("superseded_prior_fact"):
            superseded_texts.add(m["superseded_prior_fact"])
    store_texts = {m.get("fact", "") for m in store}
    final_store_entries = [dict(m) for m in store]

    # --- query-time retrieval (read-only) ---------------------------------
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

    # --- annotate retrieval candidate/selected -----------------------------
    query_by_source: Dict[int, Dict] = {q["source_turn"]: q for q in queries}
    for q in queries:
        cand_texts = {c.get("fact", "") for c in ranked_by_qid[q["qid"]]}
        sel_texts = {c.get("fact", "") for c in selected_by_qid[q["qid"]]}
        for text, rec in registry.items():
            if text in cand_texts:
                rec["retrieval_candidate"] = True
                for c in ranked_by_qid[q["qid"]]:
                    if c.get("fact") == text:
                        rec["retrieval_sim"] = c.get("retrieval_sim")
                        rec["rank"] = c.get("rank")
                        break
            if text in sel_texts:
                rec["retrieval_selected"] = True

    # --- classify lifecycle ------------------------------------------------
    lifecycle: Dict[str, str] = {}
    for text, rec in registry.items():
        lifecycle[text] = _classify(rec, store_texts, superseded_texts, query_by_source,
                                    selected_by_qid, final_store_entries)

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

    if rec["superseded_original"]:
        if present:
            return "SUPERSEDED_INCORRECTLY"
        if text in superseded_texts:
            return "SUPERSEDED_CORRECTLY"
        return _loss_state(rec)

    if present:
        # Answered by the query-time injected context for this fact's query.
        q = query_by_source.get(rec.get("source_turn"))
        retrieved = False
        if q is not None:
            retrieved = _answered(q, selected_by_qid.get(q["qid"], []), _fact_text)
        elif rec["retrieval_selected"]:
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
# Scoring-component diagnosis
# ---------------------------------------------------------------------------

def _score_components(settings: Dict, registry: Dict[str, Dict]) -> Dict[str, Dict]:
    """Per-fact scorer components for facts present in the final store.

    No weight change (0.4/0.3/0.15/0.15). Reported so the reader can see which
    component separates retrieved from missed facts.
    """
    w = settings["scoring_weights"]
    out: Dict[str, Dict] = {}
    for text, rec in registry.items():
        if rec.get("appearances", 0) == 0:
            continue
        utility = rec.get("confidence")
        final = rec.get("final_importance")
        if final is None:
            continue
        last = rec.get("last_access_turn")
        src = rec.get("source_turn")
        recency = (1.0 / (1.0 + 0.05 * (last - src))) if (last is not None and src is not None) else None
        freq = min(1.0, (rec.get("access_count") or 1) / 10.0)
        relevance = rec.get("write_time_salience")
        out[text] = {
            "relevance": relevance,
            "utility": utility,
            "recency": round(recency, 4) if recency is not None else None,
            "frequency": round(freq, 4),
            "final_importance": round(final, 4),
            "weights": w,
        }
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _lifecycle_counts(lifecycle: Dict[str, str]) -> Dict[str, int]:
    counts = {s: 0 for s in STATES}
    for state in lifecycle.values():
        counts[state] = counts.get(state, 0) + 1
    return counts


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


def _retention_curves(gt: List[Dict], registry: Dict[str, Dict],
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
    for b, v in sorted(buckets.items(), key=lambda kv: AGE_BUCKETS.index(
            next(ab for ab in AGE_BUCKETS if f"{ab[0]}-{ab[1] - 1}" == kv[0]))):
        exp = v["expected"]
        out[b] = {
            "expected": exp,
            "store_present_fraction": round(v["present"] / exp, 3) if exp else None,
            "retrieved_fraction": round(v["retrieved"] / exp, 3) if exp else None,
        }
    return out


def _salience_diagnosis(gt: List[Dict], registry: Dict[str, Dict],
                        lifecycle: Dict[str, str]) -> Dict:
    """Ablation D: write-time salience vs query-time similarity."""
    rows = []
    for g in gt:
        if g.get("superseded_by") is not None and not g.get("is_correction_target"):
            continue
        rec = registry.get(g["fact"], {})
        rows.append({
            "retrieved": lifecycle.get(g["fact"]) == "PRESENT_AND_RETRIEVED",
            "present": lifecycle.get(g["fact"]) in
                       ("PRESENT_AND_RETRIEVED", "PRESENT_BUT_NOT_RETRIEVED"),
            "write_time_salience": rec.get("write_time_salience"),
            "retrieval_sim": rec.get("retrieval_sim"),
            "rank": rec.get("rank"),
        })

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    retr = [r for r in rows if r["retrieved"]]
    miss = [r for r in rows if not r["retrieved"] and r["present"]]
    return {
        "n": len(rows),
        "mean_write_time_salience_retrieved": mean([r["write_time_salience"] for r in retr]),
        "mean_write_time_salience_present_missed": mean([r["write_time_salience"] for r in miss]),
        "mean_retrieval_sim_retrieved": mean([r["retrieval_sim"] for r in retr]),
        "mean_retrieval_sim_present_missed": mean([r["retrieval_sim"] for r in miss]),
        "spearman_salience_vs_sim": _spearman(
            [r["write_time_salience"] for r in rows], [r["retrieval_sim"] for r in rows]),
    }


def _spearman(xs: List[Optional[float]], ys: List[Optional[float]]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xr = _ranks([p[0] for p in pairs])
    yr = _ranks([p[1] for p in pairs])
    n = len(pairs)
    mx, my = sum(xr) / n, sum(yr) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xr, yr))
    den = (sum((a - mx) ** 2 for a in xr) * sum((b - my) ** 2 for b in yr)) ** 0.5
    return round(num / den, 4) if den else None


def _ranks(xs: List[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    for rank, i in enumerate(order, start=1):
        ranks[i] = float(rank)
    return ranks


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

def run_cell(seed: int, budget: int, *, turns: int = TURNS, scale: int = SCALE,
             use_embeddings: bool = True, embed_fn=None,
             decay_lambdas: Optional[Dict[str, float]] = None,
             pruning_threshold: Optional[float] = None,
             ablation: str = "full") -> Dict:
    embed_fn = _resolve_embed(use_embeddings, EMBEDDING_MODEL, embed_fn)
    stream, gt, queries = build_coding_session(seed, turns, include_corrections=True, scale=scale)
    store_budget = max(4096, budget * 4)

    settings = load_settings({
        "max_context_tokens": budget,
        "injection_token_limit": budget,
        "memory_store_token_budget": store_budget,
    })

    lambdas = dict(settings["decay_lambdas"]) if decay_lambdas is None else decay_lambdas
    threshold = (float(settings["pruning"]["threshold"])
                 if pruning_threshold is None else pruning_threshold)

    rep = _instrumented_replay(
        stream, gt, queries, settings,
        embed_fn=embed_fn, embedding_model=EMBEDDING_MODEL,
        decay_lambdas=lambdas, pruning_threshold=threshold,
        budget=budget, store_budget=store_budget,
    )

    lifecycle = rep["lifecycle"]
    counts = _lifecycle_counts(lifecycle)
    metrics = _metrics(queries, rep["selected_by_qid"], rep["store_by_qid"])
    store_tokens = sum(word_count(m.get("fact", "")) for m in rep["memories"])
    ctx_recall = metrics.get("context_recall")
    store_recall = metrics.get("store_recall")
    retrieval_loss = (round(store_recall - ctx_recall, 4)
                      if store_recall is not None and ctx_recall is not None else None)
    metrics["retrieval_loss"] = retrieval_loss
    metrics["store_tokens"] = store_tokens
    metrics["store_count"] = len(rep["memories"])

    evicted_total = sum(p.get("evicted_count", 0) for p in rep["store_pressure_log"])
    over_budget_turns = sum(1 for p in rep["store_pressure_log"]
                            if p.get("store_over_budget_before_eviction", 0) > 0)

    return {
        "seed": seed,
        "budget": budget,
        "ablation": ablation,
        "turns": len(stream),
        "scale": scale,
        "query_turn": (queries[0]["query_turn"] if queries else turns + 1),
        "store_budget": store_budget,
        "gt_count": len(gt),
        "lifecycle_counts": counts,
        "lifecycle": lifecycle,
        "registry": rep["registry"],
        "category_survival": _category_survival(gt, rep["registry"], lifecycle),
        "retention_curves": _retention_curves(
            gt, rep["registry"], lifecycle,
            queries[0]["query_turn"] if queries else turns + 1),
        "score_components": _score_components(settings, rep["registry"]),
        "salience_diagnosis": _salience_diagnosis(gt, rep["registry"], lifecycle),
        "metrics": metrics,
        "family_metrics": _family_metrics(queries, rep["selected_by_qid"], rep["store_by_qid"]),
        "store_pressure": {
            "evicted_total": evicted_total,
            "over_budget_turns": over_budget_turns,
        },
        "correction_gate": {
            "correction_recall": metrics.get("correction_recall"),
            "obsolete_retention": metrics.get("obsolete_retention"),
        },
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _aggregate_cells(cells: List[Dict], experiment: str) -> Dict:
    by_budget = defaultdict(list)
    for c in cells:
        by_budget[c["budget"]].append(c)

    agg = {}
    for budget, cs in by_budget.items():
        state_means = {s: _mean([c["lifecycle_counts"].get(s, 0) for c in cs]) for s in STATES}
        total = _mean([c["gt_count"] for c in cs]) or 1
        cause_fraction = {s: (round(v / total, 4) if v is not None else None)
                          for s, v in state_means.items()}
        agg[str(budget)] = {
            "cells": len(cs),
            "gt_count_mean": total,
            "state_counts_mean": state_means,
            "state_fraction": cause_fraction,
            "context_recall_mean": _mean([c["metrics"].get("context_recall") for c in cs]),
            "store_recall_mean": _mean([c["metrics"].get("store_recall") for c in cs]),
            "retrieval_loss_mean": _mean([c["metrics"].get("retrieval_loss") for c in cs]),
            "correction_recall_mean": _mean([c["metrics"].get("correction_recall") for c in cs]),
            "obsolete_retention_mean": _mean([c["metrics"].get("obsolete_retention") for c in cs]),
            "mean_context_tokens": _mean([c["metrics"].get("mean_context_tokens") for c in cs]),
            "store_tokens_mean": _mean([c["metrics"].get("store_tokens") for c in cs]),
            "store_count_mean": _mean([c["metrics"].get("store_count") for c in cs]),
            "evicted_total_mean": _mean([c["store_pressure"]["evicted_total"] for c in cs]),
            "over_budget_turns_mean": _mean([c["store_pressure"]["over_budget_turns"] for c in cs]),
            "salience_diagnosis": _merge_salience([c["salience_diagnosis"] for c in cs]),
        }
    return {
        "experiment": f"e14_{experiment}",
        "cells": cells,
        "aggregate_by_budget": agg,
    }


def _merge_salience(rows: List[Dict]) -> Dict:
    def m(key):
        return _mean([r.get(key) for r in rows])
    return {
        "mean_write_time_salience_retrieved": m("mean_write_time_salience_retrieved"),
        "mean_write_time_salience_present_missed": m("mean_write_time_salience_present_missed"),
        "mean_retrieval_sim_retrieved": m("mean_retrieval_sim_retrieved"),
        "mean_retrieval_sim_present_missed": m("mean_retrieval_sim_present_missed"),
        "spearman_salience_vs_sim": m("spearman_salience_vs_sim"),
    }


def run_diagnosis(budgets=None, seeds=None, turns=TURNS, scale=SCALE,
                  use_embeddings=True, embed_fn=None) -> Dict:
    budgets = budgets or BUDGETS
    seeds = seeds or SEEDS
    cells = []
    for budget in budgets:
        for seed in seeds:
            print(f"  [diagnosis] budget={budget} seed={seed} ...", flush=True)
            cells.append(run_cell(seed, budget, turns=turns, scale=scale,
                                  use_embeddings=use_embeddings, embed_fn=embed_fn,
                                  ablation="full"))
    return _aggregate_cells(cells, "diagnosis")


def run_ablation(ablation: str, budgets=None, seeds=None, turns=TURNS, scale=SCALE,
                 use_embeddings=True, embed_fn=None) -> Dict:
    budgets = budgets or BUDGETS
    seeds = seeds or ABLATION_SEEDS
    cells = []
    for budget in budgets:
        for seed in seeds:
            print(f"  [{ablation}] budget={budget} seed={seed} ...", flush=True)
            if ablation == "no_decay":
                # Experimental toggle: remove decay entirely (no time decay, no deletion).
                zero = {"transient": 0.0, "personal": 0.0,
                        "technical_preference": 0.0, "project_context": 0.0}
                cells.append(run_cell(seed, budget, turns=turns, scale=scale,
                                      use_embeddings=use_embeddings, embed_fn=embed_fn,
                                      decay_lambdas=zero, pruning_threshold=0.0,
                                      ablation=ablation))
            elif ablation == "no_prune_threshold":
                # Decay score applied, but the threshold never deletes a fact.
                cells.append(run_cell(seed, budget, turns=turns, scale=scale,
                                      use_embeddings=use_embeddings, embed_fn=embed_fn,
                                      pruning_threshold=0.0, ablation=ablation))
            else:
                raise ValueError(f"unknown ablation: {ablation}")
    return _aggregate_cells(cells, ablation)


def dominant_cause_table(result: Dict) -> Dict:
    """Overall and per-budget dominant cause-of-loss (excluding retrieval loss)."""
    loss_states = ["NEVER_STORED", "REMOVED_BY_DECAY", "REMOVED_BY_STORE_BUDGET",
                   "MERGED_BY_DEDUPE", "SUPERSEDED_INCORRECTLY", "OTHER"]
    table = {}
    for budget, agg in result["aggregate_by_budget"].items():
        frac = agg["state_fraction"]
        losses = {s: frac.get(s) or 0.0 for s in loss_states}
        dominant = max(losses.items(), key=lambda kv: kv[1])[0]
        table[budget] = {
            "loss_fractions": losses,
            "retrieval_loss_fraction": agg["state_fraction"].get("PRESENT_BUT_NOT_RETRIEVED"),
            "dominant_store_loss": dominant,
        }
    return table


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(result: Dict) -> str:
    lines = []
    lines.append("# E14 Retention Diagnosis Report")
    lines.append("")
    lines.append("Pure diagnosis of why useful facts leave the long-term store. "
                 "No memory-policy change.")
    lines.append("")
    diag = result["diagnosis"]
    cfg = {"budgets": BUDGETS, "seeds": SEEDS, "turns": TURNS, "scale": SCALE,
           "embedding": EMBEDDING_MODEL}
    lines.append("## Configuration")
    lines.append("")
    for k, v in cfg.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## 1. Lifecycle Cause-of-Loss (per budget)")
    lines.append("")
    headers = ["Budget", "gt", "stored+retr", "stored/not", "decay", "budget", "dedupe",
               "super_ok", "super_bad", "never", "other", "corr_recall", "obs_ret"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for b in sorted(diag["aggregate_by_budget"], key=lambda x: int(x)):
        a = diag["aggregate_by_budget"][b]
        s = a["state_counts_mean"] or {}
        def f(x):
            return f"{x:.2f}" if x is not None else "N/A"
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            b, f(a.get("gt_count_mean")),
            f(s.get("PRESENT_AND_RETRIEVED")), f(s.get("PRESENT_BUT_NOT_RETRIEVED")),
            f(s.get("REMOVED_BY_DECAY")), f(s.get("REMOVED_BY_STORE_BUDGET")),
            f(s.get("MERGED_BY_DEDUPE")), f(s.get("SUPERSEDED_CORRECTLY")),
            f(s.get("SUPERSEDED_INCORRECTLY")), f(s.get("NEVER_STORED")),
            f(s.get("OTHER")), f(a.get("correction_recall_mean")),
            f(a.get("obsolete_retention_mean"))))
    lines.append("")
    lines.append("## 2. Dominant Store Loss")
    lines.append("")
    dt = dominant_cause_table(diag)
    lines.append("| Budget | Dominant store loss | Decay | Store budget | Dedupe | Super bad | Retrieval loss |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for b in sorted(dt, key=lambda x: int(x)):
        row = dt[b]
        lf = row["loss_fractions"]
        lines.append(f"| {b} | {row['dominant_store_loss']} | "
                     f"{lf.get('REMOVED_BY_DECAY', 0):.3f} | "
                     f"{lf.get('REMOVED_BY_STORE_BUDGET', 0):.3f} | "
                     f"{lf.get('MERGED_BY_DEDUPE', 0):.3f} | "
                     f"{lf.get('SUPERSEDED_INCORRECTLY', 0):.3f} | "
                     f"{row['retrieval_loss_fraction']} |")
    lines.append("")
    lines.append("## 3. Category Survival (budget 128 reference)")
    lines.append("")
    lines.append("*See JSON for all budgets; the per-category table below uses the "
                 "first budget bucket.*")
    lines.append("")
    lines.append("## 4. Ablations")
    lines.append("")
    for name in ("no_decay", "no_prune_threshold"):
        if name not in result:
            continue
        lines.append(f"### Ablation: {name}")
        lines.append("")
        lines.append("| Budget | ctx_recall | store_recall | retrieval_loss | corr_recall | obs_ret | decay | budget_evict |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for b in sorted(result[name]["aggregate_by_budget"], key=lambda x: int(x)):
            a = result[name]["aggregate_by_budget"][b]
            s = a["state_counts_mean"] or {}
            lines.append(f"| {b} | {a.get('context_recall_mean')} | {a.get('store_recall_mean')} | "
                         f"{a.get('retrieval_loss_mean')} | {a.get('correction_recall_mean')} | "
                         f"{a.get('obsolete_retention_mean')} | {s.get('REMOVED_BY_DECAY')} | "
                         f"{s.get('REMOVED_BY_STORE_BUDGET')} |")
        lines.append("")
    lines.append("## 5. Write-time Salience vs Query-time Similarity (Ablation D)")
    lines.append("")
    lines.append("| Budget | sal_retrieved | sal_missed | sim_retrieved | sim_missed | spearman |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for b in sorted(diag["aggregate_by_budget"], key=lambda x: int(x)):
        sd = diag["aggregate_by_budget"][b]["salience_diagnosis"]
        lines.append(f"| {b} | {sd.get('mean_write_time_salience_retrieved')} | "
                     f"{sd.get('mean_write_time_salience_present_missed')} | "
                     f"{sd.get('mean_retrieval_sim_retrieved')} | "
                     f"{sd.get('mean_retrieval_sim_present_missed')} | "
                     f"{sd.get('spearman_salience_vs_sim')} |")
    lines.append("")
    lines.append("## 6. Limitations / Unresolved")
    lines.append("")
    lines.append("1. Lifecycle state is derived from exact fact-text presence; a fact "
                 "merged into another is classified MERGED_BY_DEDUPE even though its "
                 "information may still be retrievable in merged form.")
    lines.append("2. Store budget is intentionally 4x the active budget (>=4096); store "
                 "pressure is therefore not expected to bind at this workload scale.")
    lines.append("3. Retrieval loss (PRESENT_BUT_NOT_RETRIEVED) is not a store loss and is "
                 "reported separately.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="E14 retention diagnosis")
    parser.add_argument("--quick", action="store_true",
                        help="small configuration for smoke-testing")
    args = parser.parse_args(argv)

    if args.quick:
        budgets, seeds, turns, scale = [64, 128], [42], 120, 3
        ablations = ("no_decay",)
        use_embeddings = True
    else:
        budgets, seeds, turns, scale = BUDGETS, SEEDS, TURNS, SCALE
        ablations = ("no_decay", "no_prune_threshold")
        use_embeddings = True

    from data.coding_workload import self_test
    self_test()

    print("=" * 60)
    print("E14 Retention Diagnosis")
    print("=" * 60)

    result = {}
    print("\n[1] Running full diagnosis ...")
    result["diagnosis"] = run_diagnosis(budgets=budgets, seeds=seeds,
                                        turns=turns, scale=scale,
                                        use_embeddings=use_embeddings)
    for name in ablations:
        print(f"\n[ablation] {name} ...")
        result[name] = run_ablation(name, budgets=budgets, seeds=seeds,
                                    turns=turns, scale=scale,
                                    use_embeddings=use_embeddings)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "e14_retention_diagnosis.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {path}")

    report_path = out_dir / "e14_retention_diagnosis_report.md"
    report_path.write_text(generate_report(result))
    print(f"Wrote {report_path}")
    return result


if __name__ == "__main__":
    main()
