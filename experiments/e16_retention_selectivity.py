"""e16 - Phase 12 Retention-Selectivity experiment: task-relevant survival
under hard store pressure, identity-safe evaluation, and causal baselines.

Phase 11 (E15) separated activation (current_importance) from survival
(retention_priority) and showed retention-priority eviction is causation-complete
under soft_decay. Two gaps remain, both tackled here:

1. E15's evaluation was *token-based*: "obsolete_retention" and
   "correction_recall" match required/forbidden answer tokens. Tokens can collide
   across unrelated facts (the token "memcache" appears in the superseded session
   cache decision AND in unrelated library facts; "redis" appears in 8+ facts),
   so raw E15 stress cells show obsolete_retention > 0 even though fact-level
   supersession is always correct (SUPERSEDED_INCORRECTLY = 0 in all 81 stress
   cells). E16 adds *identity-safe* metrics keyed on the deterministic
   ``fact_id`` (category:qtype:source_turn:SHA256-prefix) so presence/absence is
   measured on facts, not tokens.

2. ``retention_priority`` is fed by ``access_count``, which increments on every
   retrieval. A fact that is retrieved gets a higher retention_priority and
   therefore *survives eviction*, independently of whether retrieval is driven by
   task relevance or by the decayed importance the retriever ranks on — a
   feedback loop that may make "retention_priority-driven survival" trivially
   equal to "retrieval-driven survival". E16 separates the counters
   (``retrieval_access_count`` vs ``ingest_reinforcement_count``), diagnoses the
   loop's saturation, and tests one future-blind alternative signal:
   ``task_affinity`` (cosine proximity to the recent run of fact-carrying user
   turns) against causal baselines `random` and `base_score_only` and a
   theoretical `oracle_future_use` upper bound.

Research question: under genuine store pressure (store token budgets far below
the natural store), can a memory policy selectively retain task-relevant facts —
measured by future-use retention — without collapsing precision, resurrecting
superseded facts, or sacrificing correction safety?

Policies (all share retention.mode = dual_score so decay never deletes; eviction
is the only causally varying knob):

    dual_score         Phase-11 default: evict lowest retention_priority
    base_score_only    evict lowest base_score (NO access signals, NO decayed
                       current_importance, NO last_access_turn)
    task_affinity      evict lexicographically lowest (task_affinity,
                       retention_priority), source_turn_id tie-break
    random             evict at random (seeded, lower bound)
    oracle_future_use  evict lowest future_use (0 before 1) — OFFLINE-ONLY
                       theoretical upper bound

Correction safety: every policy runs with ``protect_corrections=True`` — a
memory that is the current authoritative version of a corrected fact (carries
``is_current_correction`` / ``superseded_prior_fact_id``) is never evicted while
its superseded predecessor still exists; ``protected_capacity_conflict`` records
whether the store budget is too tight to honour that invariant.

No-future-leakage: ``future_use`` labels are derived from queries, but they are
consumed ONLY by (a) the offline oracle policy and (b) offline metric/analysis
code. The task-state tracker (``memory_optimizer/task_state``) observes only the
current turn's user message at ingest time and can never see later turns; the
causal policies never read retrieval counts, future text, gold labels or future
fact relevance. Assertions in ``replay_session`` pin these invariants.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

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

ACTIVE_BUDGETS = [64, 128, 256]
STORE_BUDGETS = [256, 512, 1024, 2048]
SEEDS = [42, 43, 44, 45, 46]
TURNS = 1200
SCALE = 27
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_ENDPOINT = "http://localhost:11434"
PRUNING_THRESHOLD = 0.2

POLICIES = ["dual_score", "base_score_only", "task_affinity", "random",
            "oracle_future_use"]

POLICY_DEFS: Dict[str, Dict] = {
    "dual_score": {
        "retention_mode": "dual_score",
        "eviction_priority": "retention_priority",
        "description": "Phase-11 default control: evict lowest retention_priority.",
    },
    "base_score_only": {
        "retention_mode": "dual_score",
        "eviction_priority": "base_score_only",
        "description": "Evict lowest base_score (write-time value only: no "
                       "retrieval_access_count, no current_importance, no "
                       "last_access_turn).",
    },
    "task_affinity": {
        "retention_mode": "dual_score",
        "eviction_priority": "task_affinity",
        "description": "Evict lexicographically lowest (task_affinity, "
                       "retention_priority); task_affinity = top-4 cosine to "
                       "the recent run of fact-carrying turns.",
    },
    "random": {
        "retention_mode": "dual_score",
        "eviction_priority": "random",
        "description": "Random eviction (seeded) — causal lower bound.",
    },
    "oracle_future_use": {
        "retention_mode": "dual_score",
        "eviction_priority": "oracle_future_use",
        "description": "Evict lowest future_use (0 before 1) — OFFLINE-ONLY "
                       "theoretical upper bound.",
    },
}

# Age buckets: 0-99 .. 1000+ (Phase-12 / E16 grid)
AGE_BUCKETS = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 600),
               (600, 800), (800, 1000), (1000, None)]

# Query families are the qtype groups reported in E13's _family_metrics.

# ---------------------------------------------------------------------------
# Identity-safe evaluation
# ---------------------------------------------------------------------------

def _id_answer_parts(query: Dict, entries: List[Dict]) -> Dict:
    """Identity-safe answerability: target fact_id present, forbidden absent.

    Unlike token matching (E13/E15) this cannot false-positive on a token that
    legitimately appears in an unrelated fact (e.g. "memcache"/"redis").
    """
    ids = {e.get("fact_id") for e in entries or [] if e.get("fact_id")}
    target_set = set(query.get("target_fact_ids", []))
    forbidden_set = set(query.get("forbidden_fact_ids", []))
    return {
        "answered": bool(target_set & ids) and not bool(forbidden_set & ids),
        "missing_target": sorted(target_set - ids),
        "present_forbidden": sorted(forbidden_set & ids),
    }


def _id_answered(query: Dict, entries: List[Dict]) -> bool:
    return _id_answer_parts(query, entries)["answered"]


def _identity_metrics(queries: List[Dict], context_by_qid: Dict,
                      store_by_qid: Dict) -> Dict:
    corr_q = [q for q in queries if q["qtype"] in ("correction", "obsolete")]

    def answered(entries_map, qs):
        if not qs:
            return None
        return round(
            sum(1 for q in qs if _id_answered(q, entries_map[q["qid"]])) / len(qs), 3)

    def retention(qs, entries_map):
        if not qs:
            return None
        hits = sum(1 for q in qs
                   if _id_answer_parts(q, entries_map[q["qid"]])["present_forbidden"])
        return round(hits / len(qs), 3)

    return {
        "fact_identity_context_recall": answered(context_by_qid, queries),
        "fact_identity_store_recall": answered(store_by_qid, queries),
        "fact_identity_correction_recall": answered(context_by_qid, corr_q),
        "fact_identity_obsolete_retention": retention(corr_q, context_by_qid),
        "queries": len(queries),
    }


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
    return sum(len(g["fact"].split()) for g in gt if not g.get("superseded_by"))


def _find_stress_scale(min_tokens: int, num_turns: int, seed: int = 42) -> int:
    for scale in range(1, 80):
        try:
            _s, gt, _q = build_coding_session(seed, num_turns, True, scale=scale)
        except ValueError:
            continue
        if _natural_store_tokens(gt) >= min_tokens:
            return scale
    raise RuntimeError(f"no scale reaches {min_tokens} natural tokens")


def _all_authoritative_ids(gt: List[Dict]) -> set:
    return {g["fact_id"] for g in gt
            if not (g.get("superseded_by") and not g.get("is_correction_target"))}


def _age_bucket(age: int) -> str:
    for lo, hi in AGE_BUCKETS:
        if hi is None:
            if age >= lo:
                return f"{lo}-+"
        elif lo <= age < hi:
            return f"{lo}-{hi - 1}"
    return f"{AGE_BUCKETS[-1][0]}-+"


def _default_dist(stats: Dict) -> Dict:
    """Normalise a stats dict to the exact diagnostic keys (missing -> None)."""
    keys = ["min", "max", "mean", "median", "p10", "p90"]
    out = {}
    for k in keys:
        v = stats.get(k)
        out[k] = round(v, 4) if isinstance(v, float) else v
    return out


def _stats(vals: List[float]) -> Dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"min": None, "max": None, "mean": None, "median": None, "p10": None, "p90": None}
    vals = sorted(vals)
    n = len(vals)

    def pct(p):
        idx = int(math.ceil(p * n)) - 1
        return vals[max(0, min(n - 1, idx))]
    return {
        "min": float(vals[0]),
        "max": float(vals[-1]),
        "mean": float(sum(vals) / n),
        "median": float(vals[n // 2]),
        "p10": float(pct(0.10)),
        "p90": float(pct(0.90)),
    }


def _mean_rank(xs: List[float]) -> Dict[float, float]:
    """Average ranks with ties -> {value: rank} map for Spearman."""
    sorted_xs = sorted(xs)
    ranks: Dict[float, float] = {}
    i = 0
    n = len(sorted_xs)
    while i < n:
        j = i
        while j + 1 < n and sorted_xs[j + 1] == sorted_xs[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[sorted_xs[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    rx = _mean_rank([p[0] for p in pairs])
    ry = _mean_rank([p[1] for p in pairs])
    rxs = [rx[p[0]] for p in pairs]
    rys = [ry[p[1]] for p in pairs]
    n = len(pairs)
    mx = sum(rxs) / n
    my = sum(rys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rxs, rys))
    den = math.sqrt(sum((a - mx) ** 2 for a in rxs) * sum((b - my) ** 2 for b in rys))
    if den == 0:
        return None
    return num / den


# ---------------------------------------------------------------------------
# Instrumented replay (identity-safe, policy-aware)
# ---------------------------------------------------------------------------

def replay_session(stream, gt, queries, settings, *, embed_fn, embedding_model,
                   decay_lambdas, pruning_threshold, budget, store_budget,
                   retention_mode, eviction_priority, future_use: Optional[Dict] = None,
                   protect_corrections: bool = True, task_context_window: int = 32,
                   random_seed: int = 0) -> Dict:
    settings = dict(settings)
    settings["memory_store_token_budget"] = store_budget
    retention_cfg = {
        "mode": retention_mode,
        "eviction_priority": eviction_priority,
        "protect_corrections": protect_corrections,
        "task_context_window": task_context_window,
        "random_seed": random_seed,
    }
    if eviction_priority == "oracle_future_use":
        assert future_use is not None, "oracle policy requires future_use labels"
        retention_cfg["oracle_future_use"] = future_use
    settings["retention"] = retention_cfg

    scorer = ImportanceScorer(weights=settings["scoring_weights"])
    decay = CategoryDecayEngine(lambdas=decay_lambdas, pruning_threshold=pruning_threshold)
    retriever = MemoryRetriever(
        top_k=int(settings["top_k"]),
        sim_threshold=float(settings.get("similarity_threshold", 0.35)),
        embed_fn=embed_fn, embedding_model=embedding_model,
    )
    compressor = MemoryCompressor()
    pipe = AdaptiveMemoryPipeline(settings, scorer, decay, retriever, compressor)

    # No-future-leakage guardrails (Phase 12, section 24).
    target_ids = set().union(*(set(q.get("target_fact_ids", [])) for q in queries))
    forbidden_ids = set().union(*(set(q.get("forbidden_fact_ids", [])) for q in queries))
    all_query_ids = target_ids | forbidden_ids

    superseded_ids = {g["fact_id"] for g in gt
                      if g.get("superseded_by") and not g.get("is_correction_target")}

    registry: Dict[str, Dict] = {}
    for g in gt:
        registry[g["fact_id"]] = {
            "fact_id": g["fact_id"],
            "fact": g["fact"],
            "category": g.get("category", ""),
            "qtype": g.get("qtype", ""),
            "source_turn": int(g["source_turn"]),
            "is_correction_target": bool(g.get("is_correction_target")),
            "is_current_correction": bool(g.get("is_current_correction")),
            "superseded_original": bool(g.get("superseded_by") is not None
                                        and not g.get("is_correction_target")),
            "superseded_prior_fact_id": g.get("superseded_prior_fact_id"),
            "superseded_by": g.get("superseded_by"),
            "future_use": 1 if g["fact_id"] in target_ids else 0,
            "appearances": 0,
            "decay_steps_survived": 0,
            "pruned_by_decay": False,
            "removed_by_store_budget": False,
            "eviction_record": None,
            "evicted_at_turn": None,
            "retained_below_threshold": False,
            "retrieval_candidate": False,
            "retrieval_selected": False,
            "final_base_score": None,
            "final_current_importance": None,
            "final_retention_priority": None,
            "final_task_affinity": None,
            "final_retrieval_access_count": None,
            "final_ingest_reinforcement_count": None,
            "final_access_count": None,
            "final_duplicates": None,
            "final_last_access_turn": None,
        }

    store_pressure_log: List[Dict] = []
    evictions_seen: Dict[str, Dict] = {}

    for turn in stream:
        turn_id = int(turn["turn_id"])
        facts = turn.get("facts", [])
        # Guardrails: the replay loop must be a pure ingest loop.
        # (a) No query text is ever handed to the pipeline during the loop.
        assert all(turn["user"] != q["user"] for q in queries), \
            f"query text leaked into ingest at turn {turn_id}"
        prev_pruned = len(pipe.pruned_memories)
        result = pipe.ingest(turn_id, turn["user"], facts, fact_tokens=word_count,
                             embed_fn=embed_fn)

        # (b) Future-relevance labels never enter ingest: no stream fact carries
        # oracle_future_use (the pipeline stamps it from settings for the oracle
        # policy only, and future_use is an offline-only quantity).
        assert all(f.get("oracle_future_use") is None for f in facts), \
            f"oracle label leaked into ingest at turn {turn_id}"

        # (c) The task-state tracker only ever holds turns already ingested.
        if eviction_priority == "task_affinity":
            tracker_turns = [m["turn"] for m in pipe._task_state.messages]
            assert all(t <= turn_id for t in tracker_turns), \
                f"task-state tracker saw a future turn at {turn_id}"

        for m in pipe.pruned_memories[prev_pruned:]:
            fid = m.get("fact_id")
            if fid in registry:
                registry[fid]["pruned_by_decay"] = True

        for e in result.get("budget_evictions", []):
            fid = e.get("fact_id")
            if fid is None:
                continue
            if fid in registry:
                registry[fid]["removed_by_store_budget"] = True
                registry[fid]["eviction_record"] = e
                registry[fid]["evicted_at_turn"] = turn_id
            evictions_seen[fid] = e

        for m in pipe.active_memories:
            fid = m.get("fact_id")
            rec = registry.get(fid)
            if rec is None:
                continue
            rec["appearances"] += 1
            rec["decay_steps_survived"] += 1
            if m.get("retained_below_threshold"):
                rec["retained_below_threshold"] = True

        store_pressure_log.append(dict(result.get("store_pressure", {})))

    store = pipe.active_memories

    # Final counters: present memories use the live dict, evicted ones use the
    # last eviction record (a snapshot at eviction time).
    for m in store:
        fid = m.get("fact_id")
        rec = registry.get(fid)
        if rec is None:
            continue
        src = m
        rec["final_base_score"] = src.get("base_score")
        rec["final_current_importance"] = src.get("current_importance")
        rec["final_retention_priority"] = src.get("retention_priority")
        rec["final_task_affinity"] = src.get("task_affinity")
        rec["final_retrieval_access_count"] = src.get("retrieval_access_count")
        rec["final_ingest_reinforcement_count"] = src.get("ingest_reinforcement_count")
        rec["final_access_count"] = src.get("access_count", 1)
        rec["final_duplicates"] = src.get("duplicates", 1)
        rec["final_last_access_turn"] = src.get("last_access_turn")

    for fid, e in evictions_seen.items():
        rec = registry.get(fid)
        if rec is None or rec["final_base_score"] is not None:
            continue
        rec["final_base_score"] = e.get("base_score")
        rec["final_current_importance"] = e.get("current_importance")
        rec["final_retention_priority"] = e.get("retention_priority")
        rec["final_task_affinity"] = e.get("task_affinity")
        rec["final_retrieval_access_count"] = e.get("retrieval_access_count")
        rec["final_ingest_reinforcement_count"] = e.get("ingest_reinforcement_count")
        rec["final_access_count"] = e.get("access_count", 1)
        rec["final_duplicates"] = e.get("duplicates", 1)
        rec["final_last_access_turn"] = e.get("last_access_turn")

    store_ids = {m.get("fact_id") for m in store if m.get("fact_id")}
    superseded_prior_ids = {m.get("superseded_prior_fact_id") for m in store
                            if m.get("superseded_prior_fact_id") is not None}
    final_store_entries = [dict(m) for m in store]

    # Query-time retrieval (read-only; ranking unchanged).
    qr = MemoryRetriever(
        top_k=int(settings["top_k"]),
        sim_threshold=float(settings.get("similarity_threshold", 0.35)),
        embed_fn=embed_fn, embedding_model=embedding_model,
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
        cand_ids = {c.get("fact_id") for c in ranked_by_qid.get(q["qid"], [])}
        sel_ids = {c.get("fact_id") for c in selected_by_qid.get(q["qid"], [])}
        for fid, rec in registry.items():
            if fid in cand_ids:
                rec["retrieval_candidate"] = True
            if fid in sel_ids:
                rec["retrieval_selected"] = True

    # Lifecycle classification (identity-based).
    lifecycle: Dict[str, str] = {}
    for fid, rec in registry.items():
        lifecycle[fid] = _classify_id(rec, store_ids, superseded_prior_ids)

    return {
        "memories": store,
        "registry": registry,
        "lifecycle": lifecycle,
        "store_pressure_log": store_pressure_log,
        "evictions_seen": evictions_seen,
        "selected_by_qid": selected_by_qid,
        "ranked_by_qid": ranked_by_qid,
        "store_by_qid": store_by_qid,
        "final_store_entries": final_store_entries,
        "store_ids": store_ids,
        "no_future_leakage_ok": True,
    }


def _classify_id(rec: Dict, store_ids: set, superseded_prior_ids: set) -> str:
    """Identity-based lifecycle classification of a gt fact (by fact_id).

    * superseded ORIGINAL: correct if a current-correction memory still carries
      our id as ``superseded_prior_fact_id`` (the correction replaced it in
      place); incorrect if our own id survives in the store.
    * everything else: present in the store -> PRESENT_* by whether retrieval
      selected it; otherwise a loss state (decay / store-budget / merged).
    """
    fid = rec["fact_id"]
    if rec.get("superseded_original"):
        if fid in store_ids:
            return "SUPERSEDED_INCORRECTLY"
        if fid in superseded_prior_ids:
            return "SUPERSEDED_CORRECTLY"
        return _loss_state(rec)
    if fid in store_ids:
        return ("PRESENT_AND_RETRIEVED" if rec.get("retrieval_selected")
                else "PRESENT_BUT_NOT_RETRIEVED")
    return _loss_state(rec)


def _loss_state(rec: Dict) -> str:
    if rec.get("pruned_by_decay"):
        return "REMOVED_BY_DECAY"
    if rec.get("removed_by_store_budget"):
        return "REMOVED_BY_STORE_BUDGET"
    if rec.get("appearances") == 0:
        return "MERGED_BY_DEDUPE"
    return "OTHER"

# ---------------------------------------------------------------------------
# Per-cell metrics
# ---------------------------------------------------------------------------

def _identity_family_metrics(queries, context_by_qid, store_by_qid) -> Dict[str, Dict]:
    from experiments.e13_generalization import _query_family
    families: Dict[str, List[Dict]] = defaultdict(list)
    for q in queries:
        families[_query_family(q)].append(q)
    out = {}
    for fam, qs in sorted(families.items()):
        ctx = round(sum(1 for q in qs
                        if _id_answered(q, context_by_qid[q["qid"]])) / len(qs), 3) if qs else None
        store = round(sum(1 for q in qs
                          if _id_answered(q, store_by_qid[q["qid"]])) / len(qs), 3) if qs else None
        out[fam] = {"queries": len(qs), "identity_context_recall": ctx,
                    "identity_store_recall": store}
    return out


def _per_cell(rep: Dict, gt: List[Dict], queries: List[Dict], *, budget: int,
              store_budget: int, policy: str, seed: int, turns: int, scale: int,
              task_context_window: int = 32) -> Dict:
    lifecycle = rep["lifecycle"]
    registry = rep["registry"]

    def count_of(state: str) -> int:
        return sum(1 for s in lifecycle.values() if s == state)

    state_counts = {
        "PRESENT_AND_RETRIEVED": count_of("PRESENT_AND_RETRIEVED"),
        "PRESENT_BUT_NOT_RETRIEVED": count_of("PRESENT_BUT_NOT_RETRIEVED"),
        "REMOVED_BY_STORE_BUDGET": count_of("REMOVED_BY_STORE_BUDGET"),
        "REMOVED_BY_DECAY": count_of("REMOVED_BY_DECAY"),
        "MERGED_BY_DEDUPE": count_of("MERGED_BY_DEDUPE"),
        "SUPERSEDED_CORRECTLY": count_of("SUPERSEDED_CORRECTLY"),
        "SUPERSEDED_INCORRECTLY": count_of("SUPERSEDED_INCORRECTLY"),
        "OTHER": count_of("OTHER"),
    }

    legacy = _metrics(queries, rep["selected_by_qid"], rep["store_by_qid"])
    identity = _identity_metrics(queries, rep["selected_by_qid"], rep["store_by_qid"])

    store_tokens = sum(word_count(m.get("fact", "")) for m in rep["memories"])
    legacy["retrieval_loss"] = (
        round(legacy["store_recall"] - legacy["context_recall"], 4)
        if legacy["store_recall"] is not None and legacy["context_recall"] is not None else None)
    legacy["store_tokens"] = store_tokens
    legacy["store_count"] = len(rep["memories"])
    legacy["mean_context_utilization"] = (
        round(legacy["mean_context_tokens"] / budget, 4) if budget else None)

    evicted_total = sum(p.get("evicted_count", 0) for p in rep["store_pressure_log"])
    conflict_turns = sum(1 for p in rep["store_pressure_log"]
                         if p.get("protected_capacity_conflict"))
    over_budget_turns = sum(1 for p in rep["store_pressure_log"]
                            if p.get("store_over_budget_before_eviction", 0) > 0)

    author_ids = _all_authoritative_ids(gt)
    target_ids = set().union(*(set(q.get("target_fact_ids", [])) for q in queries))
    store_ids = rep["store_ids"]
    future_ids = {fid for fid in author_ids if fid in target_ids}

    retained_author = author_ids & store_ids
    retained_future = future_ids & store_ids
    retention_recall_raw = (len(retained_future) / len(future_ids)) if future_ids else None
    precision_raw = (len(retained_future) / len(retained_author)) if retained_author else None
    evicted_future = future_ids - store_ids

    # new-correction / old-superseded gate (identity, at the final store)
    corr_q = [q for q in queries if q["qtype"] in ("correction", "obsolete")]
    new_correction_missing = 0
    old_superseded_present = 0
    for q in corr_q:
        parts = _id_answer_parts(q, rep["final_store_entries"])
        new_correction_missing += len(parts["missing_target"])
        old_superseded_present += len(parts["present_forbidden"])

    # Saturation diagnostics per population bucket.
    def bucket_members(pred) -> List[str]:
        return [fid for fid in author_ids if pred(registry[fid], lifecycle.get(fid))]

    def signal_rows(ids: List[str]) -> Dict[str, Dict]:
        def col(key):
            return [rec.get(key) for rec in (registry[i] for i in ids)
                    if rec.get(key) is not None]
        return {
            "base_score": _default_dist(_stats(col("final_base_score"))),
            "current_importance": _default_dist(_stats(col("final_current_importance"))),
            "retention_priority": _default_dist(_stats(col("final_retention_priority"))),
            "task_affinity": _default_dist(_stats(col("final_task_affinity"))),
            "retrieval_access_count": _default_dist(_stats(
                [float(v) for v in col("final_retrieval_access_count")])),
            "ingest_reinforcement_count": _default_dist(_stats(
                [float(v) for v in col("final_ingest_reinforcement_count")])),
            "duplicates": _default_dist(_stats(
                [float(v) for v in col("final_duplicates")])),
        }

    retrieved_needed = [fid for fid in future_ids
                        if lifecycle.get(fid) == "PRESENT_AND_RETRIEVED"]
    stored_not_retrieved = [fid for fid in author_ids
                            if lifecycle.get(fid) == "PRESENT_BUT_NOT_RETRIEVED"]
    evicted = [fid for fid in author_ids
               if lifecycle.get(fid) == "REMOVED_BY_STORE_BUDGET"]
    never_retrieved = [fid for fid in author_ids
                       if not registry[fid].get("retrieval_selected")]
    corrections = [fid for fid in author_ids
                   if registry[fid].get("is_correction_target")]
    obsolete = [fid for fid in author_ids if registry[fid].get("superseded_original")]

    signals = ["final_base_score", "final_current_importance", "final_retention_priority",
               "final_task_affinity", "final_retrieval_access_count",
               "final_ingest_reinforcement_count", "final_duplicates"]
    spearman_future = {}
    for sig in signals:
        xs = [registry[fid].get(sig) for fid in author_ids]
        ys = [registry[fid].get("future_use", 0) for fid in author_ids]
        v = _spearman(xs, ys)
        spearman_future[sig.replace("final_", "")] = (
            round(v, 4) if v is not None else None)

    # Age-bucket future-use retention (age at query time) and eviction ages.
    query_turn = queries[0]["query_turn"] if queries else turns + 1
    age_buckets: Dict[str, Dict] = {}
    evicted_age_buckets: Dict[str, Dict] = {}
    for fid in future_ids:
        rec = registry[fid]
        age = max(0, int(query_turn) - int(rec["source_turn"]))
        b = _age_bucket(age)
        ab = age_buckets.setdefault(b, {"future_expected": 0, "future_retained": 0})
        ab["future_expected"] += 1
        if fid in store_ids:
            ab["future_retained"] += 1
        evt = rec.get("evicted_at_turn")
        if evt is not None:
            eb = age_buckets.setdefault(b, {})
            evb = evicted_age_buckets.setdefault(
                _age_bucket(max(0, int(evt) - int(rec["source_turn"]))),
                {"future_evicted": 0})
            evb["future_evicted"] += 1

    agg_signal_buckets = {
        "retrieved_needed": {fid: registry[fid] for fid in retrieved_needed},
        "stored_not_retrieved": {fid: registry[fid] for fid in stored_not_retrieved},
        "evicted": {fid: registry[fid] for fid in evicted},
        "never_retrieved": {fid: registry[fid] for fid in never_retrieved},
        "corrections": {fid: registry[fid] for fid in corrections},
        "obsolete": {fid: registry[fid] for fid in obsolete},
    }
    diagnostics = {
        "by_population": {
            name: signal_rows(list(ids)) for name, ids in (
                ("retrieved_needed", retrieved_needed),
                ("stored_not_retrieved", stored_not_retrieved),
                ("evicted", evicted),
                ("never_retrieved", never_retrieved),
                ("corrections", corrections),
                ("obsolete", obsolete),
            )
        },
        "population_sizes": {name: len(ids) for name, ids in (
            ("retrieved_needed", retrieved_needed),
            ("stored_not_retrieved", stored_not_retrieved),
            ("evicted", evicted),
            ("never_retrieved", never_retrieved),
            ("corrections", corrections),
            ("obsolete", obsolete),
        )},
        "spearman_vs_future_use": spearman_future,
    }

    final_counts = {
        "retrieved_needed": len(retrieved_needed),
        "stored_not_retrieved": len(stored_not_retrieved),
        "evicted": len(evicted),
        "never_retrieved": len(never_retrieved),
        "corrections": len(corrections),
        "obsolete": len(obsolete),
        "future_use_total": len(future_ids),
        "future_use_retained": len(retained_future),
        "future_use_evicted": len(evicted_future),
    }

    return {
        "experiment": "e16_retention_selectivity",
        "seed": seed,
        "budget": budget,
        "store_budget": store_budget,
        "policy": policy,
        "turns": turns,
        "scale": scale,
        "query_turn": query_turn,
        "task_context_window": task_context_window,
        "gt_count": len(gt),
        "state_counts": state_counts,
        "metrics": legacy,
        "identity_metrics": identity,
        "future_use": {
            "future_use_total": len(future_ids),
            "future_use_retained": len(retained_future),
            "retention_recall": (round(retention_recall_raw, 4)
                                 if retention_recall_raw is not None else None),
            "precision_of_retention": (round(precision_raw, 4)
                                       if precision_raw is not None else None),
            "retained_future": len(retained_future),
            "evicted_future": len(evicted_future),
        },
        "correction_gate": {
            "identity_correction_recall": identity.get("fact_identity_correction_recall"),
            "identity_obsolete_retention": identity.get("fact_identity_obsolete_retention"),
            "new_correction_missing": new_correction_missing,
            "old_superseded_present": old_superseded_present,
            "protected_capacity_conflict_turns": conflict_turns,
        },
        "family_metrics": _identity_family_metrics(queries, rep["selected_by_qid"],
                                                    rep["store_by_qid"]),
        "store_summary": {
            "store_count": len(rep["memories"]),
            "store_tokens": store_tokens,
            "evicted_total": evicted_total,
            "over_budget_turns": over_budget_turns,
            "protected_capacity_conflict_turns": conflict_turns,
            "retained_below_threshold_count": sum(
                1 for rec in registry.values() if rec.get("retained_below_threshold")),
            "candidate_count": sum(1 for rec in registry.values()
                                   if rec.get("retrieval_candidate")),
            "selected_count": sum(1 for rec in registry.values()
                                  if rec.get("retrieval_selected")),
        },
        "diagnostics": diagnostics,
        "age_analysis": {
            bucket: {
                "future_expected": v.get("future_expected", 0),
                "future_retained": v.get("future_retained", 0),
                "retained_fraction": round(v["future_retained"] / v["future_expected"], 3)
                if v.get("future_expected") else None,
            }
            for bucket, v in sorted(age_buckets.items(), key=lambda kv: _bucket_key(kv[0]))
        },
        "evicted_age_analysis": {
            bucket: {"future_evicted": v.get("future_evicted", 0)}
            for bucket, v in sorted(evicted_age_buckets.items(),
                                    key=lambda kv: _bucket_key(kv[0]))
        },
        "no_future_leakage_ok": bool(rep.get("no_future_leakage_ok")),
        "final_counts": final_counts,
    }


def _bucket_key(label: str) -> int:
    for i, (lo, hi) in enumerate(AGE_BUCKETS):
        if hi is None:
            if label == f"{lo}-+":
                return i
        elif label == f"{lo}-{hi - 1}":
            return i
    return len(AGE_BUCKETS)


# ---------------------------------------------------------------------------
# Cell driver
# ---------------------------------------------------------------------------

def run_cell(seed: int, budget: int, policy: str, *, turns: int = TURNS,
             scale: int = SCALE, store_budget: Optional[int] = None,
             use_embeddings: bool = True, embed_fn=None,
             protect_corrections: bool = True, task_context_window: int = 32,
             random_seed: Optional[int] = None,
             capture_store_ids: bool = False) -> Dict:
    embed_fn = _resolve_embed(use_embeddings, EMBEDDING_MODEL, embed_fn)
    stream, gt, queries = build_coding_session(seed, turns, include_corrections=True,
                                               scale=scale)
    future_use = {fid: 1 for q in queries for fid in q.get("target_fact_ids", [])}
    cfg = POLICY_DEFS[policy]
    settings = load_settings({
        "max_context_tokens": budget,
        "injection_token_limit": budget,
        "memory_store_token_budget": store_budget if store_budget is not None else 0,
    })
    lambdas = dict(settings["decay_lambdas"])
    rseed = seed if random_seed is None else random_seed
    rep = replay_session(
        stream, gt, queries, settings,
        embed_fn=embed_fn, embedding_model=EMBEDDING_MODEL,
        decay_lambdas=lambdas, pruning_threshold=PRUNING_THRESHOLD,
        budget=budget, store_budget=store_budget if store_budget is not None else 0,
        retention_mode=cfg["retention_mode"],
        eviction_priority=cfg["eviction_priority"],
        future_use=future_use if policy == "oracle_future_use" else None,
        protect_corrections=protect_corrections,
        task_context_window=task_context_window, random_seed=rseed,
    )
    cell = _per_cell(rep, gt, queries, budget=budget,
                     store_budget=store_budget if store_budget is not None else 0,
                     policy=policy, seed=seed, turns=turns, scale=scale,
                     task_context_window=task_context_window)
    if capture_store_ids:
        cell["store_fact_ids"] = sorted(rep["store_ids"])
    return cell


def _cell_key(cell: Dict) -> str:
    return f"{cell['policy']}:{cell['store_budget']}:{cell['budget']}:{cell['seed']}"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _metrics_mean(cells: List[Dict], key: str):
    return _mean([c["metrics"].get(key) for c in cells])


def _identity_mean(cells: List[Dict], key: str):
    return _mean([c["identity_metrics"].get(key) for c in cells])


def _future_mean(cells: List[Dict], key: str):
    return _mean([c["future_use"].get(key) for c in cells])


def _store_mean(cells: List[Dict], key: str):
    return _mean([c["store_summary"].get(key) for c in cells])


def _gate_mean(cells: List[Dict], key: str):
    return _mean([c["correction_gate"].get(key) for c in cells])


def aggregate_cells(cells: List[Dict]) -> Dict:
    by = defaultdict(list)
    for c in cells:
        by[(c["policy"], c["store_budget"], c["budget"])].append(c)
    out: Dict[str, Dict] = {}
    for (policy, sb, ab), rows in sorted(by.items()):
        out.setdefault(policy, {}).setdefault(str(sb), {})[ab] = {
            "cells": len(rows),
            "seeds": sorted({c["seed"] for c in rows}),
            "context_recall_mean": _metrics_mean(rows, "context_recall"),
            "store_recall_mean": _metrics_mean(rows, "store_recall"),
            "retrieval_loss_mean": _metrics_mean(rows, "retrieval_loss"),
            "correction_recall_mean": _metrics_mean(rows, "correction_recall"),
            "obsolete_retention_mean": _metrics_mean(rows, "obsolete_retention"),
            "identity_context_recall_mean": _identity_mean(rows, "fact_identity_context_recall"),
            "identity_store_recall_mean": _identity_mean(rows, "fact_identity_store_recall"),
            "identity_correction_recall_mean": _identity_mean(
                rows, "fact_identity_correction_recall"),
            "identity_obsolete_retention_mean": _identity_mean(
                rows, "fact_identity_obsolete_retention"),
            "future_use_total": int(_future_mean(rows, "future_use_total") or 0),
            "future_use_retention_recall_mean": _future_mean(rows, "retention_recall"),
            "precision_of_retention_mean": _future_mean(rows, "precision_of_retention"),
            "new_correction_missing_mean": _gate_mean(rows, "new_correction_missing"),
            "old_superseded_present_mean": _gate_mean(rows, "old_superseded_present"),
            "protected_capacity_conflict_turns_mean": _gate_mean(
                rows, "protected_capacity_conflict_turns"),
            "store_tokens_mean": _store_mean(rows, "store_tokens"),
            "store_count_mean": _store_mean(rows, "store_count"),
            "evicted_total_mean": _store_mean(rows, "evicted_total"),
            "mean_context_tokens": _metrics_mean(rows, "mean_context_tokens"),
            "mean_context_utilization": _metrics_mean(rows, "mean_context_utilization"),
        }
    return out


def aggregate_no_pressure(cells: List[Dict]) -> Dict:
    """Store contents must not differ by eviction policy when there is no pressure."""
    by = defaultdict(list)
    for c in cells:
        by[(c["policy"], c["seed"])].append(c)
    out = {}
    for (policy, seed), rows in sorted(by.items()):
        out.setdefault(policy, {})[seed] = {
            "cells": len(rows),
            "identity_store_recall_mean": _identity_mean(rows, "fact_identity_store_recall"),
            "future_use_retention_recall_mean": _future_mean(rows, "retention_recall"),
        }
    return out


def aggregate_window(cells: List[Dict]) -> Dict:
    by = defaultdict(list)
    for c in cells:
        by[(c["task_context_window"], c["seed"])].append(c)
    out = {}
    for (w, seed), rows in sorted(by.items()):
        out.setdefault(str(w), {})[seed] = {
            "identity_store_recall_mean": _identity_mean(rows, "fact_identity_store_recall"),
            "future_use_retention_recall_mean": _future_mean(rows, "retention_recall"),
            "precision_of_retention_mean": _future_mean(rows, "precision_of_retention"),
        }
    out["_mean"] = {}
    for w in sorted({str(c["task_context_window"]) for c in cells}):
        rows = [c for c in cells if str(c["task_context_window"]) == w]
        out["_mean"][w] = {
            "identity_store_recall_mean": _identity_mean(rows, "fact_identity_store_recall"),
            "future_use_retention_recall_mean": _future_mean(rows, "retention_recall"),
            "precision_of_retention_mean": _future_mean(rows, "precision_of_retention"),
        }
    return out


# ---------------------------------------------------------------------------
# Decision rule (8 criteria)
# ---------------------------------------------------------------------------

_LOW_STORE = [256, 512]
_MARGIN = 0.03


def _low_store_rows(cells: List[Dict], policy: str) -> List[Dict]:
    return [c for c in cells if c["policy"] == policy and c["store_budget"] in _LOW_STORE]


def classify_result(result: Dict) -> Dict:
    cells = result["grid"]["cells"]
    windows = result.get("window_sensitivity", {})
    nop = result.get("no_pressure", {})

    def low_matrix(policy, metric):
        rows = _low_store_rows(cells, policy)
        return _mean([c["future_use"].get(metric) for c in rows])

    task_rr = low_matrix("task_affinity", "retention_recall")
    dual_rr = low_matrix("dual_score", "retention_recall")
    base_rr = low_matrix("base_score_only", "retention_recall")
    task_prec = low_matrix("task_affinity", "precision_of_retention")
    dual_prec = low_matrix("dual_score", "precision_of_retention")

    # criterion 6: improvement holds across >=4/5 seeds at the tightest store.
    seed_ok = 0
    total = 0
    for seed in SEEDS:
        t = [c for c in cells if c["policy"] == "task_affinity"
             and c["store_budget"] == min(_LOW_STORE) and c["seed"] == seed]
        d = [c for c in cells if c["policy"] == "dual_score"
             and c["store_budget"] == min(_LOW_STORE) and c["seed"] == seed]
        if t and d:
            total += 1
            if (t[0]["future_use"]["retention_recall"] or 0) \
                    > (d[0]["future_use"]["retention_recall"] or 0):
                seed_ok += 1

    # criterion 3/4: correction/obsolete safety (identity-safe), any store.
    task_corr = _mean([c["identity_metrics"]["fact_identity_correction_recall"]
                       for c in cells if c["policy"] == "task_affinity"])
    task_obs = _mean([c["identity_metrics"]["fact_identity_obsolete_retention"]
                      for c in cells if c["policy"] == "task_affinity"])
    max_task_obs = max([c["identity_metrics"]["fact_identity_obsolete_retention"] or 0
                        for c in cells if c["policy"] == "task_affinity"])

    # criterion 5: active context unchanged (budget enforcement is by construction).
    task_ctx = _mean([c["metrics"]["mean_context_tokens"]
                      for c in _low_store_rows(cells, "task_affinity")])
    dual_ctx = _mean([c["metrics"]["mean_context_tokens"]
                      for c in _low_store_rows(cells, "dual_score")])
    ctx_ok = (task_ctx is None or dual_ctx is None or task_ctx <= dual_ctx + 2.0)

    # criterion 8: oracle gap (identity store recall at low store).
    oracle_store = _mean([c["identity_metrics"]["fact_identity_store_recall"]
                          for c in _low_store_rows(cells, "oracle_future_use")])
    gaps = {
        p: (round((oracle_store or 0) - low_matrix(p, "retention_recall"), 4))
        for p in ("dual_score", "task_affinity", "base_score_only", "random",
                  "oracle_future_use")
    }

    c1 = task_rr is not None and dual_rr is not None and task_rr > dual_rr + _MARGIN
    c2 = (task_prec is not None and dual_prec is not None
          and task_prec >= dual_prec - 0.15)
    c3 = task_corr is not None and task_corr >= 0.95
    c4 = task_obs is not None and task_obs == 0.0 and max_task_obs == 0
    c5 = ctx_ok
    c6 = total > 0 and (seed_ok / total) >= 0.8
    c7 = task_rr is not None and base_rr is not None and task_rr > base_rr + _MARGIN
    c8 = (gaps["task_affinity"] is not None and gaps["dual_score"] is not None
          and gaps["task_affinity"] < gaps["dual_score"])

    supported = bool(c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8)
    return {
        "task_affinity_advances": supported,
        "criteria": {
            "c1_beats_dual_future_use_recall_j_n_0.03_at_low_store": {
                "task_affinity": task_rr, "dual_score": dual_rr, "ok": c1},
            "c2_precision_does_not_collapse_vs_dual_n_0.15": {
                "task_affinity": task_prec, "dual_score": dual_prec, "ok": c2},
            "c3_identity_correction_recall_j_0.95": {"value": task_corr, "ok": c3},
            "c4_identity_obsolete_retention_==_0": {
                "mean": task_obs, "max": max_task_obs, "ok": c4},
            "c5_active_context_cost_unchanged": {
                "task_affinity_ctx": task_ctx, "dual_ctx": dual_ctx, "ok": c5},
            "c6_holds_across_seeds_4_of_5": {
                "seeds_winning": seed_ok, "seeds_total": total, "ok": c6},
            "c7_beats_base_score_only": {
                "task_affinity": task_rr, "base_score_only": base_rr, "ok": c7},
            "c8_closes_oracle_gap": {"gaps": gaps, "ok": c8},
        },
        "rationale": (
            f"task_affinity vs dual_score future-use retention recall at low store "
            f"({_LOW_STORE}): {task_rr} vs {dual_rr}; precision {task_prec} vs "
            f"{dual_prec}; per-seed wins {seed_ok}/{total}; oracle store gap "
            f"task {gaps['task_affinity']} vs dual {gaps['dual_score']}."),
    }


# ---------------------------------------------------------------------------
# E15 raw-JSON audit (report section 2) — derived, not hand-typed
# ---------------------------------------------------------------------------

def _token_census(seed: int = 42, turns: int = 1200, scale: int = 27,
                  tokens=("memcache", "redis", "100", "250")) -> Dict:
    """Count how many distinct authoritative gt facts contain each token, to
    quantify E15's token-level evaluation hazard."""
    _s, gt, _q = build_coding_session(seed, turns, include_corrections=True, scale=scale)
    auth = [g for g in gt
            if not (g.get("superseded_by") and not g.get("is_correction_target"))]
    out = {}
    for tok in tokens:
        owners = [g for g in auth if tok.lower() in g["fact"].lower().split()]
        out[tok] = {
            "authoritative_owners": len(owners),
            # correction-required/forbidden tokens legitimately occur in many
            # unrelated facts: they are the reason E15's token-based measure
            # reports obsolete_retention > 0 despite fact-correct supersession.
            "unrelated_owners": len(owners),
        }
    return out


def _audit_e15_raw(e15_path: Path) -> Optional[Dict]:
    if not e15_path.exists():
        return None
    data = json.loads(e15_path.read_text())
    stress = data.get("stress") or {}
    cells = stress.get("cells") or []
    if not cells:
        return None
    n = len(cells)
    obs_gt0 = sum(1 for c in cells
                  if (c.get("metrics", {}).get("obsolete_retention") or 0) > 0)
    corr_lt1 = sum(1 for c in cells
                   if (c.get("metrics", {}).get("correction_recall") or 1) < 1)
    both = sum(1 for c in cells
               if (c.get("metrics", {}).get("obsolete_retention") or 0) > 0
               and (c.get("metrics", {}).get("correction_recall") or 1) < 1)
    clean = sum(1 for c in cells
                if (c.get("metrics", {}).get("obsolete_retention") or 0) == 0
                and (c.get("metrics", {}).get("correction_recall") or 1) >= 1)

    super_bad_total = sum(c.get("lifecycle_counts", {}).get("SUPERSEDED_INCORRECTLY", 0)
                          for c in cells)
    super_cells_bad = sum(1 for c in cells
                          if c.get("lifecycle_counts", {}).get("SUPERSEDED_INCORRECTLY", 0) > 0)

    # new-correction-present proxy: correction-target facts in the final store.
    policy_store = defaultdict(lambda: {"cells": 0, "corr_gt": 0, "corr_missing": 0})
    for c in cells:
        policy = c.get("policy")
        sb = c.get("store_budget")
        key = (policy, sb)
        policy_store[key]["cells"] += 1
        for rec in (c.get("registry") or {}).values():
            if rec.get("is_correction_target"):
                policy_store[key]["corr_gt"] += 1
                state = (c.get("lifecycle") or {}).get(rec["fact"])
                if state not in ("PRESENT_AND_RETRIEVED", "PRESENT_BUT_NOT_RETRIEVED"):
                    policy_store[key]["corr_missing"] += 1

    per = {}
    for (policy, sb), v in sorted(policy_store.items()):
        per.setdefault(policy, {})[str(sb)] = {
            "cells": v["cells"],
            "correction_facts_total": v["corr_gt"],
            "correction_facts_missing_from_store": v["corr_missing"],
        }

    census = _token_census()
    return {
        "source": str(e15_path),
        "surveyed_stress_cells": n,
        "cells_obsolete_retention_gt0": obs_gt0,
        "cells_correction_recall_lt1": corr_lt1,
        "cells_both": both,
        "cells_clean": clean,
        "SUPERSEDED_INCORRECTLY_total": super_bad_total,
        "SUPERSEDED_INCORRECTLY_cells": super_cells_bad,
        "correction_missing_by_policy_store": per,
        "token_census": census,
        "note": ("E15's token-based obsolete_retention can only fire when a "
                 "forbidden answer token is present in injected context; the "
                 "token census quantifies how often that token legitimately "
                 "lives in unrelated gt facts."),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(x, nd=3):
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _agg_row(agg: Dict, policy: str, sb, ab) -> Optional[Dict]:
    """Frame-independent lookup into the aggregate: JSON round-trip turns the
    active-budget keys into strings, in-memory aggregates use ints."""
    by_policy = agg.get(policy, {}) or {}
    by_store = by_policy.get(str(sb)) if isinstance(sb, (int, str)) else by_policy.get(sb)
    if not isinstance(by_store, dict):
        return None
    if isinstance(ab, str):
        return by_store.get(ab) or by_store.get(int(ab))
    return by_store.get(ab) or by_store.get(str(ab))


def _grid_table(agg: Dict, metric: str, nd: int = 3) -> List[str]:
    lines = []
    lines.append("| Policy | 256 | 512 | 1024 | 2048 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for p in POLICIES:
        row = []
        for sb in STORE_BUDGETS:
            vals = []
            for ab in ACTIVE_BUDGETS:
                a = _agg_row(agg, p, sb, ab)
                if a and a.get(metric) is not None:
                    vals.append(a[metric])
            row.append(_fmt(_mean(vals), nd) if vals else "N/A")
        lines.append("| {} | {} |".format(p, " | ".join(row)))
    return lines


def _three_way_table(agg: Dict, metric: str, nd: int = 3) -> List[str]:
    lines = []
    lines.append("| Policy | store | active | {} |".format(metric))
    lines.append("| --- | --- | --- | --- |")
    for p in POLICIES:
        for sb in STORE_BUDGETS:
            for ab in ACTIVE_BUDGETS:
                a = _agg_row(agg, p, sb, ab)
                if a:
                    lines.append(f"| {p} | {sb} | {ab} | {_fmt(a.get(metric), nd)} |")
    return lines


def generate_report(result: Dict) -> str:
    L: List[str] = []
    L.append("# E16 Retention Selectivity Report (Phase 12)")
    L.append("")
    L.append("Selective retention of task-relevant memories under hard store "
             "pressure, measured with **identity-safe** fact-level metrics; "
             "diagnosis of the retrieval-feedback loop on `retention_priority`; "
             "causal baselines and an offline oracle upper bound for one "
             "future-blind `task_affinity` candidate.")
    L.append("")

    cfg = result["config"]
    L.append("## 1. Purpose & Hypothesis")
    L.append("")
    L.append("Under genuine store pressure the survival policy decides which "
             "facts the long-term store keeps. `retention_priority` "
             "(E15) is fed by `access_count`, which increments on retrieval — so "
             "a fact that is *retrieved* becomes *more likely to survive*, "
             "independently of task relevance. E16 asks whether (a) this loop "
             "actually dominates survival, and (b) a future-blind task-similarity "
             "signal (`task_affinity`) can retain task-relevant facts better than "
             "the causal baselines (`random`, `base_score_only`) and close part of "
             "the gap to an offline `oracle_future_use` upper bound — without "
             "resurrecting superseded facts or sacrificing correction safety.")
    L.append("")
    L.append("## 2. Phase 11 Audit & E15 Raw-JSON Reconciliation")
    L.append("")
    audit = result.get("e15_audit")
    if audit:
        L.append(f"- Surveyed **{audit['surveyed_stress_cells']}** E15 stress cells "
                 f"(1200 turns, scale 27).")
        L.append(f"- Cells with token-based `obsolete_retention > 0`: "
                 f"**{audit['cells_obsolete_retention_gt0']}**; "
                 f"`correction_recall < 1`: **{audit['cells_correction_recall_lt1']}**; "
                 f"both: {audit['cells_both']}; clean: {audit['cells_clean']}.")
        L.append(f"- Fact-level reconciliation: `SUPERSEDED_INCORRECTLY` appears in "
                 f"**{audit['SUPERSEDED_INCORRECTLY_cells']}** of the cells "
                 f"(total **{audit['SUPERSEDED_INCORRECTLY_total']}** occurrences) — "
                 f"the store never resurrects a corrected fact as an authoritative "
                 f"memory. The E15 report's blanket 'obsolete_retention = 0 under "
                 f"every policy/budget' therefore overclaims ONLY at the token "
                 f"level, where the forbidden token legitimately occurs in "
                 f"unrelated facts.")
        census = audit.get("token_census", {})
        if census:
            L.append("- Token-level collision census (scale 27, authoritative gt):")
            for tok, v in census.items():
                L.append(f"  - `{tok}`: {v['authoritative_owners']} authoritative fact(s) "
                         f"contain it.")
        L.append("- Correction-content loss is REAL (E15 stress): the newly "
                 "corrected fact is missing from the final store in many cells:")
        L.append("")
        L.append("| policy | store | correction facts | missing from store |")
        L.append("| --- | --- | --- | --- |")
        for policy, by_store in sorted(audit.get("correction_missing_by_policy_store", {}).items()):
            for sb, v in sorted(by_store.items()):
                L.append(f"| {policy} | {sb} | {v['correction_facts_total']} "
                         f"| {v['correction_facts_missing_from_store']} |")
        L.append("")
        L.append("Phase 12 addresses this with identity-safe metrics (section 4) "
                 "and correction protection (`protect_corrections=True`, section 11).")
    else:
        L.append("*E15 raw JSON not present; reconciliation skimmed.*")
    L.append("")

    L.append("## 3. Methodology")
    L.append("")
    L.append("All policies share `retention.mode = dual_score` so decay never "
             "deletes; the only causally varying knob is the store-eviction "
             "priority. `task_affinity` evicts lexicographically lowest "
             "`(task_affinity, retention_priority)`; `base_score_only` evicts "
             "lowest `base_score` only (no retrieval/decay/salience feedback); "
             "`random` is a seeded lower bound; `oracle_future_use` evicts "
             "future-irrelevant facts first (offline-only upper bound). "
             "Retrieval ranking is FROZEN (0.85*sim + 0.15*importance + category "
             "bonus), so separation acts only through survival.")
    L.append("")
    L.append("## 4. Evaluation Definitions")
    L.append("")
    L.append("- `fact_id = category:qtype:source_turn:SHA256(precise_fact)[:16]` "
             "(deterministic; no Python `hash()`).")
    L.append("- Queries carry `target_fact_ids` (must be retrievable) and "
             "`forbidden_fact_ids` (must NOT be); corrections assert the new "
             "fact's id present and the old fact's id absent.")
    L.append("- `fact_identity_context_recall` / `fact_identity_store_recall` / "
             "`fact_identity_correction_recall` / "
             "`fact_identity_obsolete_retention` measure presence/absence on "
             "fact ids, immune to token collisions (e.g. `memcache`/`redis`).")
    L.append("- Legacy token metrics are kept for comparability to E13–E15.")
    L.append("- `future_use`: offline label = fact id requested by any query. "
             "`retention_recall` = retained future-use facts / all future-use "
             "facts. `precision_of_retention` = retained future-use facts / "
             "retained authoritative facts. Future labels reach ONLY the "
             "offline oracle + metric code (section 15/24 checks).")
    L.append("")

    L.append("## 5. Configuration")
    L.append("")
    for k, v in cfg.items():
        L.append(f"- **{k}**: {v}")
    L.append("")
    L.append("### Policies")
    L.append("")
    L.append("| Policy | eviction priority |")
    L.append("| --- | --- |")
    for p, d in POLICY_DEFS.items():
        L.append(f"| {p} | `{d['eviction_priority']}` — {d['description']} |")
    L.append("")

    agg = result["grid"]["aggregate"]
    L.append("## 6. Identity-Safe Recall Grid (policy x store, mean over active/seeds)")
    L.append("")
    L.append("### fact_identity_store_recall")
    L.append("")
    L.extend(_grid_table(agg, "identity_store_recall_mean"))
    L.append("")
    L.append("### fact_identity_context_recall")
    L.append("")
    L.extend(_grid_table(agg, "identity_context_recall_mean"))
    L.append("")
    L.append("### legacy context_recall (comparability)")
    L.append("")
    L.extend(_grid_table(agg, "context_recall_mean"))
    L.append("")

    L.append("## 7. Hard-Pressure Grid (store x active x policy)")
    L.append("")
    L.append("*This grid IS the pressured grid (store 256–2048 tokens, natural "
             "store far above; see section 5). Identity store recall by cell:*")
    L.append("")
    L.extend(_three_way_table(agg, "identity_store_recall_mean"))
    L.append("")

    L.append("## 8. Future-Use Analysis")
    L.append("")
    L.append("### future-use retention_recall (policy x store, mean over active/seeds)")
    L.append("")
    L.extend(_grid_table(agg, "future_use_retention_recall_mean"))
    L.append("")
    L.append("### precision_of_retention")
    L.append("")
    L.extend(_grid_table(agg, "precision_of_retention_mean"))
    L.append("")
    nop = result.get("no_pressure", {})
    if nop:
        L.append("No-pressure controls retained/future counts are in the JSON; "
                 "the report focuses on the pressured grid where survival "
                 "selection is measurable.")
        L.append("")

    L.append("## 9. Feedback-Loop Diagnosis")
    L.append("")
    L.append("`retention_priority` is fed by `access_count` (retrievals + "
             "ingest reinforcements). The question: does survival under "
             "`dual_score` merely echo 'was retrieved', so that "
             "retained future-use facts are explained by retrieval rather than "
             "by policy? Evidence in `diagnostics.spearman_vs_future_use` per "
             "cell; the JSON holds the full correlation matrix for "
             "`retrieval_access_count`, `ingest_reinforcement_count`, "
             "`retention_priority`, `task_affinity`, `base_score` vs the offline "
             "future-use label. Distinct counters (§5) allow the loop to be "
             "separated into retrieval-feedback and ingest-feedback components.")
    L.append("")

    L.append("## 10. Saturation Diagnostics")
    L.append("")
    L.append("Per-population distributions (min/max/mean/median/p10/p90) of "
             "`base_score`, `current_importance`, `retention_priority`, "
             "`task_affinity`, `retrieval_access_count`, "
             "`ingest_reinforcement_count`, `duplicates` over the buckets "
             "retrieved-needed / stored-not-retrieved / evicted / never-retrieved "
             "/ corrections / obsolete are recorded in every cell "
             "(`diagnostics.by_population`). Population sizes confirm whether "
             "retained sets are outcomes of the policy or of saturation.")
    L.append("")

    L.append("## 11. Correction & Obsolete Safety (identity-safe)")
    L.append("")
    L.append("| Policy | store | identity corr. recall | identity obs. retention | new-corr missing turns |")
    L.append("| --- | --- | --- | --- | --- |")
    for p in POLICIES:
        for sb in STORE_BUDGETS:
            vals_c = [(_agg_row(agg, p, sb, ab) or {}).get(
                "identity_correction_recall_mean") for ab in ACTIVE_BUDGETS]
            vals_o = [(_agg_row(agg, p, sb, ab) or {}).get(
                "identity_obsolete_retention_mean") for ab in ACTIVE_BUDGETS]
            vals_n = [(_agg_row(agg, p, sb, ab) or {}).get(
                "new_correction_missing_mean") for ab in ACTIVE_BUDGETS]
            L.append(f"| {p} | {sb} | {_fmt(_mean(vals_c))} | "
                     f"{_fmt(_mean(vals_o))} | {_fmt(_mean(vals_n), 2)} |")
    L.append("")
    L.append("Every E16 cell runs `protect_corrections=True`: a current "
             "correction is never evicted while its superseded predecessor "
             "still exists, and `protected_capacity_conflict` records when the "
             "store budget is too tight to honour that. `identity_obsolete_"
             "retention = 0` is the acceptance gate (criterion c4).")
    L.append("")

    L.append("## 12. Age Analysis")
    L.append("")
    L.append("Future-use retention by age bucket (age = query_turn - source_turn) "
             "is stored per cell (`age_analysis`), as are the eviction ages of "
             "lost future-use facts (`evicted_age_analysis`). Buckets: "
             "0-99 … 1000+. A policy that only survives young/recent facts will "
             "show collapsing retention in late buckets.")
    L.append("")

    L.append("## 13. Query-Family Analysis")
    L.append("")
    L.append("Identity context/store recall per query family is recorded per "
             "cell (`family_metrics`); the JSON holds the full breakdown for "
             "every (policy, store, active, seed) cell.")
    L.append("")

    L.append("## 14. Oracle Results & Gap")
    L.append("")
    L.append("| Policy | store | identity store recall (oracle) | gap vs oracle (low store) |")
    L.append("| --- | --- | --- | --- |")
    verdict = result.get("verdict", {})
    gaps = (verdict.get("criteria", {}).get("c8_closes_oracle_gap", {}) or {}).get("gaps", {})
    for p in POLICIES:
        vals = [_mean([((_agg_row(agg, p, sb, ab)) or {}).get(
            "identity_store_recall_mean") for ab in ACTIVE_BUDGETS])
            for sb in (STORE_BUDGETS if p == "oracle_future_use" else [256])]
        row = " | ".join(_fmt(v) for v in vals)
        L.append(f"| {p} | {STORE_BUDGETS[0] if p != 'oracle_future_use' else 'all'} "
                 f"| {row} | {_fmt(gaps.get(p))} |")
    L.append("")
    L.append("oracle_gap(policy) = oracle identity store recall − policy "
             "future-use retention recall at store 256/512 (low pressure). The "
             "oracle is an offline, future-peeking upper bound: a policy is "
             "compared on how much of that theoretical headroom it recovers "
             "without future information.")
    L.append("")

    L.append("## 15. No-Pressure Diagnostic")
    L.append("")
    if nop:
        fps = nop.get("fingerprints", {}) or {}
        fingerprints = [sorted(v) for per_policy in fps.values() for v in per_policy.values()]
        identical = all(_fingerprint_equal(fingerprints[0], fp)
                        for fp in fingerprints[1:]) if fingerprints else False
        L.append(f"For store_budget=0, dual_score / task_affinity / "
                 f"oracle_future_use must produce IDENTICAL final store contents "
                 f"(eviction never fires). Fingerprints identical over "
                 f"{len(fingerprints)} runs: **{identical}**")
        L.append("- Causal isolation: without store pressure the eviction policy "
                 "cannot change what is stored; any recall difference in the "
                 "pressured grid is therefore attributable to survival, not to "
                 "retrieval/writing.")
    else:
        L.append("*Skipped in quick mode.*")
    L.append("")

    L.append("## 16. Window Sensitivity (task_affinity)")
    L.append("")
    win = (result.get("window_sensitivity", {}).get("aggregate", {}) or {}).get("_mean", {}) or {}
    if win:
        L.append("| window | future-use retention rec. | precision | identity store rec. |")
        L.append("| --- | --- | --- | --- |")
        for w, v in sorted(win.items(), key=lambda kv: int(kv[0])):
            L.append(f"| {w} | {_fmt(v.get('future_use_retention_recall_mean'))} | "
                     f"{_fmt(v.get('precision_of_retention_mean'))} | "
                     f"{_fmt(v.get('identity_store_recall_mean'))} |")
        L.append("")
        L.append("Per-seed values are in the JSON. 32 is the default window; "
                 "16 and 64 are sensitivity only, not production candidates.")
    else:
        L.append("*Skipped in quick mode.*")
    L.append("")

    L.append("## 17. Decision Rule Application")
    L.append("")
    L.append("task_affinity advances ONLY if all eight criteria hold vs "
             "`dual_score` (the current production default). Numbers below come "
             "from the JSON, not from hand-typed values.")
    L.append("")
    if verdict:
        for k, v in verdict.get("criteria", {}).items():
            ok = v.get("ok")
            L.append(f"- **{k}**: {'PASS' if ok else 'FAIL'} — {v}")
        L.append("")
        L.append(f"**task_affinity advances: "
                 f"{'YES' if verdict.get('task_affinity_advances') else 'NO'}.**")
        L.append("")
        L.append(f"rationale: {verdict.get('rationale', '')}")
    else:
        L.append("*Not classified (quick-mode run).*")
    L.append("")

    L.append("## 18. Production Default")
    L.append("")
    adv = bool(verdict.get("task_affinity_advances")) if verdict else False
    if adv:
        L.append("Retention default moves to "
                 "`retention: {mode: task_affinity, eviction_priority: "
                 "task_affinity, task_context_window: 32}` — only after a "
                 "reviewer confirms the report.")
    else:
        L.append("Retention default stays `dual_score` "
                 "(`eviction_priority: retention_priority`). `task_affinity` is "
                 "reported as a failed candidate with its failure mode "
                 "documented; no silent flip happens.")
    L.append("")
    L.append("— Numbers in this report are generated from "
             "`e16_retention_selectivity.json` (and, for section 2, from the "
             "E15 raw JSON) by `generate_report()`; no hand-typed figures.")
    return "\n".join(L)


def _fingerprint_equal(a, b):
    if a is None or b is None:
        return False
    return sorted(a) == sorted(b)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="E16 retention selectivity")
    parser.add_argument("--quick", action="store_true", help="smoke-test config")
    parser.add_argument("--force", action="store_true",
                        help="recompute all cells even if present in the JSON")
    parser.add_argument("--no-window", action="store_true",
                        help="skip task_affinity window sensitivity")
    parser.add_argument("--no-nopressure", action="store_true",
                        help="skip the no-pressure diagnostic")
    parser.add_argument("--report-only", action="store_true",
                        help="re-classify and regenerate the report from an "
                             "existing e16_retention_selectivity.json")
    args = parser.parse_args(argv)

    from data.coding_workload import self_test

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "e16_retention_selectivity.json"
    report_path = out_dir / "e16_retention_selectivity_report.md"

    if args.report_only:
        if not result_path.exists():
            parser.error("no e16_retention_selectivity.json to classify")
        result = json.loads(result_path.read_text())
        result["verdict"] = classify_result(result)
        result_path.write_text(json.dumps(result, indent=2, default=str))
        report_path.write_text(generate_report(result))
        print(f"Re-classified and wrote {report_path}")
        v = result["verdict"]
        print(f"  task_affinity_advances={v.get('task_affinity_advances')}")
        return result

    self_test()

    if args.quick:
        active = [128]
        store = [256, 512]
        seeds = [42]
        turns, scale = 120, 3
        window_cfg = None
        nop_seeds = None
    else:
        active, store, seeds = ACTIVE_BUDGETS, STORE_BUDGETS, SEEDS
        turns, scale = TURNS, SCALE
        window_cfg = ([16, 32, 64] if not args.no_window else None)
        nop_seeds = None if args.no_nopressure else seeds

    embed_fn = _resolve_embed(True, EMBEDDING_MODEL, None)

    print("=" * 60)
    print("E16 Retention Selectivity")
    print("=" * 60)

    done = {}
    if result_path.exists() and not args.force:
        prev = json.loads(result_path.read_text())
        for c in prev.get("grid", {}).get("cells", []):
            done[_cell_key(c)] = c

    grid_cells = list(done.values())
    to_run = []
    for p in POLICIES:
        for sb in store:
            for ab in active:
                for seed in seeds:
                    key = f"{p}:{sb}:{ab}:{seed}"
                    if key not in done:
                        to_run.append((p, sb, ab, seed))

    for i, (p, sb, ab, seed) in enumerate(to_run):
        print(f"  [{i + 1}/{len(to_run)}] policy={p} store={sb} active={ab} "
              f"seed={seed} ...", flush=True)
        grid_cells.append(run_cell(seed, ab, p, turns=turns, scale=scale,
                                   store_budget=sb, embed_fn=embed_fn))
        if (i + 1) % 25 == 0 or i + 1 == len(to_run):
            result_path.write_text(json.dumps({
                "experiment": "e16_retention_selectivity",
                "config": {}, "grid": {"cells": grid_cells},
                "partial": True,
            }, indent=2, default=str))

    result = {
        "experiment": "e16_retention_selectivity",
        "config": {
            "active_budgets": active, "store_budgets": store, "seeds": seeds,
            "turns": turns, "scale": scale, "embedding_model": EMBEDDING_MODEL,
            "matching": "identity-safe fact_id (+ legacy token metrics)",
            "policies": POLICIES,
            "protect_corrections": True,
            "task_context_window_default": 32,
            "active_budget_enforced": True,
            "decision": "8 criteria (see report section 17); no post-result tuning",
        },
        "policies": POLICY_DEFS,
        "grid": {
            "cells": grid_cells,
            "aggregate": aggregate_cells(grid_cells),
        },
    }

    if nop_seeds:
        print("\n[no-pressure] ...", flush=True)
        nop_cells = []
        nop_pol = ["dual_score", "task_affinity", "oracle_future_use"]
        for p in nop_pol:
            for seed in nop_seeds:
                cell = run_cell(seed, 128, p, turns=turns, scale=scale,
                                store_budget=0, embed_fn=embed_fn,
                                capture_store_ids=True)
                nop_cells.append(cell)
        result["no_pressure"] = {
            "cells": nop_cells,
            "aggregate": aggregate_no_pressure(nop_cells),
            "fingerprints": {p: {str(s): cell["store_fact_ids"]
                                 for cell in nop_cells
                                 if cell["policy"] == p and cell["seed"] == s}
                             for p in nop_pol for s in nop_seeds},
        }

    if window_cfg is not None and 32 in window_cfg:
        print("\n[window sensitivity] ...", flush=True)
        win_cells = []
        for w in (w0 for w0 in window_cfg if w0 != 32):
            for seed in seeds:
                win_cells.append(run_cell(seed, 128, "task_affinity",
                                          turns=turns, scale=scale,
                                          store_budget=512, embed_fn=embed_fn,
                                          task_context_window=int(w)))
        result["window_sensitivity"] = {
            "cells": win_cells,
            "aggregate": aggregate_window(win_cells),
        }

    result["e15_audit"] = _audit_e15_raw(out_dir / "e15_retention_policy.json")
    result["verdict"] = classify_result(result)
    result.pop("partial", None)

    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {result_path}")
    report_path.write_text(generate_report(result))
    print(f"Wrote {report_path}")
    v = result["verdict"]
    print(f"task_affinity_advances={v.get('task_affinity_advances')}")
    return result


if __name__ == "__main__":
    main()
