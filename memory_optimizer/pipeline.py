"""L — unified adaptive memory pipeline (task L).

Single runner shared by the live server ingestion and the offline paper replay,
so the component that answers questions in production is byte-for-byte the one
measured in the paper (D1–D8 components unchanged; this only extracts the
orchestration everyone already used).

Clean store API:

    pipe = AdaptiveMemoryPipeline.from_settings(settings)      # paper replay
    pipe2 = AdaptiveMemoryPipeline(settings, scorer, decay, retriever, compressor,
                                   store=my_list, pruned=my_list)   # live server

    result = pipe.ingest(turn_id, user_message, facts, fact_tokens=fn, embed_fn=fn)
        -> {"injected", "injected_tokens", "pruned", "budget_evictions", "active_tokens"}
    pipe.retrieve(query)                       # direct retrieval against the store
    pipe.active_memories / pipe.pruned_memories / pipe.last_budget_evictions

Stages are the same five*. latencies are reported through ``on_stage(op, ms)``
so the live server can keep its latency stats feed without the pipeline knowing
about the API.
"""

from __future__ import annotations

import random
import time
from typing import Callable, Dict, List, Optional

from memory_optimizer.budget import token_budget_evict
from memory_optimizer.compression import MemoryCompressor
from memory_optimizer.decay import CategoryDecayEngine
from memory_optimizer.retrieval import MemoryRetriever
from memory_optimizer.scoring import ImportanceScorer, _write_time_salience
from memory_optimizer.task_state import TaskStateTracker

PercentCallback = Optional[Callable[[str, float], None]]


def _word_tokens(text: str) -> int:
    return max(1, len(str(text).split()))


# eviction_priority -> list of budget.py priority keys (lexicographic).
_EVICTION_KEYS = {
    "current_importance": ["current_importance"],
    "retention_priority": ["retention_priority"],
    "base_score": ["base_score"],
    "base_score_only": ["base_score"],
    "task_affinity": ["task_affinity", "retention_priority"],
    "oracle_future_use": ["oracle_future_use", "retention_priority"],
    "random": ["_evict_random"],
}


def _priority_keys(eviction_priority: str) -> List[str]:
    return list(_EVICTION_KEYS.get(eviction_priority, [eviction_priority]))


class AdaptiveMemoryPipeline:
    def __init__(self, settings: Dict, scorer: ImportanceScorer = None,
                 decay: CategoryDecayEngine = None, retriever: MemoryRetriever = None,
                 compressor: MemoryCompressor = None, store: List = None,
                 pruned: List = None, on_stage: PercentCallback = None):
        self.settings = settings
        self.scorer = scorer
        self.decay = decay
        self.retriever = retriever
        self.compressor = compressor
        self.memories = store if store is not None else []
        self.pruned_memories = pruned if pruned is not None else []
        self.last_budget_evictions: List[Dict] = []
        self.last_store_pressure: Dict = {}
        self.last_protected_capacity_conflict: bool = False
        self.on_stage = on_stage
        retention_cfg = settings.get("retention", {}) if settings else {}
        self.protect_corrections = bool(retention_cfg.get("protect_corrections", False))
        task_window = int(retention_cfg.get("task_context_window", 32))
        self._task_state = TaskStateTracker(task_window)
        self._evict_rng = random.Random(int(retention_cfg.get("random_seed", 0)))

    @classmethod
    def from_settings(cls, settings: Dict, embed_fn=None, embedding_model: str = "",
                      store: List = None, pruned: List = None) -> "AdaptiveMemoryPipeline":
        scorer = ImportanceScorer(weights=settings["scoring_weights"])
        decay = CategoryDecayEngine(lambdas=settings["decay_lambdas"],
                                    pruning_threshold=float(settings["pruning"]["threshold"]))
        retriever = MemoryRetriever(top_k=int(settings["top_k"]),
                                    sim_threshold=float(settings.get("similarity_threshold", 0.35)),
                                    embed_fn=embed_fn, embedding_model=embedding_model)
        compressor = MemoryCompressor()
        return cls(settings, scorer, decay, retriever, compressor, store, pruned)

    def _latency(self, stage: str, t0: float) -> None:
        if self.on_stage is not None:
            self.on_stage(stage, (time.perf_counter() - t0) * 1000)

    def _tok(self, text: str, fact_tokens) -> int:
        if fact_tokens is not None:
            v = fact_tokens(text)
            if v is not None:
                return int(v)
        return _word_tokens(text)

    def _ez_embed(self, text: str, embed_fn):
        if embed_fn is None:
            return None
        try:
            v = embed_fn([text])
            return v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v
        except Exception:
            return None

    def _stamp_random_keys(self) -> None:
        """Stamp deterministic per-memory random eviction keys (policy `random`).

        Stamped once per memory so repeated sorts within an eviction pass stay
        stable; the RNG is seeded from ``retention.random_seed`` so a cell is
        reproducible.
        """
        for m in self.memories:
            if "_evict_random" not in m:
                m["_evict_random"] = self._evict_rng.random()

    def _stamp_oracle_keys(self) -> None:
        """Stamp offline future-use labels for the `oracle_future_use` policy.

        The labels come from ``retention.oracle_future_use`` (a fact_id -> 1
        mapping), an OFFLINE-ONLY quantity: the oracle is a theoretical upper
        bound that is allowed to peek at the future. It is evaluated at eviction
        time so a fact that was identity-replaced by a correction receives the
        correcting fact's label.
        """
        fut = self.settings.get("retention", {}).get("oracle_future_use") or {}
        for m in self.memories:
            m["oracle_future_use"] = 1 if m.get("fact_id") in fut else 0

    def _compute_task_affinities(self, embed_fn) -> None:
        """Vectorized task_affinity for every store memory (policy task_affinity).

        Embeddings are computed once (and cached on the memory for later reuse
        by the retriever); affinity uses the tracker's top-k message window, so
        it is a pure function of turns seen so far.
        """
        for m in self.memories:
            if m.get("fact_embedding") is None:
                m["fact_embedding"] = self._ez_embed(m["fact"], embed_fn)
        vals = self._task_state.batch_affinity(
            [m.get("fact_embedding") for m in self.memories])
        for m, v in zip(self.memories, vals):
            m["task_affinity"] = float(v)

    def _eviction_record(self, m: Dict) -> Dict:
        return {
            "fact": m["fact"],
            "fact_id": m.get("fact_id"),
            "category": m.get("category", ""),
            "base_score": round(float(m.get("base_score", 0)), 3),
            "current_importance": round(m.get("current_importance", 0), 3),
            "retention_priority": round(m.get("retention_priority", 0), 3),
            "task_affinity": round(float(m.get("task_affinity", 0)), 3),
            "retrieval_access_count": int(m.get("retrieval_access_count", 0)),
            "ingest_reinforcement_count": int(m.get("ingest_reinforcement_count", 0)),
            "access_count": int(m.get("access_count", 1)),
            "duplicates": int(m.get("duplicates", 1)),
            "source_turn_id": m.get("source_turn_id"),
            "last_access_turn": m.get("last_access_turn"),
            "is_current_correction": bool(m.get("is_current_correction", False)),
            "superseded_prior_fact_id": m.get("superseded_prior_fact_id"),
            "measured_tokens": m.get("measured_tokens"),
            "eviction_reason": m.get("eviction_reason"),
        }

    def ingest(self, turn_id: int, user_message: str, facts: List[Dict],
               fact_tokens=None, embed_fn=None, query_relevance: Optional[float] = None) -> Dict:
        active_context_budget = int(self.settings["max_context_tokens"])

        injection_limit = int(self.settings.get("injection_token_limit", 0))
        if injection_limit <= 0:
            injection_limit = active_context_budget

        store_budget = int(self.settings.get("memory_store_token_budget", 0))

        t0 = time.perf_counter()
        new_facts = []
        for item in facts:
            it = dict(item)
            it["source_turn_id"] = it.get("source_turn_id", turn_id)
            it["last_access_turn"] = turn_id
            relevance = (query_relevance if query_relevance is not None
                         else _write_time_salience(user_message, it["fact"], embed_fn=embed_fn))
            it["write_time_salience"] = relevance
            it["base_score"] = self.scorer.compute_score(it, query_relevance=relevance,
                                                         current_turn=turn_id)
            new_facts.append(it)
        self._latency("scoring", t0)

        t0 = time.perf_counter()
        for item in new_facts:
            if fact_tokens is not None:
                measured = fact_tokens(item["fact"])
                if measured is not None:
                    item["measured_tokens"] = int(measured)
        self._latency("fact_token_measurement", t0)

        t0 = time.perf_counter()
        if self.settings.get("enable_compression", True):
            self.memories = self.compressor.dedupe_incremental(self.memories, new_facts,
                                                               embed_fn=embed_fn)
        else:
            self.memories.extend(new_facts)
        self._latency("compression", t0)

        t0 = time.perf_counter()
        retention_cfg = self.settings.get("retention", {})
        retention_mode = retention_cfg.get("mode", "hard_threshold")
        eviction_priority = retention_cfg.get("eviction_priority", "current_importance")
        active, pruned_this_turn = self.decay.step_decay_and_prune(
            self.memories, turn_id, retention_mode=retention_mode)
        self.memories = active
        self.pruned_memories.extend(pruned_this_turn)
        self._latency("decay", t0)

        t0 = time.perf_counter()
        if store_budget > 0 and fact_tokens is not None:
            if eviction_priority == "task_affinity":
                self._task_state.observe(turn_id, user_message, facts, embed_fn)
                self._compute_task_affinities(embed_fn)
            elif eviction_priority == "random":
                self._stamp_random_keys()
            elif eviction_priority == "oracle_future_use":
                self._stamp_oracle_keys()
            keys = _priority_keys(eviction_priority)
            pre_store_tokens = sum(fact_tokens(m["fact"]) for m in self.memories)
            if self.protect_corrections:
                protected = [m for m in self.memories
                             if m.get("is_current_correction")
                             or m.get("superseded_prior_fact_id") is not None]
                protected_ids = {id(m) for m in protected}
                evictable = [m for m in self.memories if id(m) not in protected_ids]
            else:
                protected, evictable = [], list(self.memories)
            kept_e, evicted_by_budget, _post_evictable = token_budget_evict(
                evictable, store_budget, fact_tokens, priority_key=keys)
            kept = kept_e + protected
            self.memories = kept
            self.last_budget_evictions = [self._eviction_record(m)
                                          for m in evicted_by_budget]
            post_store_tokens = sum(fact_tokens(m["fact"]) for m in kept)
            self.last_protected_capacity_conflict = post_store_tokens > store_budget
            # Store-pressure instrumentation
            self.last_store_pressure = {
                "pre_store_tokens": pre_store_tokens,
                "store_budget": store_budget,
                "store_over_budget_before_eviction": max(0, pre_store_tokens - store_budget),
                "evicted_count": len(evicted_by_budget),
                "post_store_tokens": post_store_tokens,
                "eviction_occurred": len(evicted_by_budget) > 0,
                "protected_count": len(protected),
                "protected_capacity_conflict": self.last_protected_capacity_conflict,
            }
        else:
            # No independent store cap: do not confuse active-context capacity
            # with long-term memory capacity.
            self.last_budget_evictions = []
            self.last_protected_capacity_conflict = False
            self.last_store_pressure = {
                "pre_store_tokens": sum(fact_tokens(m["fact"]) for m in self.memories) if fact_tokens else 0,
                "store_budget": store_budget,
                "store_over_budget_before_eviction": 0,
                "evicted_count": 0,
                "post_store_tokens": sum(fact_tokens(m["fact"]) for m in self.memories) if fact_tokens else 0,
                "eviction_occurred": False,
                "protected_count": 0,
                "protected_capacity_conflict": False,
            }
        self._latency("budget_evict", t0)

        t0 = time.perf_counter()
        injected = self.retriever.retrieve(user_message, self.memories, current_turn=turn_id,
                                           token_limit=injection_limit, fact_tokens=fact_tokens)
        if getattr(self.retriever.stats, "get", None) and self.retriever.stats.get("embed_ms") \
                and self.on_stage is not None:
            self.on_stage("embedding_ms", float(self.retriever.stats["embed_ms"]))
        self._latency("retrieval", t0)

        injected_tokens = sum(self._tok(f["fact"], fact_tokens) for f in injected)
        active_tokens = sum(m.get("measured_tokens") for m in self.memories
                            if m.get("measured_tokens") is not None)
        return {
            "injected": injected,
            "injected_tokens": injected_tokens,
            "pruned": pruned_this_turn,
            "budget_evictions": self.last_budget_evictions,
            "store_pressure": self.last_store_pressure,
            "active_tokens": active_tokens,
        }

    @property
    def active_memories(self) -> List:
        return self.memories

    def retrieve(self, query: str, current_turn: int = None, token_limit: int = 0,
                 fact_tokens=None) -> List[Dict]:
        return self.retriever.retrieve(query, self.memories, current_turn=current_turn,
                                       token_limit=token_limit, fact_tokens=fact_tokens)