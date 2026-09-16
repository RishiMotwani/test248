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

import time
from typing import Callable, Dict, List, Optional

from memory_optimizer.budget import token_budget_evict
from memory_optimizer.compression import MemoryCompressor
from memory_optimizer.decay import CategoryDecayEngine
from memory_optimizer.retrieval import MemoryRetriever
from memory_optimizer.scoring import ImportanceScorer

PercentCallback = Optional[Callable[[str, float], None]]


def _word_tokens(text: str) -> int:
    return max(1, len(str(text).split()))


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
        self.on_stage = on_stage

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

    def ingest(self, turn_id: int, user_message: str, facts: List[Dict],
               fact_tokens=None, embed_fn=None, query_relevance: float = 0.8) -> Dict:
        budget = int(self.settings["max_context_tokens"])
        inj_limit = int(self.settings.get("injection_token_limit", 0))

        t0 = time.perf_counter()
        new_facts = []
        for item in facts:
            it = dict(item)
            it["source_turn_id"] = it.get("source_turn_id", turn_id)
            it["last_access_turn"] = turn_id
            it["base_score"] = self.scorer.compute_score(it, query_relevance=query_relevance,
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
        active, pruned_this_turn = self.decay.step_decay_and_prune(self.memories, turn_id)
        self.memories = active
        self.pruned_memories.extend(pruned_this_turn)
        self._latency("decay", t0)

        t0 = time.perf_counter()
        if fact_tokens is not None:
            kept, evicted_by_budget, _ = token_budget_evict(self.memories, budget, fact_tokens)
            self.memories = kept
            self.last_budget_evictions = [{
                "fact": m["fact"],
                "category": m.get("category", ""),
                "current_importance": round(m.get("current_importance", 0), 3),
                "source_turn_id": m.get("source_turn_id"),
            } for m in evicted_by_budget]
        else:
            self.last_budget_evictions = []
        self._latency("budget_evict", t0)

        t0 = time.perf_counter()
        injected = self.retriever.retrieve(user_message, self.memories, current_turn=turn_id,
                                           token_limit=inj_limit, fact_tokens=fact_tokens)
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
            "active_tokens": active_tokens,
        }

    @property
    def active_memories(self) -> List:
        return self.memories

    def retrieve(self, query: str, current_turn: int = None, token_limit: int = 0,
                 fact_tokens=None) -> List[Dict]:
        return self.retriever.retrieve(query, self.memories, current_turn=current_turn,
                                       token_limit=token_limit, fact_tokens=fact_tokens)