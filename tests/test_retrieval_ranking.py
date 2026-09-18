"""Retrieval ranking regression tests (Phase 5, requirement TEST-03).

Guards the ranking contract of MemoryRetriever: relevant memories rank above
irrelevant ones, normal retrieval reinforces (bumps) selected memories, and --
after the Phase 2 supersession fix -- a corrected fact outranks the stale
template sibling it superseded for queries about the corrected truth.

Also guards the new query-first retrieval semantics (Change 2):
- query similarity dominates historical importance
- explicit token_limit is respected
"""

import pytest

from memory_optimizer.retrieval import MemoryRetriever, _token_overlap
from memory_optimizer.compression import MemoryCompressor
from memory_optimizer.pipeline import AdaptiveMemoryPipeline

TOPIC = "project feature flag configuration key_31 is enabled"
OTHER = "our team prefers heading over heading based management"
CORR = "Revised requirement: project feature flag configuration key_31 is enabled is no longer the case"


@pytest.fixture
def retriever():
    return MemoryRetriever(top_k=5, sim_threshold=0.35)


def _mem(fact, source_turn_id=1, category="project_context", importance=0.5, **extra):
    return {"fact": fact, "category": category, "source_turn_id": source_turn_id,
            "base_score": importance, "current_importance": importance,
            "access_count": 1, "last_access_turn": source_turn_id, **extra}


class TestRanking:
    def test_relevant_ranks_above_irrelevant(self, retriever):
        store = [_mem(OTHER, 1, "personal", importance=0.9),
                 _mem(TOPIC, 1, "project_context", importance=0.45)]
        top = retriever.retrieve("is the project feature flag configuration key_31 enabled",
                                 store, current_turn=5)
        assert top and TOPIC in top[0]["fact"]

    def test_ranking_with_corrected_fact(self, retriever):
        # Correction supersedes the original; a query about current truth must
        # surface the *corrected* fact ahead of the stale original.
        store = [_mem(TOPIC, 28, "project_context", importance=0.5),
                 _mem(CORR, 28, "project_context", importance=0.5,
                      superseded_prior_fact=TOPIC)]
        top = retriever.retrieve("is the project feature flag configuration key_31 enabled",
                                 store, current_turn=50)
        assert top and CORR in top[0]["fact"]

    def test_retrieval_reinforces_selected(self, retriever):
        store = [_mem(TOPIC, 1, importance=0.5)]
        before = store[0]["access_count"]
        retriever.retrieve("project feature flag configuration key_31", store, current_turn=9)
        assert store[0]["access_count"] == before + 1
        assert store[0]["last_access_turn"] == 9

    def test_ranked_view_does_not_reinforce(self, retriever):
        store = [_mem(TOPIC, 1, importance=0.5)]
        before = store[0]["access_count"]
        retriever.retrieve("project feature flag configuration key_31", store, current_turn=9,
                           ranked=True)
        assert store[0]["access_count"] == before

    def test_query_relevance_beats_historical_importance(self):
        """A memory strongly relevant to the current query must outrank an
        unrelated memory merely because the unrelated memory has higher
        historical importance.

        Uses a lower sim_threshold to ensure both candidates pass the admission
        rule; this isolates the ranking formula from the admission filter.
        """
        query = "Which service uses the RS256 signing algorithm?"

        highly_relevant = _mem(
            "Architecture: auth service signs tokens using RS256",
            source_turn_id=10, category="project_context", importance=0.2)

        highly_important_but_irrelevant = _mem(
            "Project context: billing pipeline uses Kafka for invoice events",
            source_turn_id=5, category="project_context", importance=0.95,
            access_count=20)

        # Use lower sim_threshold so both candidates pass admission (lexical sim max ~0.29)
        test_retriever = MemoryRetriever(top_k=5, sim_threshold=0.2, imp_weight=0.15, sim_weight=0.85, cat_bonus=0.02)
        ranked = test_retriever.retrieve(query,
                                    [highly_important_but_irrelevant, highly_relevant],
                                    current_turn=100, ranked=True)

        assert ranked[0]["fact"] == highly_relevant["fact"], (
            f"Expected relevant fact to rank first, got: {ranked[0]['fact']}")

    def test_compressed_correction_is_the_only_authoritative_memory(self, retriever):
        """End-to-end (Phase 14): consolidation must replace the stale fact with
        the correction even though the correction's semantics differ from the
        original, so retrieval sees exactly one authoritative memory."""
        original = (
            "User IDs are stored as integers in the users table, so callers "
            "cast them on the way in."
        )
        corrected = (
            "Correction to an earlier note: user IDs are no longer stored as "
            "integers in the users table, so callers do not cast them on the way "
            "in. User IDs are opaque strings; preserve the client-provided "
            "string exactly."
        )

        stored = [
            _mem(original, source_turn_id=80, importance=0.5),
            _mem(corrected, source_turn_id=260, importance=0.5),
        ]

        compressed = MemoryCompressor().dedupe_incremental(stored[:1], stored[1:])

        assert len(compressed) == 1
        assert compressed[0]["fact"] == corrected

        ranked = retriever.retrieve(
            "How are user IDs stored?",
            compressed,
            current_turn=300,
        )
        assert ranked and "User IDs are opaque strings" in ranked[0]["fact"]

    def test_superseded_memory_retrieves_using_current_embedding(self, retriever):
        """End-to-end (Phase 15, D36): after supersession the corrected memory
        must carry the *corrected* embedding so retrieval ranks it using the
        current semantics. A stale embedding (which the pre-fix guard preserved
        when the stored fact already had one) would describe the obsolete
        predecessor and score ~0 against a query embedded as the corrected
        truth."""
        old = (
            "User IDs are stored as integers in the users table, so callers "
            "cast them on the way in."
        )
        corrected = (
            "Correction to an earlier note: user IDs are no longer stored as "
            "integers in the users table, so callers do not cast them on the way "
            "in. User IDs are opaque strings; preserve the client-provided "
            "string exactly."
        )
        old_embedding = [1.0, 0.0, 0.0]
        corrected_embedding = [0.0, 1.0, 0.0]

        def fake_embed(texts):
            return [old_embedding if t == old else corrected_embedding
                    for t in texts]

        stored = [dict(_mem(old, source_turn_id=80, importance=0.5),
                       fact_embedding=old_embedding)]
        compressed = MemoryCompressor().dedupe_incremental(
            stored,
            [_mem(corrected, source_turn_id=260, importance=0.5)],
            embed_fn=fake_embed,
        )

        assert len(compressed) == 1
        assert compressed[0]["fact"] == corrected
        assert compressed[0]["fact_embedding"] == corrected_embedding

        # Query embedded as the corrected truth: cosine(query, stored) must be
        # ~1.0 using the corrected embedding, and the corrected fact must win
        # via the embedding engine (not lexical drift toward the old text).
        embedded_retriever = MemoryRetriever(
            top_k=5, sim_threshold=0.35, embed_fn=fake_embed)
        ranked = embedded_retriever.retrieve(
            "How are user IDs stored?", compressed, current_turn=300, ranked=True)
        assert ranked, "corrected memory must be retrieved at all"
        assert ranked[0]["fact"] == corrected
        assert ranked[0]["retrieval_engine"] == "embedding"
        assert ranked[0]["retrieval_sim"] > 0.99
        assert round(ranked[0]["retrieval_sim"], 3) == 1.0

    def test_token_limit_is_respected(self, retriever):
        """retrieve(..., token_limit=X) must never exceed X tokens and should
        select as many facts as fit (not just top_k)."""
        # Three small facts, each ~10 tokens word-count
        f1 = _mem("Fact one about alpha service", 1, "project_context", importance=0.9)
        f2 = _mem("Fact two about beta service", 2, "project_context", importance=0.8)
        f3 = _mem("Fact three about gamma service", 3, "project_context", importance=0.7)

        # token_limit allows 2 facts (~20 tokens), top_k=5
        retrieved = retriever.retrieve("alpha", [f1, f2, f3],
                                       token_limit=25, fact_tokens=lambda t: len(t.split()),
                                       current_turn=10)

        # Should fit 2 facts within 25 tokens, not just 1 (top_k=5 would allow 3)
        assert len(retrieved) >= 2
        total_tokens = sum(len(f["fact"].split()) for f in retrieved)
        assert total_tokens <= 25, f"Exceeded token limit: {total_tokens}"


class TestTokenOverlapContract:
    def test_overlap_is_bounded(self):
        assert 0.0 <= _token_overlap(TOPIC, OTHER) <= 1.0
        assert _token_overlap(TOPIC, TOPIC) >= 0.9


class TestActiveContextBudgetEnforcement:
    """Strict architectural regression tests for active-context budget."""

    def test_token_limit_exceeds_top_k(self):
        """retrieve(..., token_limit=budget) can return more than top_k
        when the budget permits, and never exceeds the token limit."""
        retriever = MemoryRetriever(top_k=2, sim_threshold=0.1, imp_weight=0.15, sim_weight=0.85)
        # Four small facts, each ~10 tokens
        facts = [
            _mem("Fact one about alpha service", 1, "project_context", importance=0.9),
            _mem("Fact two about beta service", 2, "project_context", importance=0.8),
            _mem("Fact three about gamma service", 3, "project_context", importance=0.7),
            _mem("Fact four about delta service", 4, "project_context", importance=0.6),
        ]

        # token_limit=35 allows 3 facts (~30 tokens), top_k=2 would only allow 2
        retrieved = retriever.retrieve("alpha", facts,
                                       token_limit=35, fact_tokens=lambda t: len(t.split()),
                                       current_turn=10)

        # Should fit 3 facts within 35 tokens (not limited to top_k=2)
        assert len(retrieved) >= 3, f"Expected >= 3 facts, got {len(retrieved)}"
        total_tokens = sum(len(f["fact"].split()) for f in retrieved)
        assert total_tokens <= 35, f"Exceeded token limit: {total_tokens}"

    def test_token_limit_never_exceeded(self):
        """retrieve(..., token_limit) must never exceed the token limit."""
        retriever = MemoryRetriever(top_k=10, sim_threshold=0.1)
        facts = [_mem(f"Fact {i} about service", i, "project_context", importance=0.5) for i in range(20)]

        for limit in [10, 25, 50, 100]:
            retrieved = retriever.retrieve("service", facts,
                                           token_limit=limit, fact_tokens=lambda t: len(t.split()),
                                           current_turn=10)
            total_tokens = sum(len(f["fact"].split()) for f in retrieved)
            assert total_tokens <= limit, f"Limit {limit} exceeded: {total_tokens}"


class TestStorePressureIndependence:
    """Strict architectural regression test: store capacity independent of active context."""

    def test_store_budget_independent_of_active_context(self):
        """Active-context budget=32, store budget=256: store should retain facts
        exceeding active budget but bounded by store budget."""
        settings = {
            "max_context_tokens": 32,
            "memory_store_token_budget": 256,
            "injection_token_limit": 32,
            "top_k": 5,
            "similarity_threshold": 0.35,
            "scoring_weights": {
                "w1_relevance": 0.4,
                "w2_utility": 0.3,
                "w3_recency": 0.15,
                "w4_frequency": 0.15,
            },
            "decay_lambdas": {
                "transient": 0.0,
                "personal": 0.0,
                "technical_preference": 0.0,
                "project_context": 0.0,
            },
            "pruning": {"threshold": 0.0},
            "compression": {"enabled": False},
        }

        # Construct facts: each ~15 tokens. We need enough to exceed 32 (active)
        # but remain below 256 (store). 10 facts * 15 = 150 tokens.
        facts = []
        for i in range(10):
            facts.append({
                "fact": f"Requirement: the service_{i} API must handle {100 + i * 10} requests per second",
                "category": "technical_preference",
                "confidence": 0.5,
            })

        pipe = AdaptiveMemoryPipeline.from_settings(settings)
        # Disable decay and compression for this pure budget test
        pipe.decay.lambdas = {k: 0.0 for k in pipe.decay.lambdas}
        pipe.settings["enable_compression"] = False

        # Ingest all facts in one turn
        result = pipe.ingest(1, "test message", facts, fact_tokens=lambda t: len(t.split()))

        # Active context injected should be bounded by 32 tokens
        assert result["injected_tokens"] <= 32

        # Store should have retained more than active context (bounded by 256)
        store_tokens = sum(len(m["fact"].split()) for m in pipe.active_memories)
        assert store_tokens > 32, f"Store tokens {store_tokens} should exceed active budget 32"
        assert store_tokens <= 256, f"Store tokens {store_tokens} should not exceed store budget 256"

        # Now test that store budget is enforced: add more facts exceeding 256
        more_facts = []
        for i in range(10, 25):  # 15 more * ~15 = 225 tokens -> total ~375 > 256
            more_facts.append({
                "fact": f"Constraint: the service_{i} service may use at most {64 + i * 5} megabytes of memory",
                "category": "technical_preference",
                "confidence": 0.5,
            })

        result = pipe.ingest(2, "more facts", more_facts, fact_tokens=lambda t: len(t.split()))

        # Store should be capped at 256
        store_tokens = sum(len(m["fact"].split()) for m in pipe.active_memories)
        assert store_tokens <= 256, f"Store tokens {store_tokens} should not exceed store budget 256"

    def test_unbounded_store_allows_growth(self):
        """memory_store_token_budget=0 means no hard store cap; retention governed by decay/dedupe."""
        settings = {
            "max_context_tokens": 32,
            "memory_store_token_budget": 0,  # unbounded
            "injection_token_limit": 32,
            "top_k": 5,
            "similarity_threshold": 0.35,
            "scoring_weights": {
                "w1_relevance": 0.4,
                "w2_utility": 0.3,
                "w3_recency": 0.15,
                "w4_frequency": 0.15,
            },
            "decay_lambdas": {
                "transient": 0.0,
                "personal": 0.0,
                "technical_preference": 0.0,
                "project_context": 0.0,
            },
            "pruning": {"threshold": 0.0},
            "compression": {"enabled": False},
        }

        facts = []
        for i in range(10):
            facts.append({
                "fact": f"Requirement: the service_{i} API must handle {100 + i * 10} requests per second",
                "category": "technical_preference",
                "confidence": 0.5,
            })

        pipe = AdaptiveMemoryPipeline.from_settings(settings)
        pipe.decay.lambdas = {k: 0.0 for k in pipe.decay.lambdas}
        pipe.settings["enable_compression"] = False

        result = pipe.ingest(1, "test", facts, fact_tokens=lambda t: len(t.split()))

        # Active context bounded by 32
        assert result["injected_tokens"] <= 32

        # Store can exceed active budget (no hard cap)
        store_tokens = sum(len(m["fact"].split()) for m in pipe.active_memories)
        assert store_tokens > 32, f"Unbounded store should exceed active budget: {store_tokens}"