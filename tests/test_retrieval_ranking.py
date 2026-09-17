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