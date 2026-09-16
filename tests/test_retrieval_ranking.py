"""Retrieval ranking regression tests (Phase 5, requirement TEST-03).

Guards the ranking contract of MemoryRetriever: relevant memories rank above
irrelevant ones, normal retrieval reinforces (bumps) selected memories, and --
after the Phase 2 supersession fix -- a corrected fact outranks the stale
template sibling it superseded for queries about the corrected truth.
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


class TestTokenOverlapContract:
    def test_overlap_is_bounded(self):
        assert 0.0 <= _token_overlap(TOPIC, OTHER) <= 1.0
        assert _token_overlap(TOPIC, TOPIC) >= 0.9