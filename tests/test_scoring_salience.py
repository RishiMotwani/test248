"""Write-time salience tests (Phase 3, requirements SCOR-01..SCOR-03).

Validates that relevance used in importance scoring is computed from the
turn's user message against each fact (cosine when embeddings are available,
lexical overlap otherwise), rather than the historical hardcoded 0.8.
"""

import pytest

from memory_optimizer.pipeline import AdaptiveMemoryPipeline
from memory_optimizer.scoring import ImportanceScorer, _write_time_salience


def _settings():
    return {
        "max_context_tokens": 4096,
        "injection_token_limit": 0,
        "enable_compression": True,
        "decay_lambdas": {"transient": 0.3, "personal": 0.05,
                          "technical_preference": 0.05, "project_context": 0.02},
        "pruning": {"threshold": 0.20},
        "top_k": 5,
        "similarity_threshold": 0.35,
        "scoring_weights": {"w1_relevance": 0.40, "w2_utility": 0.30,
                            "w3_recency": 0.15, "w4_frequency": 0.15},
    }


class TestWriteTimeSalience:
    def test_overlap_salience_reflects_message(self):
        # Message shares tokens with the fact -> high overlap salience.
        high = _write_time_salience(
            "is the project feature flag configuration key_14 enabled",
            "Project feature flag configuration key_14 is enabled")
        low = _write_time_salience(
            "what is your favorite color palette",
            "Project feature flag configuration key_14 is enabled")
        assert high > low
        assert 0.0 <= high <= 1.0
        assert 0.0 <= low <= 1.0

    def test_embedding_salience_uses_cosine(self):
        calls = []

        def fake_embed(texts):
            # Stub embedder: 1-dim vector = token count; cosine of two texts
            # scales with how many query words appear in the fact.
            calls.append(list(texts))
            return [[float(len(t.split()))] for t in texts]

        high = _write_time_salience(
            "key_14 is enabled project flag configuration",
            "Project feature flag configuration key_14 is enabled",
            embed_fn=fake_embed)
        assert len(calls) == 1
        assert len(calls[0]) == 2  # user message + fact
        assert 0.0 <= high <= 1.0

    def test_empty_message_or_fact_is_zero(self):
        assert _write_time_salience("", "fact") == 0.0
        assert _write_time_salience("msg", "") == 0.0


class TestPipelineIngestSalience:
    def test_no_override_computes_write_time_salience(self):
        pipe = AdaptiveMemoryPipeline.from_settings(_settings())
        # A fact that is highly relevant to the message should score higher
        # than an unrelated fact, purely from the user message (no override).
        result = pipe.ingest(1, "project feature flag configuration key_14 enabled", [
            {"fact": "Project feature flag configuration key_14 is enabled",
             "category": "project_context", "confidence": 0.5},
        ])
        relevant = pipe.active_memories[0]["base_score"]

        pipe2 = AdaptiveMemoryPipeline.from_settings(_settings())
        pipe2.ingest(1, "tell me about your favorite color palette", [
            {"fact": "Project feature flag configuration key_14 is enabled",
             "category": "project_context", "confidence": 0.5},
        ])
        unrelated = pipe2.active_memories[0]["base_score"]
        assert relevant > unrelated

    def test_explicit_override_still_respected(self):
        pipe = AdaptiveMemoryPipeline.from_settings(_settings())
        pipe.ingest(5, "unrelated message", [
            {"fact": "Production must run PostgreSQL",
             "category": "technical_preference", "confidence": 0.5},
        ], query_relevance=0.99)
        # Explicit relevance wins even on an unrelated message.
        assert pipe.active_memories[0]["base_score"] >= 0.99 * 0.40


class TestScorer:
    def test_compute_score_uses_given_relevance(self):
        scorer = ImportanceScorer()
        mem = {"confidence": 0.5, "access_count": 1, "source_turn_id": 1}
        high = scorer.compute_score(mem, query_relevance=1.0, current_turn=1)
        low = scorer.compute_score(mem, query_relevance=0.0, current_turn=1)
        assert high > low
        assert 0.0 <= high <= 1.0