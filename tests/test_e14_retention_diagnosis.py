"""Phase 10 retention-diagnosis tests.

Guards the E14 instrumentation and lifecycle classification without changing any
memory policy. Also asserts production defaults are untouched.
"""

from experiments.e14_retention_diagnosis import (
    _classify,
    _instrumented_replay,
    _lifecycle_counts,
)
from experiments.paper import load_settings
from memory_optimizer.retrieval import MemoryRetriever

FACT = "Constraint: the widget must use at most 64 megabytes of memory"
FACT_CATEGORY = "transient"


def _stream_with_fact():
    stream = [{
        "turn_id": 1,
        "user": f"Please note: {FACT}.",
        "tokens": 12,
        "facts": [{
            "fact": FACT,
            "category": FACT_CATEGORY,
            "confidence": 0.9,
            "source_turn_id": 1,
        }],
    }]
    for t in range(2, 260):
        stream.append({
            "turn_id": t,
            "user": f"Turn {t}: standup notes, nothing blocking.",
            "tokens": 8,
            "facts": [],
        })
    return stream


def _gt_and_queries():
    gt = [{
        "source_turn": 1,
        "fact": FACT,
        "category": FACT_CATEGORY,
        "is_trap": False,
        "qtype": "constraint",
        "expected_needed_later": True,
    }]
    queries = [{
        "qid": "q1",
        "user": "What is the memory constraint for the widget?",
        "required": ["64"],
        "forbidden": [],
        "qtype": "constraint",
        "source_turn": 1,
        "query_turn": 261,
        "long_range": True,
    }]
    return gt, queries


def _replay(decay_lambdas, pruning_threshold, store_budget=0, budget=8):
    settings = load_settings({"max_context_tokens": budget})
    stream = _stream_with_fact()
    gt, queries = _gt_and_queries()
    return _instrumented_replay(
        stream, gt, queries, settings,
        embed_fn=None, embedding_model="",
        decay_lambdas=decay_lambdas,
        pruning_threshold=pruning_threshold,
        budget=budget, store_budget=store_budget,
    )


class TestLifecycleTracking:
    def test_decay_removal_detected(self):
        rep = _replay({"transient": 0.15, "technical_preference": 0.02, "project_context": 0.01},
                      pruning_threshold=0.2)
        assert rep["lifecycle"][FACT] == "REMOVED_BY_DECAY"
        rec = rep["registry"][FACT]
        assert rec["pruned_by_decay"] is True
        assert rec["prune_reason"]

    def test_no_decay_toggle_prevents_decay_loss(self):
        rep = _replay({"transient": 0.0, "technical_preference": 0.0, "project_context": 0.0},
                      pruning_threshold=0.0)
        assert rep["lifecycle"][FACT] != "REMOVED_BY_DECAY"
        assert rep["registry"][FACT]["pruned_by_decay"] is False

    def test_no_prune_threshold_applies_score_without_deletion(self):
        rep = _replay({"transient": 0.15, "technical_preference": 0.02, "project_context": 0.01},
                      pruning_threshold=0.0)
        assert rep["lifecycle"][FACT] != "REMOVED_BY_DECAY"
        rec = rep["registry"][FACT]
        assert rec["pruned_by_decay"] is False
        # Score still decays even though nothing is deleted.
        assert rec["final_importance"] is not None

    def test_lifecycle_counts_sum_to_registry(self):
        rep = _replay({"transient": 0.15, "technical_preference": 0.02, "project_context": 0.01},
                      pruning_threshold=0.2)
        counts = _lifecycle_counts(rep["lifecycle"])
        assert sum(counts.values()) == len(rep["lifecycle"])


class TestStorePressureInstrumentation:
    def test_store_budget_eviction_detected(self):
        # A tiny store budget forces token_budget_evict on the first ingest.
        few = [{
            "turn_id": 1,
            "user": "Please note: several constraints.",
            "tokens": 10,
            "facts": [
                {"fact": "Constraint: alpha service uses at most 10 megabytes of memory",
                 "category": "technical_preference", "confidence": 0.5, "source_turn_id": 1},
                {"fact": "Constraint: beta service uses at most 20 megabytes of memory",
                 "category": "technical_preference", "confidence": 0.5, "source_turn_id": 1},
                {"fact": "Constraint: gamma service uses at most 30 megabytes of memory",
                 "category": "technical_preference", "confidence": 0.5, "source_turn_id": 1},
            ],
        }]
        settings = load_settings({"max_context_tokens": 512})
        gt, queries = [], []
        rep = _instrumented_replay(
            few, gt, queries, settings,
            embed_fn=None, embedding_model="",
            decay_lambdas={"technical_preference": 0.0, "project_context": 0.0},
            pruning_threshold=0.0,
            budget=512, store_budget=12,
        )
        states = _lifecycle_counts(rep["lifecycle"])
        assert states["REMOVED_BY_STORE_BUDGET"] >= 1
        pressure = rep["store_pressure_log"][0]
        assert pressure["store_over_budget_before_eviction"] > 0
        assert pressure["evicted_count"] >= 1
        assert pressure["post_store_tokens"] <= pressure["store_budget"]


class TestClassifier:
    def _rec(self, **kw):
        base = {
            "fact": "X", "source_turn": 1, "superseded_original": False,
            "is_correction_target": False, "appearances": 1,
            "pruned_by_decay": False, "removed_by_store_budget": False,
        }
        base.update(kw)
        return base

    def test_decay_state(self):
        assert _classify(self._rec(pruned_by_decay=True), set(), set(), {}, {}, []) == \
            "REMOVED_BY_DECAY"

    def test_budget_state(self):
        assert _classify(self._rec(removed_by_store_budget=True), set(), set(), {}, {}, []) == \
            "REMOVED_BY_STORE_BUDGET"

    def test_merged_state(self):
        assert _classify(self._rec(appearances=0), set(), set(), {}, {}, []) == \
            "MERGED_BY_DEDUPE"

    def test_superseded_correctly(self):
        rec = self._rec(superseded_original=True, fact="OLD")
        assert _classify(rec, set(), {"OLD"}, {}, {}, []) == "SUPERSEDED_CORRECTLY"

    def test_superseded_incorrectly_when_stale_present(self):
        rec = self._rec(superseded_original=True, fact="OLD")
        assert _classify(rec, {"OLD"}, {"OLD"}, {}, {}, []) == "SUPERSEDED_INCORRECTLY"

    def test_present_but_not_retrieved(self):
        rec = self._rec(fact="F", source_turn=1)
        q = {"qid": "q1", "required": ["missing"], "forbidden": [], "source_turn": 1}
        state = _classify(rec, {"F"}, set(), {1: q}, {"q1": []}, [])
        assert state == "PRESENT_BUT_NOT_RETRIEVED"

    def test_present_and_retrieved(self):
        rec = self._rec(fact="F", source_turn=1)
        q = {"qid": "q1", "required": ["alpha"], "forbidden": [], "source_turn": 1}
        state = _classify(rec, {"F"}, set(), {1: q}, {"q1": [{"fact": "has alpha"}]}, [])
        assert state == "PRESENT_AND_RETRIEVED"


class TestProductionDefaultsUnchanged:
    def test_retrieval_weights_default(self):
        r = MemoryRetriever()
        assert (r.imp_weight, r.sim_weight, r.cat_bonus) == (0.15, 0.85, 0.02)
        assert r.top_k == 5
        assert r.sim_threshold == 0.35

    def test_config_defaults(self):
        s = load_settings()
        assert s["pruning"]["threshold"] == 0.2
        assert s["decay_lambdas"] == {
            "transient": 0.15, "personal": 0.005,
            "technical_preference": 0.02, "project_context": 0.01,
        }
        assert s["scoring_weights"] == {
            "w1_relevance": 0.4, "w2_utility": 0.3,
            "w3_recency": 0.15, "w4_frequency": 0.15,
        }
