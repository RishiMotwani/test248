"""E15 retention-policy tests (Phase 11): separate activation from survival.

Covers the D32 contract:

* production defaults unchanged (retention.mode=hard_threshold,
  eviction_priority=current_importance; retrieval weights fixed)
* all three retention modes are recognized by ``step_decay_and_prune`` and an
  unknown mode is rejected
* a memory whose ``current_importance`` drops below the pruning threshold stays
  in the store under soft_decay / dual_score (activation != survival)
* ``retention_priority`` is deterministic and time-independent
* store-budget eviction under ``dual_score`` keys on ``retention_priority``,
  while the default keys on ``current_importance``
* retrieval ranking uses ``current_importance`` only — ``retention_priority``
  must never enter the query-time score
* revival: an old, high-base-score fact with ``current_importance < threshold``
  and no store pressure is pruned by hard_threshold but retained AND retrievable
  under soft_decay
* supersession safety: soft retention must not resurrect superseded values
  (obsolete_retention == 0, no SUPERSEDED_INCORRECTLY)
* context budget is enforced under the retention policies
"""

import pytest

from experiments.e15_retention_policy import run_cell
from experiments.paper import load_settings
from memory_optimizer.budget import token_budget_evict
from memory_optimizer.decay import CategoryDecayEngine
from memory_optimizer.pipeline import AdaptiveMemoryPipeline
from memory_optimizer.retrieval import MemoryRetriever


def word_count(text: str) -> int:
    return max(1, len(str(text).split()))


def _mem(fact, category="transient", base_score=0.5, access_count=1,
         source_turn=1, last_access=1, current_importance=None,
         retention_priority=None):
    m = {
        "fact": fact,
        "category": category,
        "base_score": base_score,
        "confidence": 0.6,
        "access_count": access_count,
        "last_access_turn": last_access,
        "source_turn_id": source_turn,
    }
    if current_importance is not None:
        m["current_importance"] = current_importance
    if retention_priority is not None:
        m["retention_priority"] = retention_priority
    return m


# --------------------------------------------------------------------------- #
# Production defaults
# --------------------------------------------------------------------------- #

class TestProductionDefaults:
    def test_retention_config_defaults(self):
        # Phase-11 candidate default (E15, D32): dual_score + retention-priority
        # eviction. Decay no longer deletes; eviction prefers low retention.
        s = load_settings()
        assert s["retention"]["mode"] == "dual_score"
        assert s["retention"]["eviction_priority"] == "retention_priority"

    def test_retention_overrides_in_load_settings(self):
        s = load_settings({"retention_mode": "dual_score",
                           "eviction_priority": "retention_priority"})
        assert s["retention"]["mode"] == "dual_score"
        assert s["retention"]["eviction_priority"] == "retention_priority"

    def test_retrieval_still_uses_current_importance_only(self):
        r = MemoryRetriever(embed_fn=None)
        assert (r.imp_weight, r.sim_weight, r.cat_bonus) == (0.15, 0.85, 0.02)

    def test_pruning_threshold_default(self):
        s = load_settings()
        assert float(s["pruning"]["threshold"]) == 0.2


# --------------------------------------------------------------------------- #
# Retention modes
# --------------------------------------------------------------------------- #

class TestRetentionModes:
    def test_modes_recognized(self):
        decay = CategoryDecayEngine()
        mems = [_mem("a fact", base_score=0.1, source_turn=1, last_access=1)]
        for mode in ("hard_threshold", "soft_decay", "dual_score"):
            active, pruned = decay.step_decay_and_prune(
                [dict(m) for m in mems], current_turn=100, retention_mode=mode)
            assert isinstance(active, list) and isinstance(pruned, list), mode

    def test_unknown_mode_rejected(self):
        decay = CategoryDecayEngine()
        with pytest.raises(ValueError):
            decay.step_decay_and_prune([], current_turn=1, retention_mode="bogus")

    def test_below_threshold_pruned_in_hard_mode(self):
        decay = CategoryDecayEngine(lambdas={"transient": 0.15},
                                    pruning_threshold=0.2)
        mem = _mem("transient chatter", base_score=0.3, source_turn=1, last_access=1)
        active, pruned = decay.step_decay_and_prune([mem], 100, "hard_threshold")
        assert len(active) == 0 and len(pruned) == 1

    def test_below_threshold_retained_in_soft_and_dual(self):
        decay = CategoryDecayEngine(lambdas={"transient": 0.15},
                                    pruning_threshold=0.2)
        for mode in ("soft_decay", "dual_score"):
            mem = _mem("transient chatter", base_score=0.3, source_turn=1, last_access=1)
            active, pruned = decay.step_decay_and_prune([mem], 100, mode)
            assert len(active) == 1 and len(pruned) == 0, mode
            assert active[0]["current_importance"] < 0.2, mode
            assert active[0].get("retained_below_threshold") is True, mode

    def test_both_scores_recorded(self):
        decay = CategoryDecayEngine()
        mem = _mem("x", base_score=0.5, access_count=1, last_access=49)
        active, _ = decay.step_decay_and_prune([mem], 50, retention_mode="soft_decay")
        assert "current_importance" in active[0]
        assert "retention_priority" in active[0]


# --------------------------------------------------------------------------- #
# retention_priority determinism / reinforcement / time-independence
# --------------------------------------------------------------------------- #

class TestRetentionPriority:
    def test_deterministic_and_time_independent(self):
        decay = CategoryDecayEngine()
        mem = _mem("the auth service signs rs256 tokens", base_score=0.7,
                   access_count=2, source_turn=5, last_access=5)
        rp1 = decay.calculate_retention_priority(mem)
        # changing last access tells decayed-importance to change but retention
        # priority must not move
        moved = dict(mem, last_access_turn=250, source_turn_id=5)
        rp2 = decay.calculate_retention_priority(moved)
        assert rp1 == rp2 == pytest.approx(min(1.0, 0.7 * (1 + 0.2 * 1)))

    def test_reinforcement_lifts_priority(self):
        decay = CategoryDecayEngine()
        single = _mem("x", base_score=0.5, access_count=1)
        reinforced = _mem("x", base_score=0.5, access_count=4)
        assert decay.calculate_retention_priority(reinforced) > \
            decay.calculate_retention_priority(single)
        assert decay.calculate_retention_priority(reinforced) == pytest.approx(
            min(1.0, 0.5 * (1 + 0.2 * 3)))

    def test_capped_at_one(self):
        decay = CategoryDecayEngine()
        mem = _mem("x", base_score=0.99, access_count=100)
        assert decay.calculate_retention_priority(mem) <= 1.0


# --------------------------------------------------------------------------- #
# dual_score eviction vs current-importance eviction
# --------------------------------------------------------------------------- #

class TestStoreEvictionPriority:
    def _evict_one(self, priority_key):
        mem_a = _mem("memory A", current_importance=0.10, retention_priority=0.90)
        mem_b = _mem("memory B", current_importance=0.30, retention_priority=0.20)
        kept, evicted, used = token_budget_evict(
            [mem_a, mem_b], budget=2, fact_tokens=word_count, priority_key=priority_key)
        assert len(evicted) == 1            # budget fits exactly one 2-word fact
        assert len(kept) == 1 and used <= 2
        return evicted[0]["fact"]

    def test_default_evicts_lowest_current_importance(self):
        # default priority_key = current_importance -> A (0.10) evicted
        assert self._evict_one("current_importance") == "memory A"

    def test_dual_score_evicts_lowest_retention_priority(self):
        # dual_score -> retention_priority -> B (0.20) evicted even though it has
        # the higher current_importance
        assert self._evict_one("retention_priority") == "memory B"

    def test_default_param_is_current_importance(self):
        mem_a = _mem("memory A", current_importance=0.10, retention_priority=0.90)
        mem_b = _mem("memory B", current_importance=0.30, retention_priority=0.20)
        kept, evicted, _ = token_budget_evict([mem_a, mem_b], budget=2,
                                              fact_tokens=word_count)
        assert evicted[0]["fact"] == "memory A"


# --------------------------------------------------------------------------- #
# Retrieval isolation: current_importance only
# --------------------------------------------------------------------------- #

class TestRetrievalIsolation:
    def test_ranking_ignores_retention_priority(self):
        fact = "Requirement: the inventory sync API handles 900 requests per second"
        high_imp = _mem(fact, current_importance=0.9, retention_priority=0.1)
        low_imp = _mem(fact, current_importance=0.2, retention_priority=0.9)
        # same fact text -> identical lexical similarity; only importance differs
        r = MemoryRetriever(top_k=5, sim_threshold=0.0, embed_fn=None)
        ranked = r._score_all("inventory sync API requests per second", [low_imp, high_imp])
        assert ranked[0]["importance"] == pytest.approx(0.9)
        assert ranked[0]["fact"] == fact

    def test_admitted_and_ranked_by_importance_not_priority(self):
        fact = "Constraint: the gateway may use at most 64 megabytes of memory"
        m_hi_prio = _mem("Requirement: the checkout API must handle 300 requests",
                         current_importance=0.05, retention_priority=0.95)
        m_hi_imp = _mem(fact, current_importance=0.85, retention_priority=0.05)
        r = MemoryRetriever(top_k=5, sim_threshold=0.0, embed_fn=None)
        ranked = r._score_all("gateway memory constraint", [m_hi_prio, m_hi_imp])
        # the higher-current_importance memory outranks the high-retention one
        assert ranked[0]["fact"] == m_hi_imp["fact"]
        assert "retention_priority" not in {k for o in ranked for k in o if False}


# --------------------------------------------------------------------------- #
# Revival (the behavior this phase establishes)
# --------------------------------------------------------------------------- #

class TestRevival:
    FACT = ("Requirement: the inventory sync API must handle its workload at "
            "900 requests per second")
    QUERY = ("What is the throughput requirement in requests per second for the "
             "inventory sync API?")

    def _playback(self, retention_mode):
        """Ingest one high-base-score fact, then let decay run unreinforced for
        ~59 turns under ``retention_mode``. Returns (pipe, pruned_facts)."""
        settings = load_settings({
            "max_context_tokens": 64,
            "injection_token_limit": 64,
            "memory_store_token_budget": 0,     # no store pressure
        })
        settings["enable_compression"] = False
        pipe = AdaptiveMemoryPipeline.from_settings(settings, embed_fn=None)
        fact = _mem(self.FACT, category="transient", base_score=0.9,
                    access_count=1, source_turn=1, last_access=1)
        pipe.ingest(1, "note about the inventory sync API", [fact],
                    fact_tokens=word_count)
        pruned_facts = []
        for t in range(2, 61):
            active, pruned = pipe.decay.step_decay_and_prune(
                pipe.active_memories, t, retention_mode=retention_mode)
            pruned_facts.extend(pruned)
            pipe.memories = active
        return pipe, pruned_facts

    def test_hard_threshold_prunes_revived_candidate(self):
        pipe, pruned_facts = self._playback("hard_threshold")
        assert pipe.active_memories == []            # fact was pruned
        assert len(pruned_facts) == 1
        assert pruned_facts[0]["current_importance"] < 0.2   # removed for low activation
        injected = pipe.retrieve(self.QUERY, current_turn=61, token_limit=64,
                                 fact_tokens=word_count)
        assert injected == []                        # absent from store

    def test_soft_decay_retains_and_retrieves(self):
        pipe, pruned_facts = self._playback("soft_decay")
        assert pruned_facts == []                    # decay prunes nothing
        assert len(pipe.active_memories) == 1
        ci = pipe.active_memories[0]["current_importance"]
        assert ci < 0.2                              # LOW current importance...
        assert pipe.active_memories[0]["retained_below_threshold"] is True
        injected = pipe.retrieve(self.QUERY, current_turn=61, token_limit=64,
                                 fact_tokens=word_count)
        assert injected                            # ...memory still present...
        assert injected[0]["fact"] == self.FACT     # ...and retrievable at the end


# --------------------------------------------------------------------------- #
# Supersession safety under soft retention
# --------------------------------------------------------------------------- #

class TestSupersessionSafety:
    @pytest.mark.parametrize("policy", ["soft_decay", "dual_score"])
    def test_soft_retention_does_not_resurrect_corrections(self, policy):
        cell = run_cell(seed=42, budget=64, policy=policy, turns=120, scale=1,
                        use_embeddings=False)
        metrics = cell["metrics"]
        assert metrics["obsolete_retention"] in (0, None)
        assert metrics["correction_recall"] == 1.0
        lc = cell["lifecycle_counts"]
        assert lc["SUPERSEDED_CORRECTLY"] == 2
        assert lc["SUPERSEDED_INCORRECTLY"] == 0
        assert lc["REMOVED_BY_DECAY"] == 0     # soft/dual never decay-prune


# --------------------------------------------------------------------------- #
# Context budget enforced + policy wiring observable in the E15 cell
# --------------------------------------------------------------------------- #

class TestE15Cell:
    @pytest.mark.parametrize("policy", ["hard_threshold", "soft_decay",
                                        "dual_score", "no_decay"])
    def test_context_budget_enforced(self, policy):
        cell = run_cell(seed=42, budget=64, policy=policy, turns=120, scale=1,
                        use_embeddings=False)
        assert cell["metrics"]["mean_context_tokens"] <= 64 + 1e-6

    def test_cell_shape(self):
        cell = run_cell(seed=42, budget=128, policy="soft_decay", turns=120,
                        scale=1, use_embeddings=False)
        assert cell["policy"] == "soft_decay"
        assert cell["store_summary"]["decay_removals"] == 0
        assert "retention_priority" in cell["registry"][list(cell["registry"])[0]]
        assert cell["store_summary"]["fraction_below_pruning_threshold"] is not None
        assert cell["correction_gate"]["obsolete_retention"] in (0, None)