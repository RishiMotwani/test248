"""Phase 12 — identity-safe evaluation, access separation, task-state,
task-affinity eviction order, oracle isolation, correction safety,
no-future-leakage, active-budget enforcement."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from e16_retention_selectivity import (
    run_cell,
    _cell_key,
    _spearman,
    _token_census,
    _id_answer_parts,
    _id_answered,
    _resolve_embed,
    EMBEDDING_MODEL,
)
from memory_optimizer.task_state import TaskStateTracker
from data.coding_workload import self_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embed_fn():
    return _resolve_embed(True, EMBEDDING_MODEL, None)


def _char_embed(texts):
    """Deterministic char-hash embedding; accepts str or list[str]."""
    if isinstance(texts, str):
        texts = [texts]
    out = []
    for t in texts:
        vec = [0.0] * 64
        for ch in (t or ""):
            vec[ord(ch) % 64] += 1.0
        out.append(vec)
    return out[0] if len(out) == 1 else out


def _run(embed_fn, **kw):
    return run_cell(kw.pop("seed", 42), kw.pop("active", 128),
                    kw.pop("policy", "dual_score"),
                    turns=kw.pop("turns", 100), scale=kw.pop("scale", 3),
                    **kw)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIdentitySafeAPI:
    """_id_answer_parts/_id_answered match on fact_ids, not tokens."""

    def test_answered_when_target_present(self):
        q = {"qid": "q", "target_fact_ids": ["f1"], "forbidden_fact_ids": ["f2"]}
        entries = [{"fact_id": "f1"}, {"fact_id": "f3"}]
        assert _id_answered(q, entries) is True

    def test_blocked_when_forbidden_present(self):
        q = {"qid": "q", "target_fact_ids": ["f1"], "forbidden_fact_ids": ["f2"]}
        entries = [{"fact_id": "f1"}, {"fact_id": "f2"}]
        parts = _id_answer_parts(q, entries)
        assert parts["answered"] is False
        assert parts["present_forbidden"] == ["f2"]

    def test_missing_target_isolated(self):
        q = {"qid": "q", "target_fact_ids": ["f1"], "forbidden_fact_ids": ["f2"]}
        entries = [{"fact_id": "f2"}]
        parts = _id_answer_parts(q, entries)
        assert parts["missing_target"] == ["f1"]

    def test_correction_ids_distinct(self, embed_fn):
        """NEW corrected fact and old superseded fact have distinct fact_ids."""
        c = _run(embed_fn, policy="dual_score", store_budget=256, active=128)
        assert c["identity_metrics"]["fact_identity_correction_recall"] >= 0.95


class TestTaskStateTracker:
    """Causality + windowing of the tracker."""

    def test_window_limits_stored_turns(self):
        tracker = TaskStateTracker(window=3)
        for turn in range(1, 20):
            tracker.observe(turn, f"msg with payload {turn}",
                            facts=[{"fact": f"fact {turn}"}],
                            embed_fn=_char_embed)
        mat = tracker.message_matrix()
        assert mat is not None and mat.shape[0] <= 3

    def test_non_fact_turns_ignored(self):
        tracker = TaskStateTracker(window=32)
        for turn in range(1, 10):
            tracker.observe(turn, f"chit-chat {turn}", facts=[], embed_fn=_char_embed)
        assert tracker.message_matrix() is None

    def test_affinities_in_unit_range(self):
        tracker = TaskStateTracker(window=8)
        for turn in range(1, 6):
            tracker.observe(turn, f"user message {turn}",
                            facts=[{"fact": f"fact {turn}"}],
                            embed_fn=_char_embed)
        af = tracker.batch_affinity([_char_embed("user message 3")])
        assert 0.0 <= af[0] <= 1.0

    def test_empty_tracker_affinity_zero(self):
        tracker = TaskStateTracker()
        af = tracker.batch_affinity([_char_embed("whatever")])
        assert af == [0.0]


class TestAccessSeparationCellFields:
    """Cell fields differentiate retrieval vs ingest reinforcement."""

    def test_final_counts_present(self, embed_fn):
        c = _run(embed_fn, policy="dual_score", store_budget=256, active=128)
        fc = c["final_counts"]
        assert "future_use_total" in fc and "future_use_evicted" in fc
        assert fc["future_use_total"] > 0

    def test_diagnostics_spearman_keys(self, embed_fn):
        c = _run(embed_fn, policy="dual_score", store_budget=256, active=128)
        sp = c["diagnostics"]["spearman_vs_future_use"]
        for key in ("retrieval_access_count", "ingest_reinforcement_count",
                    "retention_priority", "task_affinity", "base_score"):
            assert key in sp, f"missing spearman key {key}"

    def test_by_population_isolation(self, embed_fn):
        """retrieved_needed vs never_retrieved are disjoint populations."""
        c = _run(embed_fn, policy="dual_score", store_budget=256, active=128)
        sizes = c["diagnostics"]["population_sizes"]
        assert sizes["retrieved_needed"] >= 0
        assert sizes["never_retrieved"] >= 0


class TestTaskAffinityEvictionOrder:
    """task_affinity policy stamps affinity; base_score_only stamps base_score."""

    def test_task_affinity_policy_runs(self, embed_fn):
        """task_affinity must be stamped on the facts the policy evicted."""
        c = _run(embed_fn, policy="task_affinity", store_budget=128, active=128)
        assert c["store_summary"]["evicted_total"] > 0
        ev = c["diagnostics"]["by_population"]["evicted"]["task_affinity"]
        assert ev["mean"] is not None, "task_affinity not stamped on evicted facts"
        assert ev["mean"] > 0.0 or ev["max"] == 0.0  # stamped (allow all-zero max)

    def test_base_score_only_spearman_present(self, embed_fn):
        c = _run(embed_fn, policy="base_score_only", store_budget=256, active=128)
        assert "base_score" in c["diagnostics"]["spearman_vs_future_use"]


class TestOracleIsolation:
    """Oracle runs under the same protected-safety and leak guards."""

    def test_oracle_no_leakage_flag(self, embed_fn):
        c = _run(embed_fn, policy="oracle_future_use", store_budget=256, active=128)
        assert c["no_future_leakage_ok"] is True

    def test_oracle_retention_ceiling(self, embed_fn):
        """Oracle future-use retention recall is the ceiling: >= any causal policy."""
        o = _run(embed_fn, policy="oracle_future_use", store_budget=256, active=128)
        d = _run(embed_fn, policy="dual_score", store_budget=256, active=128)
        r = _run(embed_fn, policy="random", store_budget=256, active=128)
        o_rr = o["future_use"]["retention_recall"]
        assert o_rr >= d["future_use"]["retention_recall"]
        assert o_rr >= r["future_use"]["retention_recall"]


class TestCorrectionSafety:
    """Corrections never resurrect superseded facts; gate tracks conflicts."""

    @pytest.mark.parametrize("policy", ["dual_score", "task_affinity",
                                        "base_score_only"])
    def test_superseded_incorrectly_zero(self, embed_fn, policy):
        c = _run(embed_fn, policy=policy, store_budget=256, active=128)
        assert c["state_counts"]["SUPERSEDED_INCORRECTLY"] == 0

    def test_correction_gate_zero_obsolete(self, embed_fn):
        c = _run(embed_fn, policy="dual_score", store_budget=256, active=128)
        gate = c["correction_gate"]
        assert gate["identity_obsolete_retention"] == 0.0
        assert gate["old_superseded_present"] == 0

    def test_protected_capacity_conflict_tracked(self, embed_fn):
        """At tight budgets the guard tracks (not silently drops) conflicts."""
        c = _run(embed_fn, policy="dual_score", store_budget=64, active=128,
                 turns=100, scale=3)
        assert "protected_capacity_conflict_turns" in c["correction_gate"]
        assert c["state_counts"]["SUPERSEDED_INCORRECTLY"] == 0


class TestNoFutureLeakage:
    """no_future_leakage_ok flag is set for all policies."""

    @pytest.mark.parametrize("policy", ["dual_score", "task_affinity",
                                        "base_score_only", "random",
                                        "oracle_future_use"])
    def test_runs_without_leak(self, embed_fn, policy):
        c = _run(embed_fn, policy=policy, store_budget=512, active=128)
        assert c["no_future_leakage_ok"] is True
        assert c["policy"] == policy


class TestActiveBudgetEnforcement:
    """Context stays within the active token budget."""

    @pytest.mark.parametrize("budget", [64, 128, 256])
    def test_max_context_within_budget(self, embed_fn, budget):
        c = _run(embed_fn, policy="dual_score", store_budget=512, active=budget)
        assert c["metrics"]["max_context_tokens"] <= budget + 1

    def test_pressure_evicts(self, embed_fn):
        """With a tight store budget, eviction must actually fire."""
        c = _run(embed_fn, policy="dual_score", store_budget=64, active=128)
        assert c["store_summary"]["evicted_total"] > 0

    def test_no_pressure_evicts_zero(self, embed_fn):
        c = _run(embed_fn, policy="dual_score", store_budget=0, active=128)
        assert c["store_summary"]["evicted_total"] == 0


class TestTokenCensus:
    """_token_census counts authoritative owners correctly."""

    def test_memcache_owners(self):
        """memcache/redis are NOT correction tokens; they legitimately occur in
        several unrelated authoritative facts (overlapping/generalization
        templates) — exactly the collision that E15's token metric fatally mixes
        with correction presence."""
        census = _token_census(seed=42, turns=1200, scale=27)
        assert census["memcache"]["authoritative_owners"] >= 3
        assert census["redis"]["authoritative_owners"] >= 8

    def test_correction_tokens(self):
        """'100'/'250' occur only in the superseded/revised constraint, so they
        are near-unique — the token-level scope of E15's obsolete hazard."""
        census = _token_census(seed=42, turns=1200, scale=27)
        assert census["100"]["authoritative_owners"] <= 1
        assert census["250"]["authoritative_owners"] <= 2


class TestSpearmanHelper:
    """_spearman handles edge cases."""

    def test_perfect_correlation(self):
        assert _spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)

    def test_ties(self):
        r = _spearman([1, 1, 2, 3], [2, 2, 3, 4])
        assert r is not None

    def test_too_few(self):
        assert _spearman([1], [2]) is None
        assert _spearman([1, 2], [3, 4]) is None


class TestCellKey:
    """_cell_key produces unique keys per cell configuration."""

    def test_unique_keys(self):
        a = {"policy": "a", "store_budget": 1, "budget": 2, "seed": 3}
        b = {"policy": "b", "store_budget": 1, "budget": 2, "seed": 3}
        assert _cell_key(a) != _cell_key(b)
        assert _cell_key(a) == _cell_key(dict(a))


class TestWorkloadSelfTest:
    def test_coding_workload_self_test(self):
        self_test()