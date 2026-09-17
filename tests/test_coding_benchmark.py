"""E12 coding-context benchmark tests (task: coding-context usefulness).

Verifies the workload generator is deterministic and grounds every query's
answer tokens, and that the benchmark harness returns well-formed, bounded
metrics without any LLM calls (lexical retrieval only).
"""

import pytest

from data.coding_workload import build_coding_session
from experiments.e12_coding_benchmark import run_cell


class TestCodingWorkload:
    def test_deterministic_seeded(self):
        s1, g1, q1 = build_coding_session(seed=7, num_turns=200)
        s2, g2, q2 = build_coding_session(seed=7, num_turns=200)
        assert s1 == s2
        assert g1 == g2
        assert q1 == q2

    def test_scale_grows_store(self):
        s1, _, _ = build_coding_session(seed=7, num_turns=300, scale=1)
        s2, _, _ = build_coding_session(seed=7, num_turns=300, scale=4)
        facts1 = sum(len(t.get("facts", [])) for t in s1)
        facts4 = sum(len(t.get("facts", [])) for t in s2)
        assert facts4 > facts1

    def test_ground_truth_query_coverage(self):
        stream, gt, queries = build_coding_session(seed=3, num_turns=200)
        # every query's required token lives in its source fact
        by_turn = {}
        for g in gt:
            by_turn.setdefault(g["source_turn"], []).append(g)
        for q in queries:
            srcs = by_turn[q["source_turn"]]
            src_words = " ".join(s["fact"] for s in srcs).lower().split()
            for tok in q["required"]:
                assert tok.lower() in src_words, f"{q['qid']} req {tok} not in fact"
            for tok in q["forbidden"]:
                # forbidden tokens (obsolete/old values) must NOT be in the new fact
                assert tok.lower() not in src_words, f"{q['qid']} forb {tok} in new fact"

    def test_corrections_supersede(self):
        stream, gt, queries = build_coding_session(seed=5, num_turns=200)
        corr = [g for g in gt if g.get("is_correction_target")]
        assert len(corr) == 2, f"expected 2 corrections, got {len(corr)}"
        for g in corr:
            old = [o for o in gt
                   if o["source_turn"] == g["supersedes_turn"]
                   and not o.get("is_correction_target")]
            assert len(old) == 1
            assert old[0]["superseded_by"] == g["source_turn"]

    def test_stream_tokens_positive(self):
        stream, gt, queries = build_coding_session(seed=9, num_turns=200)
        assert all(t.get("tokens") and t["tokens"] > 0 for t in stream)
        assert any(t.get("facts") for t in stream)  # some signal turns


class TestE12Harness:
    def test_run_cell_shape_and_bounds(self):
        cell = run_cell(seed=42, budget=256, turns=400, scale=3, use_embeddings=False)
        assert cell["raw_conversation_tokens"] > 0
        assert cell["query_count"] >= 10
        for m, v in cell["methods"].items():
            assert 0.0 <= (v["context_recall"] or 0.0) <= 1.0
            assert 0.0 <= (v["store_recall"] or 0.0) <= 1.0
            assert 0.0 <= (v["obsolete_retention"] or 0.0) <= 1.0
            assert v["mean_context_tokens"] > 0
            assert v["recall_per_1k_tokens"] is not None
        assert set(cell["methods"]) == {
            "adaptive", "sliding_window", "memgpt_style", "summarization_only", "vanilla_rag"}

    def test_deterministic_cell(self):
        a = run_cell(seed=7, budget=512, turns=400, scale=3, use_embeddings=False)
        b = run_cell(seed=7, budget=512, turns=400, scale=3, use_embeddings=False)
        assert a == b

    def test_budget_binds_at_low_budget(self):
        # natural store must exceed the largest budget so it genuinely binds
        stream, gt, queries = build_coding_session(seed=42, num_turns=400, scale=3)
        natural = sum(len(g["fact"].split()) for g in gt if not g.get("superseded_by"))
        assert natural > 512

    def test_facts_spread_across_session(self):
        # fairness: facts must not be packed into the first half and left to
        # decay through a long filler tail before the end-of-session queries
        stream, gt, queries = build_coding_session(seed=42, num_turns=400, scale=3)
        src = [g["source_turn"] for g in gt]
        assert max(src) > 240, f"latest fact at turn {max(src)} not spread"
        assert min(src) < 40, f"earliest fact at turn {min(src)} not spread"
