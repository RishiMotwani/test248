"""e12 - Coding-context usefulness benchmark (task: does adaptive memory help coding tasks?).

What E10 could not answer
-------------------------
E10 probed the *store* with verbatim fact repetitions: it measured retention
("is the fact still stored"), not usefulness ("does the active context actually
handed to the model at query time contain the information needed to answer a
coding question"). Retention without retrieval means nothing for a coding task:
a context-constrained LLM can only answer from what is injected.

E12 design
----------
A deterministic coding-session workload (``data/coding_workload.py``) is replayed
once per method at the SAME active-context token budget. At the end of the
session each method is asked the same ground-truth coding questions. For every
question we inspect the method's query-time ``retrieve`` result — the tokens it
would actually inject — and score answerability by *exact token presence*:

* the required value must appear (e.g. ``300``, ``pubsub``, ``rs256``,
  ``idempotency``, ``auth``/``database``),
* for correction/obsolete questions the superseded value must NOT appear
  (``100``, ``memcache``) — stale information surfaced to the model is scored as
  retained, not answered.

Metrics (never collapsed into one number):
* ``context_recall`` — answerability against the injected active context
  (the real deliverable), store_recall — answerability against the method's full
  store/held facts (retention without injection),
* ``long_range_recall`` — sub-set of questions whose answer fact lies >= 20
  turns behind,
* ``correction_recall`` — new value surfaced AND old value absent,
* ``obsolete_retention`` — share of obsolete questions where the stale value is
  STILL injected (lower is better),
* ``mean_context_tokens`` — tokens the LLM would actually receive per question,
* ``recall_per_1k_tokens`` — task-answerable info per 1000 active tokens
  (= INFO / ACTIVE CONTEXT TOKENS, section 4 of the objective).

All evaluation is deterministic and local: word-count tokens, lexical retrieval
(embed_fn=None), no LLM calls. This is an upper-bound-ish proxy for task success
(answerable-in-context), honestly labelled as such — it is not a downstream
end-to-end QA score.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.coding_workload import build_coding_session, self_test  # noqa: E402
from experiments.paper import load_settings, replay_adaptive  # noqa: E402
from memory_optimizer.compression import MemoryCompressor  # noqa: E402
from memory_optimizer.decay import CategoryDecayEngine  # noqa: E402
from memory_optimizer.retrieval import MemoryRetriever  # noqa: E402
from memory_optimizer.scoring import ImportanceScorer  # noqa: E402

BUDGETS = [128, 256, 512]
TURNS = 400
SEEDS = [42, 43, 44]
SCALE = 3
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_ENDPOINT = "http://localhost:11434"
METHODS = ["adaptive", "sliding_window", "memgpt_style", "summarization_only", "vanilla_rag"]


def word_count(text: str) -> int:
    return max(1, len(str(text).split()))


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _entry_tokens(entry: Dict) -> int:
    if entry.get("injected_tokens") is not None:
        return int(entry["injected_tokens"])
    return word_count(entry.get("fact", entry.get("user", "")))


def _answered(query: Dict, entries: List[Dict], text_of) -> bool:
    """Exact-token answerability against a list of injected/store entries."""
    haystack: set = set()
    for e in entries or []:
        haystack |= _tokens(text_of(e))
    for tok in query.get("required", []):
        if tok.lower() not in haystack:
            return False
    for tok in query.get("forbidden", []):
        if tok.lower() in haystack:
            return False
    return True


def _answer_parts(query: Dict, entries: List[Dict], text_of) -> Dict:
    haystack: set = set()
    for e in entries or []:
        haystack |= _tokens(text_of(e))
    missing_req = [t for t in query.get("required", []) if t.lower() not in haystack]
    present_forb = [t for t in query.get("forbidden", []) if t.lower() in haystack]
    return {
        "answered": not missing_req and not present_forb,
        "missing_required": missing_req,
        "present_forbidden": present_forb,
    }


def _fact_text(e: Dict) -> str:
    return e.get("fact", e.get("user", ""))


def _metrics(queries: List[Dict], context_entries_by_qid: Dict,
             store_entries_by_qid: Dict) -> Dict:
    n = len(queries)
    corr_q = [q for q in queries if q["qtype"] in ("correction", "obsolete")]
    long_q = [q for q in queries if q.get("long_range")]

    def answered(entries_map, qs):
        if not qs:
            return None
        return round(
            sum(1 for q in qs if _answered(q, entries_map[q["qid"]], _fact_text)) / len(qs), 3)

    cx = {k: context_entries_by_qid.get(k, []) for k in context_entries_by_qid}
    st = {k: store_entries_by_qid.get(k, []) for k in store_entries_by_qid}

    # obsolete_retention: at least one forbidden value injected among obsolete
    # questions (present_forbidden non-empty for that question).
    def retention(qs, entries_map):
        if not qs:
            return None
        hits = 0
        for q in qs:
            parts = _answer_parts(q, entries_map[q["qid"]], _fact_text)
            if parts["present_forbidden"]:
                hits += 1
        return round(hits / len(qs), 3)

    token_costs = [sum(_entry_tokens(e) for e in cx[q["qid"]]) for q in queries]
    mean_tokens = round(sum(token_costs) / n, 2) if n else 0.0

    context_recall = answered(cx, queries)
    store_recall = answered(st, queries)
    long_range = answered(cx, long_q)
    correction = answered(cx, corr_q)
    obsolete_ret = retention(corr_q, cx)
    density = (round(context_recall * 1000.0 / mean_tokens, 4)
               if context_recall is not None and mean_tokens else None)

    return {
        "context_recall": context_recall,
        "store_recall": store_recall,
        "long_range_recall": long_range,
        "correction_recall": correction,
        "obsolete_retention": obsolete_ret,
        "mean_context_tokens": mean_tokens,
        "recall_per_1k_tokens": density,
        "queries": n,
    }


def run_cell(seed: int, budget: int, turns: int = TURNS, scale: int = SCALE,
             methods: List[str] = None, embed_fn=None,
             embedding_model: str = EMBEDDING_MODEL,
             use_embeddings: bool = True) -> Dict:
    methods = methods or METHODS
    if use_embeddings and embed_fn is None:
        from memory_optimizer.embeddings import embed_ollama
        embed_fn = lambda texts: embed_ollama(texts, model=embedding_model,
                                              endpoint=OLLAMA_ENDPOINT)
    stream, gt, queries = build_coding_session(seed, turns, include_corrections=True,
                                               scale=scale)
    raw_tokens = sum(h.get("tokens") or 1 for h in stream)
    natural_store_tokens = sum(len(g["fact"].split()) for g in gt if not g.get("superseded_by"))
    settings = load_settings({"max_context_tokens": budget})
    top_k = int(settings["top_k"])
    sim_th = float(settings.get("similarity_threshold", 0.35))

    def prepare_adaptive():
        scorer = ImportanceScorer(weights=settings["scoring_weights"])
        decay = CategoryDecayEngine(lambdas=settings["decay_lambdas"],
                                    pruning_threshold=float(settings["pruning"]["threshold"]))
        retriever = MemoryRetriever(top_k=top_k, sim_threshold=sim_th, embed_fn=embed_fn,
                                    embedding_model=embedding_model)
        compressor = MemoryCompressor()
        rep = replay_adaptive(stream, settings, scorer, decay, retriever, compressor,
                              fact_tokens=word_count, embed_fn=embed_fn,
                              embedding_model=embedding_model)
        return rep["memories"], rep["per_turn"]

    def prepare_baseline(name):
        from baselines.memgpt_style import MemGPTStyleBaseline
        from baselines.sliding_window import SlidingWindowBaseline
        from baselines.summarization_only import SummarizationOnlyBaseline
        from baselines.vanilla_rag import VanillaRAGBaseline
        table = {
            "sliding_window": SlidingWindowBaseline,
            "memgpt_style": MemGPTStyleBaseline,
            "summarization_only": SummarizationOnlyBaseline,
            "vanilla_rag": VanillaRAGBaseline,
        }
        return table[name](budget=budget, top_k=top_k, embed_fn=embed_fn,
                           fact_tokens=word_count, embedding_model=embedding_model)

    out_methods: Dict[str, Dict] = {}

    # ---- adaptive ---------------------------------------------------------
    if "adaptive" in methods:
        memories, per_turn = prepare_adaptive()
        qr = MemoryRetriever(top_k=top_k, sim_threshold=sim_th, embed_fn=embed_fn,
                             embedding_model=embedding_model)
        cx, st = {}, {}
        for q in queries:
            injected = qr.retrieve(q["user"], memories, current_turn=q["query_turn"],
                                   fact_tokens=word_count)
            cx[q["qid"]] = injected
            st[q["qid"]] = memories
        m = _metrics(queries, cx, st)
        m["store_count"] = len(memories)
        m["store_tokens"] = sum(word_count(x["fact"]) for x in memories)
        m["injected"] = [
            {"turn_id": r["turn_id"], "tokens": r["injected_tokens"]} for r in per_turn]
        # per-method mean injected tokens over the whole session (E1 parity)
        m["mean_session_injected_tokens"] = round(
            sum(r["injected_tokens"] for r in per_turn) / len(per_turn), 2)
        m["replay"] = "adaptive"
        out_methods["adaptive"] = m

    # ---- baselines ---------------------------------------------------------
    for name in methods:
        if name == "adaptive":
            continue
        baseline = prepare_baseline(name)
        for turn in stream:
            baseline.observe(turn)
        cx, st = {}, {}
        for q in queries:
            injected = baseline.retrieve(q["user"])
            cx[q["qid"]] = injected
            held = baseline.held_facts()
            st[q["qid"]] = held
        m = _metrics(queries, cx, st)
        m["store_count"] = len(st[queries[0]["qid"]])
        m["store_tokens"] = sum(_entry_tokens(x) for x in st[queries[0]["qid"]])
        session_inj = 0.0
        for turn in stream:
            session_inj += sum(_entry_tokens(e) for e in baseline.retrieve(turn["user"]))
        m["mean_session_injected_tokens"] = round(
            session_inj / len(stream), 2) if stream else 0.0
        out_methods[name] = m

    return {
        "seed": seed,
        "budget": budget,
        "turns": len(stream),
        "scale": scale,
        "raw_conversation_tokens": raw_tokens,
        "natural_store_tokens": natural_store_tokens,
        "budget_stressed": natural_store_tokens > budget,
        "query_count": len(queries),
        "methods": out_methods,
    }


def run_benchmark(budgets=None, seeds=None, methods=None, turns: int = TURNS,
                  scale: int = SCALE, use_embeddings: bool = True) -> Dict:
    """Run the full E12 cell grid.

    ``use_embeddings=True`` routes retrieval through Ollama's nomic-embed-text
    (deterministic for identical inputs; local, no LLM) — the pipeline's actual
    production retrieval path. ``use_embeddings=False`` falls back to lexical
    overlap, which is only meaningful for the baselines and is reported as such.
    """
    budgets = budgets or BUDGETS
    seeds = seeds or SEEDS
    methods = methods or METHODS

    if use_embeddings:
        from memory_optimizer.embeddings import embed_ollama
        embed_fn = lambda texts: embed_ollama(texts, model=EMBEDDING_MODEL, endpoint=OLLAMA_ENDPOINT)
        embedding_model = EMBEDDING_MODEL
    else:
        embed_fn, embedding_model = None, "lexical"

    cells = []
    for budget in budgets:
        for seed in seeds:
            print(f"  cell budget={budget} seed={seed} ...", flush=True)
            cells.append(run_cell(seed, budget, turns, scale, methods, embed_fn,
                                  embedding_model))

    agg_by_budget: Dict[str, Dict] = {}
    for budget in budgets:
        rows = [c for c in cells if c["budget"] == budget]
        m_agg = {}
        for m in methods:
            vals = [c["methods"][m] for c in rows if m in c["methods"]]

            def mean(key, fmt=3):
                xs = [v[key] for v in vals if v.get(key) is not None]
                return round(float(sum(xs) / len(xs)), fmt) if xs else None

            m_agg[m] = {
                "context_recall_mean": mean("context_recall"),
                "store_recall_mean": mean("store_recall"),
                "long_range_recall_mean": mean("long_range_recall"),
                "correction_recall_mean": mean("correction_recall"),
                "obsolete_retention_mean": mean("obsolete_retention"),
                "mean_context_tokens": mean("mean_context_tokens", 2),
                "recall_per_1k_tokens": mean("recall_per_1k_tokens", 4),
                "mean_session_injected_tokens": mean("mean_session_injected_tokens", 2),
                "store_tokens_mean": mean("store_tokens"),
                "store_count_mean": mean("store_count"),
            }
        raw = [c["raw_conversation_tokens"] for c in rows]
        store = [c["natural_store_tokens"] for c in rows]
        agg_by_budget[str(budget)] = {
            "budget": budget,
            "raw_conversation_tokens": round(sum(raw) / len(raw), 1),
            "natural_store_tokens": round(sum(store) / len(store), 1),
            "budget_stressed": all(c["budget_stressed"] for c in rows),
            "store_to_budget_ratio": round(
                (sum(store) / len(store)) / budget, 2) if store else None,
            "methods": m_agg,
        }

    return {
        "experiment": "e12_coding_benchmark",
        "config": {
            "turns": turns, "budgets": budgets, "seeds": seeds, "methods": methods,
            "scale": scale, "token_mode": "word_count", "embedding_model": embedding_model,
            "write": "oracle", "workload": "data.coding_workload.build_coding_session",
            "matching": "exact answer-token presence in query-time retrieve() output",
        },
        "cells": cells,
        "aggregate_by_budget": agg_by_budget,
        "note": (
            "context_recall = answerable from the query-time INJECTED context "
            "(tokens the model actually receives); store_recall = answerable from the "
            "method's full store (retention without retrieval). recall_per_1k_tokens = "
            "1000 * context_recall / mean_context_tokens (useful task info per active "
            "context token). obsolete_retention = share of correction/obsolete questions "
            "where the superseded value is still injected (lower better). budget_stressed "
            "means the natural store (all facts, no eviction) exceeds the budget — the "
            "budget genuinely binds. embeddings route through Ollama nomic-embed-text "
            "(local, deterministic for identical inputs); lexical mode is reported only "
            "for the baselines."
        ),
    }


if __name__ == "__main__":
    self_test()
    payload = run_benchmark()
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "e12_coding_benchmark.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {path}")