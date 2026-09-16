"""E8 — external-style needled-QA benchmark (task B/E8).

A local, versioned suite modelled on the LongMemEval protocol: short
conversations, each carrying one distant fact, terminated by a question whose
answer requires that fact. Each memory method is scored on whether the needed
fact is actually *retrievable at the point of need*, plus the token cost paid.

Honesty: this is a local needled-QA benchmark tagged ``synthetic`` — it is *not*
the real LongMemEval dataset and makes no claim to be. It exists to test each
method's recall-under-quality-of-query behavior in a formalized QA form, and to
give the paper pipeline an external-data-shaped number.

Suite version: ``e8-v1``. Categories mirror LongMemEval-style item types.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from experiments.paper import load_settings, replay_adaptive  # noqa: E402
from baselines.baseline_runner import BaselineRunner, fact_matches  # noqa: E402

SUITE_VERSION = "e8-v1"

_CATEGORIES = [
    "who_am_i",
    "project_knowledge",
    "statement_on_question",
    "question_on_statement",
    "correction",
    "trap_remains",
]


def _turns(messages: List[str], facts: Dict[str, List[Dict]]) -> List[Dict]:
    """Turn dicts for a list of user messages.
    facts maps turn-index (1-based) -> list of fact dicts (source_turn assigned)."""
    stream = []
    for i, msg in enumerate(messages, start=1):
        f_list = facts.get(i, [])
        for f in f_list:
            f["source_turn_id"] = i
        stream.append({
            "turn_id": i,
            "user": msg,
            "tokens": max(1, len(msg.split())),
            "facts": [dict(f) for f in f_list],
        })
    return stream


def _distractor(bank_id: int) -> str:
    pools = [
        "Routine check: all daemons report healthy and the nightly backup window is clear.",
        "Standup notes: the CI matrix passed on every runner and no regressions leaked in.",
        "Ops log: cache hit ratios are nominal and error budgets show headroom.",
        "No outstanding incidents; the on-call rotation is quiet today.",
    ]
    return pools[bank_id % len(pools)]


def build_suite(n_per_category: int = 4, idle_turns: int = 6) -> List[Dict]:
    """Deterministic needled conversation suite.

    idle_turns = filler turns between the injection and the question, controlling
    how far the fact must travel across the window/decay curve."""
    items: List[Dict] = []
    counter = 0

    def add(cat: str, inject_msg: str, inject_fact: Dict, question: str,
            expected: str, paraphrase: str = None, correction=False):
        nonlocal counter
        counter += 1
        messages = [inject_msg] + [_distractor(i) for i in range(idle_turns)] + [question]
        if correction:
            inject_fact["category"] = "project_context"
            fact_b = {"fact": expected, "category": "project_context", "confidence": 0.95}
            # correction turn sits midway between injection and question
            messages.insert(idle_turns // 2 + 1,
                            f"Update that decision: {expected} is the new rule.")
            injected_week = {1: [dict(inject_fact)],
                             idle_turns // 2 + 2: [fact_b]}
            expected_source = idle_turns // 2 + 2
        else:
            injected_week = {1: [dict(inject_fact)]}
            expected_source = 1
        stream = _turns(messages, injected_week)
        items.append({
            "id": f"{cat}-{counter:02d}",
            "category": cat,
            "turns": stream,
            "question_turn_id": stream[-1]["turn_id"],
            "question": question,
            "expected_fact": expected,
            "expected_source_turn": expected_source,
            "paraphrase_question": paraphrase or question,
            "is_trap": cat in ("trap_remains",),
        })

    for _ in range(n_per_category):
        add("who_am_i",
            "Heads up, remember this: Dana prefers mentor review over table stakes.",
            {"fact": "Dana prefers mentor review over table stakes", "category": "personal", "confidence": 0.9},
            "What does Dana prefer for code review?",
            "Dana prefers mentor review over table stakes")
        add("project_knowledge",
            "Requirement to record: the rollout uses the canary lane first, then full.",
            {"fact": "the rollout uses the canary lane first, then full",
             "category": "project_context", "confidence": 0.95},
            "Which lane does the rollout hit first?",
            "the canary lane")
        add("statement_on_question",
            "Note this: every feature flag now defaults to off at org level.",
            {"fact": "every feature flag now defaults to off at org level",
             "category": "technical_preference", "confidence": 0.9},
            "At org level, what is the default state for feature flags?",
            "off")
        add("question_on_statement",
            "The retention window for audit logs is ninety days.",
            {"fact": "retention window for audit logs is ninety days",
             "category": "technical_preference", "confidence": 0.9},
            "How long should audit logs be retained?",
            "ninety days")
        add("correction",
            "Decision: the staging branch is the default merge target.",
            {"fact": "the staging branch is the default merge target",
             "category": "project_context", "confidence": 0.9},
            "Which branch is the default merge target now?",
            "the production branch",
            correction=True)
        add("trap_remains",
            "Remember the database port for the core service stays 5432.",
            {"fact": "core service database port stays 5432",
             "category": "technical_preference", "confidence": 0.9},
            "What port does the core service's database use?",
            "5432")

    return items


def evaluate_item_adaptive(item: Dict, settings: Dict, scorer, decay, retriever,
                           compressor, fact_tokens, embed_fn, embedding_model) -> Dict:
    replay = replay_adaptive(item["turns"], settings, scorer, decay, retriever, compressor,
                             fact_tokens=fact_tokens, embed_fn=embed_fn,
                             embedding_model=embedding_model)
    retrieved = replay["per_turn"][-1]["retrieved"]
    hit = any(fact_matches({"source_turn": item["expected_source_turn"], "fact": item["expected_fact"]},
                           [r], overlap=0.65) for r in retrieved)
    tokens = replay["per_turn"][-1]["injected_tokens"]
    return {"recalled": hit, "injected_tokens": tokens, "retrieved_n": len(retrieved)}


def evaluate_item_baseline(item: Dict, settings: Dict, method: str,
                           fact_tokens, embed_fn, embedding_model) -> Dict:
    runner = BaselineRunner(budget=int(settings["max_context_tokens"]),
                            top_k=int(settings["top_k"]),
                            embed_fn=embed_fn, fact_tokens=fact_tokens,
                            embedding_model=embedding_model)
    res = runner.run(item["turns"], [], method)
    last = res["per_turn"][-1]
    retrieved = last["retrieved"] if "retrieved" in last else []
    expected = {"source_turn": item["expected_source_turn"], "fact": item["expected_fact"]}
    hit = any(fact_matches(expected, [r], overlap=0.65) for r in retrieved)
    return {"recalled": hit, "injected_tokens": last["injected_tokens"], "retrieved_n": len(retrieved)}


def evaluate_suite(methods: List[str], settings: Dict, suite: List[Dict],
                   fact_tokens=None, embed_fn=None, embedding_model: str = "nomic-embed-text") -> Dict:
    from memory_optimizer.compression import MemoryCompressor
    from memory_optimizer.decay import CategoryDecayEngine
    from memory_optimizer.retrieval import MemoryRetriever
    from memory_optimizer.scoring import ImportanceScorer

    scorer = ImportanceScorer(weights=settings["scoring_weights"])
    decay = CategoryDecayEngine(lambdas=settings["decay_lambdas"],
                                pruning_threshold=float(settings["pruning"]["threshold"]))
    compressor = MemoryCompressor()

    results = {m: [] for m in methods}
    for item in suite:
        for m in methods:
            if m == "adaptive":
                retriever = MemoryRetriever(top_k=int(settings["top_k"]),
                                            sim_threshold=float(settings["similarity_threshold"]),
                                            embed_fn=embed_fn, embedding_model=embedding_model)
                r = evaluate_item_adaptive(item, settings, scorer, decay, retriever, compressor,
                                           fact_tokens, embed_fn, embedding_model)
            else:
                r = evaluate_item_baseline(item, settings, m, fact_tokens, embed_fn, embedding_model)
            results[m].append({**r, "id": item["id"], "category": item["category"]})

    summary = {}
    for m, rows in results.items():
        by_cat = {}
        for cat in _CATEGORIES:
            cat_rows = [r for r in rows if r["category"] == cat]
            if not cat_rows:
                continue
            hits = sum(r["recalled"] for r in cat_rows)
            by_cat[cat] = {
                "n": len(cat_rows),
                "recall": round(hits / len(cat_rows), 3),
            }
        overall = sum(r["recalled"] for r in rows) / len(rows) if rows else 0.0
        mean_tokens = sum(r["injected_tokens"] for r in rows) / len(rows) if rows else 0.0
        summary[m] = {
            "overall_recall": round(overall, 3),
            "mean_injected_tokens": round(mean_tokens, 2),
            "by_category": by_cat,
            "n_items": len(rows),
        }
    return summary, results


def main() -> None:
    ap = argparse.ArgumentParser(description="E8 needled-QA suite")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--methods", default="adaptive,sliding_window,memgpt_style,summarization_only,vanilla_rag")
    ap.add_argument("--n-per-category", type=int, default=4)
    ap.add_argument("--idle-turns", type=int, default=6)
    ap.add_argument("--embedding-model", default="nomic-embed-text")
    args = ap.parse_args()

    if args.quick:
        args.n_per_category, args.idle_turns, args.embedding_model = 1, 2, ""

    settings = load_settings()
    suite = build_suite(n_per_category=args.n_per_category, idle_turns=args.idle_turns)

    embed_fn = None
    if args.embedding_model:
        from memory_optimizer.embeddings import embed_ollama
        embed_fn = lambda texts: embed_ollama(texts, model=args.embedding_model,
                                              endpoint="http://localhost:11434")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    summary, per_item = evaluate_suite(methods, settings, suite,
                                       fact_tokens=None, embed_fn=embed_fn,
                                       embedding_model=args.embedding_model)

    payload = {
        "suite_version": SUITE_VERSION,
        "run_at": time.time(),
        "items": len(suite),
        "idle_turns": args.idle_turns,
        "embedding_model": args.embedding_model or "lexical",
        "tag": "synthetic (local needled-QA; NOT the real LongMemEval dataset)",
        "per_item": per_item,
        "summary": summary,
    }

    out = BASE_DIR / "experiments/results" / f"e8_{SUITE_VERSION}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))

    print(f"E8 suite {SUITE_VERSION}: {len(suite)} items, {args.idle_turns} idle turns")
    for m, s in summary.items():
        print(f"  {m:18s} recall={s['overall_recall']:.3f}  mean_inj_tok={s['mean_injected_tokens']:7.2f}  n={s['n_items']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()