"""e13 - Phase 9 Generalization, Ablation, and Fairness Validation.

This experiment validates whether the Phase-8 architectural changes (separate
memory-store capacity from active context, query-first retrieval) generalize
across seeds, budgets, query families, and retrieval modes.

It performs:
1. Generalization experiment across new seeds and budgets
2. Causal ablation A: store-capacity separation (coupled vs matched vs 4x vs unbounded)
4. Causal ablation B: retrieval policy profiles (Phase-8 vs legacy vs pure sim vs pure imp)
5. Embedding-sensitivity validation (embedding vs lexical)
6. Query-family analysis by qtype
7. Strict budget utilization and violation tracking

All results are deterministic and machine-readable.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.coding_workload import build_coding_session  # noqa: E402
from experiments.paper import load_settings, replay_adaptive  # noqa: E402
from memory_optimizer.compression import MemoryCompressor  # noqa: E402
from memory_optimizer.decay import CategoryDecayEngine  # noqa: E402
from memory_optimizer.retrieval import MemoryRetriever  # noqa: E402
from memory_optimizer.scoring import ImportanceScorer  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BUDGETS = [64, 128, 256, 512, 1024]
SEEDS = [42, 43, 44, 45, 46]
TURNS = 400
SCALE = 3
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_ENDPOINT = "http://localhost:11434"
METHODS = ["adaptive", "sliding_window", "memgpt_style", "summarization_only", "vanilla_rag"]

# Ablation A: Store-capacity policies
STORE_POLICIES = [
    ("coupled_old_behavior", "active"),      # store_budget = active_budget (pre-Phase-8)
    ("matched_budget", "active"),            # store_budget = active_budget (matched)
    ("separated_4x", "4x"),                  # store_budget = max(4096, active * 4) (Phase-8)
    ("unbounded", 0),                        # store_budget = 0 (no hard cap)
]

# Ablation B: Retrieval profiles
RETRIEVAL_PROFILES = {
    "phase8": {
        "imp_weight": 0.15,
        "sim_weight": 0.85,
        "cat_bonus": 0.02,
    },
    "legacy_phase7": {
        "imp_weight": 0.60,
        "sim_weight": 0.40,
        "cat_bonus": 0.10,
    },
    "pure_similarity": {
        "imp_weight": 0.00,
        "sim_weight": 1.00,
        "cat_bonus": 0.00,
    },
    "pure_importance": {
        "imp_weight": 1.00,
        "sim_weight": 0.00,
        "cat_bonus": 0.00,
    },
}

# Lexical sensitivity
LEXICAL_SEEDS = [42, 43]
LEXICAL_BUDGETS = [128, 256, 512]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _query_family(q: Dict) -> str:
    """Map qtype to query family for analysis."""
    qtype = q.get("qtype", "unknown")
    # Map to broader families
    family_map = {
        "requirement": "requirements",
        "architecture": "architecture",
        "constraint": "constraints",
        "implementation": "implementation",
        "bug_fix": "bug_fix",
        "module_relation": "module_relation",
        "feature_flag": "feature_flag",
        "correction": "correction",
        "obsolete": "obsolete",
    }
    return family_map.get(qtype, qtype)


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
    max_tokens = max(token_costs) if token_costs else 0

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
        "max_context_tokens": max_tokens,
        "recall_per_1k_tokens": density,
        "queries": n,
    }


def _family_metrics(queries: List[Dict], context_entries_by_qid: Dict,
                    store_entries_by_qid: Dict) -> Dict[str, Dict]:
    """Compute metrics broken down by query family."""
    cx = {k: context_entries_by_qid.get(k, []) for k in context_entries_by_qid}
    st = {k: store_entries_by_qid.get(k, []) for k in store_entries_by_qid}

    families = {}
    for q in queries:
        fam = _query_family(q)
        if fam not in families:
            families[fam] = []
        families[fam].append(q)

    result = {}
    for fam, qs in families.items():
        token_costs = [sum(_entry_tokens(e) for e in cx[q["qid"]]) for q in qs]
        mean_tokens = round(sum(token_costs) / len(qs), 2) if qs else 0.0
        max_tokens = max(token_costs) if token_costs else 0

        corr_q = [q for q in qs if q["qtype"] in ("correction", "obsolete")]
        long_q = [q for q in qs if q.get("long_range")]

        def answered(entries_map, qs_list):
            if not qs_list:
                return None
            return round(
                sum(1 for q in qs_list if _answered(q, entries_map[q["qid"]], _fact_text)) / len(qs_list), 3)

        def retention(qs_list, entries_map):
            if not qs_list:
                return None
            hits = 0
            for q in qs_list:
                parts = _answer_parts(q, entries_map[q["qid"]], _fact_text)
                if parts["present_forbidden"]:
                    hits += 1
            return round(hits / len(qs_list), 3)

        result[fam] = {
            "query_count": len(qs),
            "context_recall": answered(cx, qs),
            "store_recall": answered(st, qs),
            "long_range_recall": answered(cx, long_q),
            "correction_recall": answered(cx, corr_q),
            "obsolete_retention": retention(corr_q, cx),
            "mean_context_tokens": mean_tokens,
            "max_context_tokens": max_tokens,
        }
    return result


def run_cell(
    seed: int,
    budget: int,
    turns: int = TURNS,
    scale: int = SCALE,
    methods: List[str] = None,
    embed_fn=None,
    embedding_model: str = EMBEDDING_MODEL,
    use_embeddings: bool = True,
    # Ablation controls
    store_policy: str = "separated_4x",
    store_multiplier: float = 4.0,
    retrieval_profile: str = "phase8",
) -> Dict:
    """Run a single benchmark cell with optional ablation controls."""
    methods = methods or METHODS
    if use_embeddings and embed_fn is None:
        from memory_optimizer.embeddings import embed_ollama
        embed_fn = lambda texts: embed_ollama(texts, model=embedding_model,
                                              endpoint=OLLAMA_ENDPOINT)

    stream, gt, queries = build_coding_session(seed, turns, include_corrections=True,
                                               scale=scale)
    raw_tokens = sum(h.get("tokens") or 1 for h in stream)
    natural_store_tokens = sum(len(g["fact"].split()) for g in gt if not g.get("superseded_by"))

    # Resolve store budget based on policy
    if store_policy == "coupled_old_behavior":
        store_budget = budget
    elif store_policy == "matched_budget":
        store_budget = budget
    elif store_policy == "separated_4x":
        store_budget = max(4096, int(budget * store_multiplier))
    elif store_policy == "unbounded":
        store_budget = 0
    else:
        raise ValueError(f"Unknown store_policy: {store_policy}")

    # Resolve retrieval profile
    profile = RETRIEVAL_PROFILES.get(retrieval_profile, RETRIEVAL_PROFILES["phase8"])

    settings = load_settings({
        "max_context_tokens": budget,
        "injection_token_limit": budget,
        "memory_store_token_budget": store_budget,
    })
    top_k = int(settings["top_k"])
    sim_th = float(settings.get("similarity_threshold", 0.35))

    # Prepare adaptive with custom retrieval profile
    def prepare_adaptive():
        scorer = ImportanceScorer(weights=settings["scoring_weights"])
        decay = CategoryDecayEngine(lambdas=settings["decay_lambdas"],
                                    pruning_threshold=float(settings["pruning"]["threshold"]))
        retriever = MemoryRetriever(
            top_k=top_k, sim_threshold=sim_th, embed_fn=embed_fn,
            embedding_model=embedding_model,
            imp_weight=profile["imp_weight"],
            sim_weight=profile["sim_weight"],
            cat_bonus=profile["cat_bonus"],
        )
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
        qr = MemoryRetriever(
            top_k=top_k, sim_threshold=sim_th, embed_fn=embed_fn,
            embedding_model=embedding_model,
            imp_weight=profile["imp_weight"],
            sim_weight=profile["sim_weight"],
            cat_bonus=profile["cat_bonus"],
        )
        cx, st = {}, {}
        store_recalls = []
        retrieval_losses = []
        budget_violations = 0
        per_query_diagnostics = []

        for q in queries:
            injected = qr.retrieve(q["user"], memories, current_turn=q["query_turn"],
                                   token_limit=budget, fact_tokens=word_count)
            actual_tokens = sum(word_count(e["fact"]) for e in injected)
            violation = actual_tokens > budget
            if violation:
                budget_violations += 1

            cx[q["qid"]] = injected
            st[q["qid"]] = memories

            store_ans = _answered(q, memories, _fact_text)
            context_ans = _answered(q, injected, _fact_text)
            if store_ans:
                store_recalls.append(1)
                retrieval_losses.append(0 if context_ans else 1)
            else:
                store_recalls.append(0)
                retrieval_losses.append(0)

            per_query_diagnostics.append({
                "qid": q["qid"],
                "qtype": q["qtype"],
                "family": _query_family(q),
                "actual_context_tokens": actual_tokens,
                "budget_violation": violation,
                "store_ans": store_ans,
                "context_ans": context_ans,
            })

        m = _metrics(queries, cx, st)
        m["store_count"] = len(memories)
        m["store_tokens"] = sum(word_count(x["fact"]) for x in memories)
        m["injected"] = [
            {"turn_id": r["turn_id"], "tokens": r["injected_tokens"]} for r in per_turn]
        m["mean_session_injected_tokens"] = round(
            sum(r["injected_tokens"] for r in per_turn) / len(per_turn), 2)
        m["replay"] = "adaptive"
        m["active_context_budget"] = budget
        m["store_budget"] = store_budget
        m["mean_context_utilization"] = round(
            m["mean_context_tokens"] / budget, 4) if budget else None
        m["max_context_utilization"] = round(
            m.get("max_context_tokens", 0) / budget, 4) if budget else None
        m["budget_violation_count"] = budget_violations
        m["store_recall"] = round(sum(store_recalls) / len(store_recalls), 3) if store_recalls else None
        m["retrieval_loss"] = round(sum(retrieval_losses) / len(retrieval_losses), 3) if retrieval_losses else None
        m["store_budget_binding"] = sum(word_count(x["fact"]) for x in memories) > store_budget if store_budget > 0 else None
        m["natural_store_tokens"] = natural_store_tokens
        m["budget_eviction_count"] = len([e for e in per_turn if e.get("budget_evicted", False)])
        m["per_query_diagnostics"] = per_query_diagnostics
        m["family_metrics"] = _family_metrics(queries, cx, st)
        m["retrieval_profile"] = retrieval_profile
        m["store_policy"] = store_policy
        out_methods["adaptive"] = m

    # ---- baselines ---------------------------------------------------------
    for name in methods:
        if name == "adaptive":
            continue
        baseline = prepare_baseline(name)
        for turn in stream:
            baseline.observe(turn)
        cx, st = {}, {}
        budget_violations = 0
        for q in queries:
            injected = baseline.retrieve(q["user"])
            actual_tokens = sum(_entry_tokens(e) for e in injected)
            if actual_tokens > budget:
                budget_violations += 1
            cx[q["qid"]] = injected
            held = baseline.held_facts()
            st[q["qid"]] = held
        m = _metrics(queries, cx, st)
        m["store_count"] = len(st[queries[0]["qid"]])
        m["store_tokens"] = sum(_entry_tokens(x) for x in st[queries[0]["qid"]])
        session_inj = 0.0
        max_inj = 0
        for turn in stream:
            inj = sum(_entry_tokens(e) for e in baseline.retrieve(turn["user"]))
            session_inj += inj
            if inj > max_inj:
                max_inj = inj
        m["mean_session_injected_tokens"] = round(
            session_inj / len(stream), 2) if stream else 0.0
        m["max_injected_tokens"] = max_inj
        m["budget_violation_count"] = budget_violations
        m["mean_context_utilization"] = round(
            m["mean_context_tokens"] / budget, 4) if budget else None
        m["max_context_utilization"] = round(
            m.get("max_context_tokens", 0) / budget, 4) if budget else None
        m["family_metrics"] = _family_metrics(queries, cx, st)
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


def run_generalization() -> Dict:
    """Run the main generalization experiment across seeds and budgets."""
    cells = []
    for budget in BUDGETS:
        for seed in SEEDS:
            print(f"  [generalization] budget={budget} seed={seed} ...", flush=True)
            cells.append(run_cell(seed, budget, turns=TURNS, scale=SCALE,
                                  methods=METHODS, use_embeddings=True))

    return _aggregate(cells, "generalization")


def run_store_ablation() -> Dict:
    """Run ablation A: store-capacity separation."""
    cells = []
    for policy_name, _ in STORE_POLICIES:
        for budget in BUDGETS:
            for seed in [42, 43, 44]:  # subset for ablation
                print(f"  [store_ablation] policy={policy_name} budget={budget} seed={seed} ...", flush=True)
                cells.append(run_cell(seed, budget, turns=TURNS, scale=SCALE,
                                      methods=["adaptive"], use_embeddings=True,
                                      store_policy=policy_name))
    return _aggregate(cells, "store_ablation")


def run_retrieval_ablation() -> Dict:
    """Run ablation B: retrieval policy profiles."""
    cells = []
    for profile_name in RETRIEVAL_PROFILES.keys():
        for budget in [128, 256, 512]:
            for seed in [42, 43, 44]:
                print(f"  [retrieval_ablation] profile={profile_name} budget={budget} seed={seed} ...", flush=True)
                cells.append(run_cell(seed, budget, turns=TURNS, scale=SCALE,
                                      methods=["adaptive"], use_embeddings=True,
                                      store_policy="separated_4x",
                                      retrieval_profile=profile_name))
    return _aggregate(cells, "retrieval_ablation")


def run_lexical_sensitivity() -> Dict:
    """Run lexical vs embedding sensitivity check."""
    cells = []
    for budget in LEXICAL_BUDGETS:
        for seed in LEXICAL_SEEDS:
            print(f"  [lexical_sensitivity] budget={budget} seed={seed} ...", flush=True)
            cells.append(run_cell(seed, budget, turns=TURNS, scale=SCALE,
                                  methods=["adaptive", "vanilla_rag", "sliding_window"],
                                  use_embeddings=False,
                                  store_policy="separated_4x"))
    return _aggregate(cells, "lexical_sensitivity")


def _aggregate(cells: List[Dict], experiment_type: str) -> Dict:
    """Aggregate cells into summary statistics."""
    from collections import defaultdict

    agg_by_budget = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for c in cells:
        budget = c["budget"]
        for m, v in c["methods"].items():
            for key, val in v.items():
                if isinstance(val, (int, float)) and val is not None:
                    agg_by_budget[budget][m][key].append(val)

    agg_result = {}
    for budget, methods in agg_by_budget.items():
        agg_result[str(budget)] = {}
        for m, vals in methods.items():
            def mean(key):
                xs = vals.get(key, [])
                return round(sum(xs) / len(xs), 4) if xs else None

            agg_result[str(budget)][m] = {
                "context_recall_mean": mean("context_recall"),
                "store_recall_mean": mean("store_recall"),
                "long_range_recall_mean": mean("long_range_recall"),
                "correction_recall_mean": mean("correction_recall"),
                "obsolete_retention_mean": mean("obsolete_retention"),
                "mean_context_tokens": mean("mean_context_tokens"),
                "max_context_tokens": mean("max_context_tokens"),
                "recall_per_1k_tokens": mean("recall_per_1k_tokens"),
                "mean_context_utilization": mean("mean_context_utilization"),
                "max_context_utilization": mean("max_context_utilization"),
                "budget_violation_count_mean": mean("budget_violation_count"),
                "store_tokens_mean": mean("store_tokens"),
                "store_count_mean": mean("store_count"),
                "store_recall_mean": mean("store_recall"),
                "retrieval_loss_mean": mean("retrieval_loss"),
                "store_budget_binding_mean": mean("store_budget_binding"),
                "budget_eviction_count_mean": mean("budget_eviction_count"),
            }

    return {
        "experiment": f"e13_{experiment_type}",
        "config": {
            "turns": TURNS,
            "budgets": BUDGETS,
            "seeds": SEEDS,
            "scale": SCALE,
            "token_mode": "word_count",
            "embedding_model": EMBEDDING_MODEL,
            "write": "oracle",
            "workload": "data.coding_workload.build_coding_session",
            "matching": "exact answer-token presence in query-time retrieve() output",
            "store_policies": [p[0] for p in STORE_POLICIES],
            "retrieval_profiles": list(RETRIEVAL_PROFILES.keys()),
        },
        "cells": cells,
        "aggregate_by_budget": agg_result,
        "note": (
            "context_recall = answerable from query-time INJECTED context; "
            "store_recall = answerable from full store; "
            "retrieval_loss = store_recall - context_recall; "
            "mean_context_utilization = mean_context_tokens / active_context_budget; "
            "max_context_utilization = max_context_tokens / active_context_budget; "
            "store_budget_binding = whether store tokens exceeded store_budget; "
            "budget_violation_count = queries where injected tokens > budget."
        ),
    }


def build_coding_session(seed: int, num_turns: int = 120,
                         include_corrections: bool = True,
                         scale: int = 4):
    # Import here to avoid circular import
    from data.coding_workload import build_coding_session as _build
    return _build(seed, num_turns, include_corrections, scale)


def generate_report(result: Dict) -> str:
    """Generate a markdown report from the experiment results."""
    lines = []
    lines.append("# E13 Generalization and Ablation Report")
    lines.append("")
    lines.append("## Experiment Configuration")
    lines.append("")
    gen = result["generalization"]
    lines.append(f"- Turns: {gen['config']['turns']}")
    lines.append(f"- Scale: {gen['config']['scale']}")
    lines.append(f"- Budgets: {gen['config']['budgets']}")
    lines.append(f"- Seeds: {gen['config']['seeds']}")
    lines.append(f"- Embedding: {gen['config']['embedding_model']}")
    lines.append(f"- Methods: {gen['config'].get('methods', 'N/A')}")
    lines.append("")

    # Generalization results
    lines.append("## 1. Generalization Results (5 seeds × 5 budgets)")
    lines.append("")
    _add_aggregate_table(lines, result["generalization"]["aggregate_by_budget"],
                         "Generalization (Phase-8 defaults)")
    lines.append("")

    # Store ablation
    lines.append("## 2. Store-Capacity Ablation (Ablation A)")
    lines.append("")
    _add_ablation_table(lines, result["store_ablation"]["aggregate_by_budget"],
                        "Store-Capacity Policy", "store_ablation")
    lines.append("")

    # Retrieval ablation
    lines.append("## 3. Retrieval-Policy Ablation (Ablation B)")
    lines.append("")
    _add_ablation_table(lines, result["retrieval_ablation"]["aggregate_by_budget"],
                        "Retrieval Profile", "retrieval_ablation")
    lines.append("")

    # Lexical sensitivity
    lines.append("## 4. Embedding vs Lexical Sensitivity")
    lines.append("")
    _add_ablation_table(lines, result["lexical_sensitivity"]["aggregate_by_budget"],
                        "Retrieval Mode", "lexical_sensitivity")
    lines.append("")

    # Limitations
    lines.append("## 5. Limitations and Unresolved Questions")
    lines.append("")
    lines.append("1. **Workload scale**: Current coding workload (scale=3, ~84 facts, ~1024 natural store tokens) may not create genuine store pressure at 4x store budget (4096+). The unbounded policy may not differ from 4x if natural store < 4096.")
    lines.append("2. **Embedding variance**: Only nomic-embed-text tested. Other embedding models may yield different similarity distributions.")
    lines.append("3. **Query coverage**: Query families derived from existing qtypes; may not cover all realistic coding question types.")
    lines.append("4. **Decay not varied**: Decay policy held fixed per Phase 9 directive. Store pressure interacts with decay.")
    lines.append("5. **Budget range**: 64-token budget may be too small for meaningful retrieval; 1024 may exceed workload needs.")
    lines.append("")

    return "\n".join(lines)


def _add_aggregate_table(lines: List[str], agg: Dict, title: str):
    """Add a formatted table for aggregate results."""
    lines.append(f"### {title}")
    lines.append("")
    headers = ["Budget", "Method", "ctx_recall", "store_recall", "retrieval_loss",
               "mean_ctx_tok", "max_ctx_tok", "util", "max_util", "store_tok",
               "budget_viol", "corr_recall", "obs_ret"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    methods_order = ["adaptive", "sliding_window", "memgpt_style", "summarization_only", "vanilla_rag"]
    for budget_str in sorted(agg.keys(), key=lambda x: int(x)):
        for m in methods_order:
            if m not in agg[budget_str]:
                continue
            v = agg[budget_str][m]
            ctx = v.get("context_recall_mean")
            store = v.get("store_recall_mean")
            loss = v.get("retrieval_loss_mean")
            mean_tok = v.get("mean_context_tokens")
            max_tok = v.get("max_context_tokens")
            util = v.get("mean_context_utilization")
            max_util = v.get("max_context_utilization")
            store_tok = v.get("store_tokens_mean")
            viol = v.get("budget_violation_count_mean")
            corr = v.get("correction_recall_mean")
            obs = v.get("obsolete_retention_mean")

            def fmt(x):
                return f"{x:.3f}" if x is not None else "N/A"

            lines.append(f"| {budget_str} | {m} | {fmt(ctx)} | {fmt(store)} | {fmt(loss)} | "
                         f"{fmt(mean_tok)} | {fmt(max_tok)} | {fmt(util)} | {fmt(max_util)} | "
                         f"{fmt(store_tok)} | {fmt(viol)} | {fmt(corr)} | {fmt(obs)} |")
    lines.append("")


def _add_ablation_table(lines: List[str], agg: Dict, ablation_type: str, exp_type: str):
    """Add ablation comparison table."""
    lines.append(f"### {ablation_type} Ablation")
    lines.append("")
    lines.append("*See detailed JSON for full breakdown.*")
    lines.append("")
    if exp_type == "store_ablation":
        lines.append("- **Coupled/Matched** (store=active): Store capped at active budget.")
        lines.append("- **Separated 4x** (Phase-8): Store up to 4x active budget.")
        lines.append("- **Unbounded** (0): No hard store cap.")
    elif exp_type == "retrieval_ablation":
        lines.append("- **Phase-8** (0.15/0.85): Query-first.")
        lines.append("- **Legacy Phase-7** (0.6/0.4): Importance-dominant.")
        lines.append("- **Pure Similarity** (0/1.0): Query-only.")
        lines.append("- **Pure Importance** (1.0/0): History-only.")
    elif exp_type == "lexical_sensitivity":
        lines.append("- **Embeddings**: Production path; higher similarity discrimination.")
        lines.append("- **Lexical**: Fallback path; lower discrimination.")
    lines.append("")


if __name__ == "__main__":
    # Run self-test first
    from data.coding_workload import self_test
    self_test()

    # Run all experiments
    print("=" * 60)
    print("E13 Generalization Experiment")
    print("=" * 60)

    print("\n[1/4] Running generalization experiment...")
    gen_result = run_generalization()

    print("\n[2/4] Running store-capacity ablation (A)...")
    store_ablation_result = run_store_ablation()

    print("\n[3/4] Running retrieval-profile ablation (B)...")
    retrieval_ablation_result = run_retrieval_ablation()

    print("\n[4/4] Running lexical sensitivity check...")
    lexical_result = run_lexical_sensitivity()

    # Combine all results
    full_result = {
        "generalization": gen_result,
        "store_ablation": store_ablation_result,
        "retrieval_ablation": retrieval_ablation_result,
        "lexical_sensitivity": lexical_result,
    }

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "e13_generalization.json"
    path.write_text(json.dumps(full_result, indent=2, default=str))
    print(f"\nWrote {path}")

    # Generate markdown report
    report_path = out_dir / "e13_generalization_report.md"
    report_path.write_text(generate_report(full_result))
    print(f"Wrote {report_path}")


# Fix the missing 'title' variable in _add_ablation_table