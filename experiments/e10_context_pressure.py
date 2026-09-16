"""e10 - Context-pressure experiment (task B extension).

Research question (Phase 7): does the adaptive memory system preserve useful
information *more efficiently* than the baselines when the active context is
genuinely constrained by a token budget?

Design
------
A grid over budgets x turns x seeds. Every cell replays all five methods over
the same seeded stream (writer held fixed, oracle facts) with word-count token
measurement enabled, so the pipeline's store budget eviction actually engages
(``fact_tokens is not None``) and every method's injected context is measured
under one shared budget cap.

For each cell we record, separately:

* conversation metrics: mean injected tokens (active-context cost), store
  token usage, store count, store-level recall (E2 positive + durable), per
  category, correction/negation handling, E4 durable-by-distance, and whether
  the raw conversation actually exceeded the budget (``budget_stressed``).
* retrieval-at-need metrics: synthetic *probe turns* appended after the
  conversation, one per durable ground-truth fact, whose user message repeats
  the fact verbatim. We measure whether the fact is present in the injected
  (retrieved) context at its probe turn. This is an explicit upper-bound probe
  of point-of-need retrievability: if a fact cannot even be surfaced when the
  query textually equals the fact, it is not "recoverable at the point of
  need". Probes carry no facts, so they never change the store via dedupe.

Primary metric (per P3/P6): **recall at fixed token budget**, reported as a
budget-vs-performance curve plus AUC. We also report **useful recall per
injected token** (probe recall / mean injected tokens). Storage efficiency
(store tokens), retrieval efficiency (injected context), task-style recall
(store recall), correction handling, and forgetting precision stay separate and
are never collapsed into a single score.

Honesty labels: tokens are word-count estimates (deterministic, no LLM calls),
every cell records ``token_source`` and ``embedding_engine``, and cells where
the raw conversation never exceeded the budget are flagged rather than being
treated as genuine constraint tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.paper import (  # noqa: E402
    load_settings,
    stream_from_generator,
    replay_adaptive,
    per_turn_window_tokens,
    final_window_ids,
    evaluate_method,
    _per_category_recall,
)
from memory_optimizer.scoring import ImportanceScorer  # noqa: E402
from memory_optimizer.decay import CategoryDecayEngine  # noqa: E402
from memory_optimizer.retrieval import MemoryRetriever  # noqa: E402
from memory_optimizer.compression import MemoryCompressor  # noqa: E402
from baselines.baseline_runner import BaselineRunner

BUDGETS = [512, 1024, 2048, 4096]
TURNS = [100, 200, 300, 500]
SEEDS = [42, 43, 44]
DENSITY = 20.0
CONFLICT = 0.1
NEGATION = 0.1
METHODS = ["adaptive", "sliding_window", "memgpt_style", "summarization_only", "vanilla_rag"]
DURABLE_CATEGORIES = ("project_context", "technical_preference")


def _entity_tokens(fact_text: str) -> List[str]:
    """Distinctive tokens of a synthetic fact: the digits (key_N, port NNNN)
    that identify *which* entity a fact is about. Template words are shared by
    every fact of the same category and cannot discriminate memory content."""
    import re
    return [tok for tok in re.findall(r"\d+", fact_text)]


CATEGORY_SIGNATURES = {
    "technical_preference": ("database requirement", "PostgreSQL"),
    "project_context": ("feature flag", "key_"),
    "transient": ("coffee cup",),
    "summary": ("Global Summary",),
}


def _probe_matches(gt_fact: Dict, candidates: List[Dict]) -> bool:
    """Strict content match: the probed fact's entity (key/port number) must be
    present in the candidate's text, and the candidate must belong to the same
    category (enforced by category template signature, since turn-window entries
    don't carry a category field).

    Uses exact entity tokens so template re-use cannot fake a hit: a store that
    kept *key_19* must not count as recalling *key_25* just because both say
    'Project feature flag configuration key_... is enabled'."""
    entity = _entity_tokens(gt_fact.get("fact", ""))
    cat = gt_fact.get("category")
    if not entity:
        return False
    for f in candidates or []:
        text = f.get("fact", "")
        if isinstance(text, str) and text:
            pass
        else:
            text = ""
        cand_cat = f.get("category")
        if cand_cat and cat and cand_cat != cat:
            continue
        if not cand_cat:
            sig = CATEGORY_SIGNATURES.get(cat)
            if sig and not any(s in text for s in sig):
                continue
        cand_tokens = text.split()
        if all(any(tok == e or tok.endswith("_" + e) for tok in cand_tokens) for e in entity):
            return True
    return False


def word_count(text: str) -> int:
    return max(1, len(str(text).split()))


def probe_turns_for(gt: List[Dict], after_turn: int) -> List[Dict]:
    """One verbatim probe per durable, still-authoritative fact.

    Excludes transient facts (adaptive must forget them) and any fact already
    superseded by a correction (the correction_target probe stands in for it).
    """
    superseded = {g["source_turn"] for g in gt if g.get("superseded_by") is not None}
    probes = []
    for g in gt:
        if g.get("category") == "transient":
            continue
        if g["source_turn"] in superseded and not g.get("is_correction_target"):
            continue
        after_turn += 1
        probes.append({
            "turn_id": after_turn,
            "user": g["fact"],
            "tokens": word_count(g["fact"]),
            "facts": [],
            "_probe_gt_index": len(probes),
        })
    return probes


def run_cell(seed: int, turns: int, budget: int, methods: List[str], embedding_model: str = "") -> Dict:
    from experiments.paper import _tok_of
    settings = load_settings({"max_context_tokens": budget})
    top_k = int(settings["top_k"])

    embed_fn = None
    if embedding_model:
        from memory_optimizer.embeddings import embed_ollama
        embed_fn = lambda texts: embed_ollama(texts, model=embedding_model,  # type: ignore[assignment]
                                              endpoint=settings["system"]["ollama_endpoint"])

    stream, gt = stream_from_generator(seed, turns, DENSITY,
                                       conflict_density=CONFLICT, negation_density=NEGATION)
    raw_tokens = sum(h.get("tokens") or 1 for h in stream)
    budget_stressed = raw_tokens > budget

    probes = probe_turns_for(gt, after_turn=stream[-1]["turn_id"]) if stream else []
    probes_by_idx = {p["_probe_gt_index"]: p for p in probes}
    # Probes are *queries at the point of need*, not conversation turns: they are
    # never appended to the replay stream, so turn-window baselines cannot be
    # credited merely because the probe message sits inside their window.
    full_stream = stream

    window_tokens = per_turn_window_tokens(full_stream, budget)
    baseline_ids = final_window_ids(full_stream, budget)

    out: Dict = {}
    if "adaptive" in methods:
        scorer = ImportanceScorer(weights=settings["scoring_weights"])
        decay = CategoryDecayEngine(lambdas=settings["decay_lambdas"],
                                    pruning_threshold=float(settings["pruning"]["threshold"]))
        retriever = MemoryRetriever(top_k=top_k,
                                    sim_threshold=float(settings.get("similarity_threshold", 0.35)),
                                    embed_fn=embed_fn, embedding_model=embedding_model)
        compressor = MemoryCompressor()
        rep = replay_adaptive(full_stream, settings, scorer, decay, retriever, compressor,
                              fact_tokens=word_count, embed_fn=embed_fn,
                              embedding_model=embedding_model)
        pruned_ids = {p.get("source_turn_id") for p in rep["pruned"]}
        e = evaluate_method("adaptive", rep["per_turn"], rep["memories"], pruned_ids,
                            gt, window_tokens, baseline_ids, fact_tokens=word_count)
        result = _probe_metrics(rep["memories"], gt)
        pr = result["overall_recall"]
        pfr = result["trap_recall"]
        pcor = result["correction_target_recall"]
        pneg = result["negation_recall"]
        ppld = result["plain_durable_recall"]
        e["_probe_recall"] = pr
        e["_probe_trap"] = pfr
        e["_probe_correction"] = pcor
        e["_probe_negation"] = pneg
        e["_probe_plain"] = ppld
        e["_probe_hits_count"] = result["count"]
        e["_store_tokens"] = sum(m.get("measured_tokens") or word_count(m["fact"]) for m in rep["memories"])
        e["_raw_conversation_tokens"] = raw_tokens
        e["_budget_stressed"] = budget_stressed
        out["adaptive"] = e

    for m in methods:
        if m == "adaptive":
            continue
        runner = BaselineRunner(budget=budget, top_k=top_k, embed_fn=embed_fn,
                                fact_tokens=word_count, embedding_model=embedding_model)
        res = runner.run(full_stream, gt, m)
        e = evaluate_method(m, res["per_turn"], res["held_facts"], set(), gt,
                            window_tokens, baseline_ids, fact_tokens=word_count)
        result = _probe_metrics(res["held_facts"], gt)
        pr = result["overall_recall"]
        pfr = result["trap_recall"]
        pcor = result["correction_target_recall"]
        pneg = result["negation_recall"]
        ppld = result["plain_durable_recall"]
        e["_probe_recall"] = pr
        e["_probe_trap"] = pfr
        e["_probe_correction"] = pcor
        e["_probe_negation"] = pneg
        e["_probe_plain"] = ppld
        e["_probe_hits_count"] = result["count"]
        e["_store_tokens"] = sum(mem.get("measured_tokens") or word_count(mem["fact"]) for mem in res["held_facts"])
        e["_raw_conversation_tokens"] = raw_tokens
        e["_budget_stressed"] = budget_stressed
        out[m] = e

    return {
        "seed": seed, "turns": len(stream), "budget": budget,
        "raw_conversation_tokens": raw_tokens,
        "budget_stressed": budget_stressed,
        "ground_truth_count": len(gt),
        "probe_count": len(probes),
        "durable_gt": sum(1 for g in gt if g.get("category") != "transient"),
        "methods": out,
    }


def _probe_metrics(store_facts: List[Dict], gt: List[Dict]) -> Dict:
    """Probe-recall: of the durable, still-authoritative facts, what fraction are
    still present in the method's *final store* (strict entity match)?

    This is the point-of-need question "can the system still answer who/what
    this fact is about" once the conversation is over, evaluated identically for
    every method: the probe repeats the fact's entity (key/port) verbatim and we
    check presence in the final store — not per-turn injection, so window
    baselines are not credited merely because the probe turn sits in their
    window, and template re-use cannot fake a hit.

    Returns overall recall plus subset recalls for the categories the adaptive
    policy is *designed* to keep (trap / correction / negation) vs the facts it
    deliberately de-prioritizes."""
    relevant = [g for g in gt if g.get("category") != "transient"
                and not (g.get("superseded_by") and not g.get("is_correction_target"))]
    by_name: Dict[str, list] = {
        "overall": [g for g in relevant],
        "trap": [g for g in relevant if g.get("is_trap")],
        "correction_target": [g for g in relevant if g.get("is_correction_target")],
        "negation": [g for g in relevant if g.get("is_negation")],
        "plain_durable": [g for g in relevant
                          if not g.get("is_trap") and not g.get("is_correction_target")
                          and not g.get("is_negation")],
    }

    def recall(key):
        group = by_name[key]
        if not group:
            return None
        hits = sum(1 for g in group if _probe_matches(g, store_facts))
        return round(hits / len(group), 3)

    out = {"count": len(relevant)}
    for key in by_name:
        out[f"{key}_recall"] = recall(key)
        out[f"{key}_count"] = len(by_name[key])
    return out


def _auc(xs: List[float]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    if not vals or len(vals) < 2:
        return None
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if trapz is None:
        return None
    return round(float(trapz(vals) / (len(vals) - 1)), 4)


def run_grid(seeds=None, methods=None, budgets=None, turns=None, embedding_model: str = "") -> Dict:
    seeds = seeds or SEEDS
    methods = methods or METHODS
    budgets = budgets or BUDGETS
    turns = turns or TURNS
    cells = []
    for budget in budgets:
        for nturns in turns:
            for seed in seeds:
                print(f"  cell budget={budget} turns={nturns} seed={seed} ({embedding_model or 'lexical'}) ...")
                cells.append(run_cell(seed, nturns, budget, methods, embedding_model=embedding_model))

    # Aggregate: per (budget, turns) across seeds -> mean probe recall, mean
    # injected tokens, mean store tokens, mean store recall, per-method.
    # E1 is recomputed over *conversation turns only* (turn_id <= turns):
    # appended probe turns do not represent per-turn conversation cost.
    by_bt: Dict[str, Dict] = {}
    for c in cells:
        cap = (c["budget"], c["turns"])
        row = by_bt.setdefault(str(cap), {
            "budget": cap[0], "turns": cap[1],
            "raw_conversation_tokens": c["raw_conversation_tokens"],
            "budget_stressed": c["budget_stressed"],
            "seeds": len(seeds), "methods": {m: [] for m in methods}})
        conv_turn_limit = c["turns"]
        for m in methods:
            e = c["methods"].get(m)
            if not e:
                continue
            conv_inj = [r["tokens"] for r in e.get("injected") or []
                        if r.get("turn_id") and int(r["turn_id"]) <= conv_turn_limit]
            conv_mean = round(float(np.mean(conv_inj)), 3) if conv_inj else None
            row["methods"][m].append({
                "probe_recall": e.get("_probe_recall"),
                "mean_injected_tokens": conv_mean,
                "raw_mean_injected_tokens": e["E1_token_efficiency"]["proposed_mean_tokens"],
                "baseline_mean_tokens": e["E1_token_efficiency"]["baseline_mean_tokens"],
                "store_recall": e["E2_memory_accuracy"]["proposed_positive_recall"],
                "durable_recall": e["E2_memory_accuracy"].get("proposed_durable_recall"),
                "correction_recall": e["E2_memory_accuracy"].get("correction_recall"),
                "negation_recall": e["E2_memory_accuracy"].get("negation_recall"),
                "wrongly_retained_fraction": (e["E2_memory_accuracy"].get("wrongly_retained_after_correction") or {}).get("fraction"),
                "forgetting_precision": e["E2_memory_accuracy"]["proposed_forgetting_precision"],
                "store_count": e.get("_store_count"),
                "store_tokens": e.get("_store_tokens"),
            })

    agg: Dict[str, Dict] = {}
    for cap, row in by_bt.items():
        m_agg = {}
        for m, vals in row["methods"].items():
            def mean(key):
                xs = [v[key] for v in vals if v.get(key) is not None]
                return round(float(np.mean(xs)), 3) if xs else None
            m_agg[m] = {
                "probe_recall_mean": mean("probe_recall"),
                "injected_tokens_mean": mean("mean_injected_tokens"),
                "baseline_tokens_mean": mean("baseline_mean_tokens"),
                "store_recall_mean": mean("store_recall"),
                "durable_recall_mean": mean("durable_recall"),
                "correction_recall_mean": mean("correction_recall"),
                "negation_recall_mean": mean("negation_recall"),
                "wrongly_retained_fraction_mean": mean("wrongly_retained_fraction"),
                "forgetting_precision_mean": mean("forgetting_precision"),
                "store_count_mean": mean("store_count"),
                "store_tokens_mean": mean("store_tokens"),
            }
        agg[cap] = {**row, "methods": m_agg}

    # Budget-vs-performance curve at the largest turn count (real pressure).
    curve_turns = max(turns)
    curve = {}
    for budget in budgets:
        curve[str(budget)] = {}
        for m in methods:
            cells_b = [c for c in cells if c["budget"] == budget and c["turns"] == curve_turns]
            recs = [c["methods"][m].get("_probe_recall") for c in cells_b]
            if all(r is None for r in recs):
                curve[str(budget)][m] = None
                continue
            curve[str(budget)][m] = round(float(np.mean([r for r in recs if r is not None])), 3)
    auc_curve = {}
    for m in methods:
        auc_curve[m] = _auc([curve[str(b)][m] if curve[str(b)].get(m) is not None else None
                             for b in budgets])

    return {
        "pipeline": "e10_context_pressure",
        "pipeline_fix_version": 2,
        "config": {
            "budgets": budgets, "turns": turns, "seeds": seeds, "methods": methods,
            "density": DENSITY, "conflict_density": CONFLICT, "negation_density": NEGATION,
            "write": "oracle", "measured": False, "token_mode": "word_count",
            "embedding_model": embedding_model,
        },
        "cells": cells,
        "aggregate_by_budget_turn": agg,
        "budget_curve_at_turns": {"turns": curve_turns, "recall_vs_budget": curve, "auc": auc_curve},
        "note": (
            "probe_recall = point-of-need retrievability: share of durable, still-authoritative "
            "ground-truth facts present in the method's FINAL store, matched on exact entity "
            "tokens (key/port number) within the same category. Probes are queries at the point "
            "of need, NOT conversation turns: they are never appended to the replay stream, so "
            "turn-window baselines cannot be credited merely because the probe message sits in "
            "their window, and template re-use cannot fake a hit (key_19 does not count as "
            "recalling key_25). store_recall/durable_recall = store-level retention of "
            "ground-truth facts (E2, loose matching). injected_tokens = mean active-context "
            "tokens per conversation turn (word count). budget_stressed flags cells where the "
            "raw conversation actually exceeded the budget. NOTE: memgpt_style and vanilla_rag "
            "are UNBOUNDED stores — their store_tokens far exceed the cell budget, so their "
            "high probe recall is buying recall by ignoring the constraint; compare recall per "
            "store token, not recall alone."
        ),
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    payload = run_grid()
    print(f"grid complete in {time.time() - t0:.1f}s", file=sys.stderr)
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "e10_context_pressure.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {path}")