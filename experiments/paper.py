"""Paper experiment engine (task B): deterministic, seeded, batch E1-E6.

Offline replays over a shared pre-extracted fact stream, mirroring the semantics
of experiments.live (same wilcoxon E1, same E6 power, same source-turn-or-overlap
matching) so paper numbers are directly comparable to live dashboard numbers.
The writer is held fixed across methods (brain.md D9): the stream's facts are the
oracle planted facts (`--write oracle`) or real LLM-extracted facts
(`--write extract`); every memory *policy* consumes exactly the same facts.

Honesty labels: tokens are `measured` only when an LLM token measurement function
was supplied; otherwise the manifest says `estimated(word-count)`. Latency (E3)
is only measurable when real LLM calls happen (`--write extract` and
`--measured`), otherwise E3 is reported as not_measured.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from memory_optimizer.budget import token_budget_evict
from memory_optimizer.compression import MemoryCompressor, _is_supersession as fact_is_supersession
from memory_optimizer.decay import CategoryDecayEngine
from memory_optimizer.retrieval import MemoryRetriever
from memory_optimizer.scoring import ImportanceScorer

from baselines.baseline_runner import (
    MATCH_OVERLAP,
    BaselineRunner,
    fact_matches,
)
from memory_optimizer.retrieval import _token_overlap as overlap

FactTokens = Callable[[str], int]


def _sanitize(obj):
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def stream_from_generator(seed: int, turns: int, density: float,
                          conflict_density: float = 0.0,
                          negation_density: float = 0.0) -> tuple:
    from data.synthetic_generator import SyntheticConversationGenerator

    convs, gt = SyntheticConversationGenerator(seed=seed).generate_conversation(
        num_turns=turns, signal_density=density,
        conflict_density=conflict_density, negation_density=negation_density)
    gt_by_turn = defaultdict(list)
    for g in gt:
        gt_by_turn[g["source_turn"]].append(g)
    stream = []
    for t in convs:
        facts = [dict(g, source_turn_id=g["source_turn"]) for g in gt_by_turn.get(t["turn_id"], [])]
        stream.append({
            "turn_id": t["turn_id"],
            "user": t["user"],
            "tokens": max(1, len(t["user"].split())),
            "facts": facts,
        })
    return stream, gt


def extract_stream_facts(stream: List[Dict], extractor) -> tuple:
    """Real LLM extraction pass over the stream (--write extract).

    Returns (stream, extraction_ms_list). Extraction is the shared writer cost
    every memory policy pays identically, so its latency belongs in E3 for every
    method arm."""
    out, latencies = [], []
    for t in stream:
        t0 = time.perf_counter()
        facts, meta = extractor.extract_facts(t["turn_id"], t["user"])
        latencies.append((time.perf_counter() - t0) * 1000)
        for f in facts:
            f["source_turn_id"] = f.get("source_turn_id", t["turn_id"])
        out.append(dict(t, facts=facts, extraction_meta={
            "fallback": bool(meta.get("fallback")),
            "prompt_eval_count": meta.get("prompt_eval_count"),
        }))
    return out, latencies


def per_turn_window_tokens(stream: List[Dict], budget: int) -> List[int]:
    """Running sum of raw history that fits the budget after each turn."""
    result, j, used = [], 0, 0
    for i, h in enumerate(stream):
        used += h.get("tokens") or 1
        while used > budget and j <= i:
            used -= stream[j].get("tokens") or 1
            j += 1
        result.append(used)
    return result


def final_window_ids(stream: List[Dict], budget: int) -> set:
    used, by_id = 0, set()
    for h in stream:
        tok = h.get("tokens") or 1
        if tok > budget:
            by_id = {h["turn_id"]}
            used = tok
            continue
        if used + tok > budget:
            # drop oldest until it fits
            drop = []
            for hh in stream:
                if hh["turn_id"] in by_id:
                    used -= hh.get("tokens") or 1
                    drop.append(hh["turn_id"])
                    if used + tok <= budget:
                        break
            for d in drop:
                by_id.discard(d)
        by_id.add(h["turn_id"])
        used += tok
    return by_id


def replay_adaptive(
    stream: List[Dict],
    settings: Dict,
    scorer: ImportanceScorer,
    decay: CategoryDecayEngine,
    retriever: MemoryRetriever,
    compressor: MemoryCompressor,
    fact_tokens: FactTokens = None,
    embed_fn=None,
    embedding_model: str = "nomic-embed-text",
) -> Dict:
    """Offline replay of the adaptive pipeline over a stream (same stages as
    server._ingest_turn: scoring -> measurement -> dedupe -> decay -> budget ->
    retrieval), returning per-turn injection records plus final store/prune state."""
    memories: List[Dict] = []
    pruned: List[Dict] = []
    per_turn = []
    budget = int(settings["max_context_tokens"])

    retriever.top_k = int(settings["top_k"])
    retriever.sim_threshold = float(settings.get("similarity_threshold", 0.35))

    from memory_optimizer.pipeline import AdaptiveMemoryPipeline
    pipe = AdaptiveMemoryPipeline(settings, scorer, decay, retriever, compressor,
                                  store=memories, pruned=pruned)

    def tok(text: str) -> int:
        if fact_tokens is not None:
            v = fact_tokens(text)
            if v is not None:
                return int(v)
        return max(1, len(str(text).split()))

    for entry in stream:
        t_id = entry["turn_id"]
        result = pipe.ingest(t_id, entry["user"], entry.get("facts", []),
                             fact_tokens=fact_tokens, embed_fn=embed_fn)
        memories = pipe.active_memories
        per_turn.append({
            "turn_id": t_id,
            "injected_tokens": result["injected_tokens"],
            "retrieved": [dict(f) for f in result["injected"]],
        })

    return {"memories": memories, "pruned": pruned, "per_turn": per_turn}


def _wilcoxon_effect(proposed: List[int], baseline: List[int]) -> Dict:
    from experiments.statistics import wilcoxon_paired
    p = wilcoxon_paired(proposed, baseline)
    return {"p_value": p, "significant": bool(p < 0.05) if p is not None else None}


def evaluate_method(
    name: str,
    per_turn_records: List[Dict],
    store_facts: List[Dict],
    pruned_ids: set,
    gt: List[Dict],
    window_tokens: List[int],
    baseline_included_ids: set,
    fact_tokens: FactTokens = None,
    extraction_ms: float = None,
) -> Dict:
    proposed_series = [r["injected_tokens"] for r in per_turn_records]
    baseline_series = list(window_tokens)
    n = min(len(proposed_series), len(baseline_series))
    proposed_series = proposed_series[:n]
    baseline_series = baseline_series[:n]

    proposed_store_recall = (
        sum(1 for g in gt if fact_matches(g, store_facts)) / len(gt) if gt else 0.0
    )
    baseline_recall = (
        sum(1 for g in gt if g["source_turn"] in baseline_included_ids) / len(gt)
        if gt else 0.0
    )

    transient = [g for g in gt if g["category"] == "transient"]
    proposed_forget = (
        sum(1 for g in transient if g["source_turn"] in pruned_ids) / len(transient)
        if transient else 0.0
    )
    baseline_forget = (
        sum(1 for g in transient if g["source_turn"] not in baseline_included_ids) / len(transient)
        if transient else 0.0
    )

    w = _wilcoxon_effect(proposed_series, baseline_series)
    e1 = {
        "proposed_mean_tokens": float(np.mean(proposed_series)) if proposed_series else 0,
        "baseline_mean_tokens": float(np.mean(baseline_series)) if baseline_series else 0,
        "n": n,
        "pairs_equal": len(set(proposed_series)) <= 1 or len(set(baseline_series)) <= 1,
        "wilcoxon": w,
        "p_value": w["p_value"] if not (len(set(proposed_series)) <= 1 or len(set(baseline_series)) <= 1) else None,
        "significant": w["significant"] if not (len(set(proposed_series)) <= 1 or len(set(baseline_series)) <= 1) else None,
    }

    e2 = {
        "proposed_positive_recall": round(proposed_store_recall, 3),
        "baseline_positive_recall": round(baseline_recall, 3),
        "proposed_forgetting_precision": round(proposed_forget, 3),
        "baseline_forgetting_precision": round(baseline_forget, 3),
        "per_category_recall": _per_category_recall(gt, store_facts),
        "trap_recall": _recall_of(gt, store_facts, lambda g: g.get("is_trap")),
        "correction_recall": _recall_of(gt, store_facts, lambda g: g.get("is_correction_target")),
        "negation_recall": _recall_of(gt, store_facts, lambda g: g.get("is_negation")),
        "wrongly_retained_after_correction": _wrongly_retained(gt, store_facts),
    }

    e4 = _needle_by_distance(gt, store_facts, baseline_included_ids)
    e4["hard_case_accuracy_by_distance"] = _needle_by_distance(
        gt, store_facts, baseline_included_ids,
        filt=lambda g: g.get("is_trap") or g.get("is_correction_target"))

    e6 = _power_check(proposed_series, baseline_series)

    e3 = {
        "extraction_ms": extraction_ms,
        "window_replay_ms": None,
        "answer_ms": None,
        "measured_stages": ["extraction"] if extraction_ms is not None else [],
        "note": "latency measurable only when real LLM extraction runs (--write extract)",
    }

    return {
        "E1_token_efficiency": e1,
        "E2_memory_accuracy": e2,
        "E3_latency": e3,
        "E4_needle_in_haystack": e4,
        "E6_power_check": e6,
        "stats": {"proposed_series": proposed_series, "baseline_series": baseline_series,
                  "pairs": n},
    }


def _recall_of(gt: List[Dict], store_facts: List[Dict], pred) -> float:
    rel = [g for g in gt if pred(g)]
    if not rel:
        return None
    return round(sum(1 for g in rel if fact_matches(g, store_facts)) / len(rel), 3)


def _per_category_recall(gt: List[Dict], store_facts: List[Dict]) -> Dict:
    by_cat = {}
    for g in gt:
        by_cat.setdefault(g["category"], []).append(g)
    return {
        cat: {"expected": len(items),
              "recall": round(sum(1 for g in items if fact_matches(g, store_facts)) / len(items), 3)}
        for cat, items in sorted(by_cat.items())
    }


def _wrongly_retained(gt: List[Dict], store_facts: List[Dict]) -> Dict:
    """Stale-fact retention: of the gt facts that were superseded by a
    correction, how many are still treated as authoritative by the store.
    Lower is better.

    A gt is only ``wrongly retained`` when NO correction record exists for it
    (no memory whose ``superseded_prior_fact`` matches it, and no memory whose
    stored fact is itself a supersession of it) AND its old fact still matches
    the store. A correction record proves the supersession was APPLIED
    (``correction recorded correctly``) — even if unrelated same-category
    template facts still overlap the old gt lexically, that is not a stale
    retention of the corrected fact. This keeps the detector able to tell
    ``correction recorded correctly`` apart from ``correction ignored (stale
    fact still authoritative)``.
    """
    superseded = [g for g in gt if g.get("superseded_by") is not None]
    if not superseded:
        return {"superseded_expected": 0, "wrongly_retained": 0, "fraction": None}

    def _is_record(g: Dict, f: Dict) -> bool:
        prior = f.get("superseded_prior_fact")
        if prior and overlap(g.get("fact", ""), prior) >= MATCH_OVERLAP:
            return True
        return fact_is_supersession(g.get("fact", ""), f.get("fact", ""))

    n_wrong = sum(
        1 for g in superseded
        if not any(_is_record(g, f) for f in store_facts or [])
        and any(fact_matches(g, [f]) for f in store_facts or [])
    )
    return {"superseded_expected": len(superseded), "wrongly_retained": n_wrong,
            "fraction": round(n_wrong / len(superseded), 3)}


def _needle_by_distance(gt: List[Dict], store_facts: List[Dict], baseline_ids: set,
                        filt=None) -> Dict:
    if not gt:
        return {"distances": [], "proposed_accuracy": [], "baseline_accuracy": []}
    _gt = [g for g in gt if filt is None or filt(g)]
    current_turn = max(g["source_turn"] for g in _gt) + 1
    distances = [d for d in (10, 25, 50, 100, 200, 500, 1000, 2500, 5000)
                 if d <= current_turn - 5 and any(current_turn - g["source_turn"] >= d for g in _gt)]
    proposed, baseline = [], []
    for d in distances:
        relevant = [g for g in _gt if current_turn - g["source_turn"] >= d]
        if not relevant:
            proposed.append(None)
            baseline.append(None)
            continue
        p = sum(1 for g in relevant if fact_matches(g, store_facts)) / len(relevant)
        b = sum(1 for g in relevant if g["source_turn"] in baseline_ids) / len(relevant)
        proposed.append(round(p, 3))
        baseline.append(round(b, 3))
    return {"distances": distances, "proposed_accuracy": proposed, "baseline_accuracy": baseline}


def _power_check(proposed: List[int], baseline: List[int]) -> Dict:
    from experiments.live import _e6
    return _e6(proposed, baseline)


def run_seed(
    seed: int,
    turns: int,
    density: float,
    settings: Dict,
    methods: List[str],
    write: str = "oracle",
    measured: bool = False,
    model: str = None,
    ollama_endpoint: str = "http://localhost:11434",
    embedding_model: str = "nomic-embed-text",
    extractor=None,
    conflict_density: float = 0.0,
    negation_density: float = 0.0,
) -> Dict:
    stream, gt = stream_from_generator(seed, turns, density,
                                       conflict_density=conflict_density,
                                       negation_density=negation_density)

    embed_fn = None
    if embedding_model:
        from memory_optimizer.embeddings import embed_ollama
        embed_fn = lambda texts: embed_ollama(texts, model=embedding_model, endpoint=ollama_endpoint)

    measurer = TokenMeasurer(model=model, endpoint=ollama_endpoint) if measured and model else None
    fact_tokens = measurer.measure if measurer else None

    if write == "extract":
        if extractor is None:
            from memory_optimizer.extraction import FactExtractor
            extractor = FactExtractor(endpoint=ollama_endpoint, model=model)
        stream, extraction_latencies = extract_stream_facts(stream, extractor)
        extraction_ms = round(float(np.mean(extraction_latencies)), 2) if extraction_latencies else None
    else:
        extraction_ms = None

    budget = int(settings["max_context_tokens"])
    top_k = int(settings["top_k"])
    window_tokens = per_turn_window_tokens(stream, budget)
    baseline_ids = final_window_ids(stream, budget)

    out_methods: Dict[str, Dict] = {}
    adaptive = None
    if "adaptive" in methods:
        scorer = ImportanceScorer(weights=settings["scoring_weights"])
        decay = CategoryDecayEngine(lambdas=settings["decay_lambdas"],
                                    pruning_threshold=float(settings["pruning"]["threshold"]))
        retriever = MemoryRetriever(top_k=top_k,
                                    sim_threshold=float(settings.get("similarity_threshold", 0.35)),
                                    embed_fn=embed_fn, embedding_model=embedding_model)
        compressor = MemoryCompressor()
        replay = replay_adaptive(stream, settings, scorer, decay, retriever, compressor,
                                 fact_tokens=fact_tokens, embed_fn=embed_fn,
                                 embedding_model=embedding_model)
        adaptive = replay
        pruned_ids = {p.get("source_turn_id") for p in replay["pruned"]}
        e = evaluate_method("adaptive", replay["per_turn"], replay["memories"], pruned_ids,
                            gt, window_tokens, baseline_ids, extraction_ms=extraction_ms)
        e["E5_ablations"] = _adaptive_ablations(stream, settings, gt, window_tokens, baseline_ids,
                                                 scorer, fact_tokens, embed_fn, embedding_model)
        out_methods["adaptive"] = e

    for m in methods:
        if m == "adaptive":
            continue
        runner = BaselineRunner(budget=budget, top_k=top_k, embed_fn=embed_fn,
                                fact_tokens=fact_tokens, embedding_model=embedding_model)
        res = runner.run(stream, gt, m)
        pruned_ids = set()
        e = evaluate_method(m, res["per_turn"], res["held_facts"], pruned_ids, gt,
                            window_tokens, baseline_ids, extraction_ms=extraction_ms)
        out_methods[m] = e

    return {
        "seed": seed,
        "turns": len(stream),
        "ground_truth_count": len(gt),
        "methods": out_methods,
    }


def _adaptive_ablations(stream, settings, gt, window_tokens, baseline_ids, scorer,
                        fact_tokens, embed_fn, embedding_model) -> Dict:
    def replay_variant(lambdas=None, weights=None, dedupe=True):
        decay = CategoryDecayEngine(lambdas=lambdas or settings["decay_lambdas"],
                                    pruning_threshold=float(settings["pruning"]["threshold"]))
        scorer_v = ImportanceScorer(weights=weights or settings["scoring_weights"])
        retriever = MemoryRetriever(top_k=int(settings["top_k"]),
                                    sim_threshold=float(settings["similarity_threshold"]),
                                    embed_fn=embed_fn, embedding_model=embedding_model)
        compressor = MemoryCompressor()
        settings_v = dict(settings, enable_compression=dedupe)
        return replay_adaptive(stream, settings_v, scorer_v, decay, retriever, compressor,
                               fact_tokens=fact_tokens, embed_fn=embed_fn,
                               embedding_model=embedding_model)

    full = replay_variant()
    full_mean = float(np.mean([r["injected_tokens"] for r in full["per_turn"]])) or 1.0
    zero_lambdas = {k: 0.0 for k in settings["decay_lambdas"]}
    rows = [
        ("full_system", full),
        ("no_decay", replay_variant(lambdas=zero_lambdas)),
        ("no_compression", replay_variant(dedupe=False)),
        ("equal_weight_scoring", replay_variant(weights={
            "w1_relevance": 0.25, "w2_utility": 0.25, "w3_recency": 0.25, "w4_frequency": 0.25})),
    ]
    e5 = {}
    for label, rep in rows:
        mean_tok = float(np.mean([r["injected_tokens"] for r in rep["per_turn"]])) or 0.0
        accuracy = sum(1 for g in gt if fact_matches(g, rep["memories"])) / len(gt) if gt else 0.0
        e5[label] = {
            "ablation": label,
            "accuracy": round(accuracy, 3),
            "token_overhead_vs_full": round(mean_tok / full_mean, 2),
            "pruned_count": len(rep["pruned"]),
        }
    return e5


class TokenMeasurer:
    """Measures content tokens via Ollama prompt_eval_count (num_predict=1)."""

    def __init__(self, model: str, endpoint: str):
        import requests
        self.model = model
        self.endpoint = endpoint
        self._s = requests.Session()
        self._cache = {}

    def measure(self, text: str) -> Optional[int]:
        from memory_optimizer.extraction import build_extraction_prompt
        if text in self._cache:
            return self._cache[text]
        try:
            r = self._s.post(f"{self.endpoint}/api/generate", json={
                "model": self.model, "prompt": str(text), "stream": False,
                "keep_alive": "30m", "options": {"temperature": 0.1, "num_predict": 1, "num_ctx": 8192},
            }, timeout=60)
            v = r.json().get("prompt_eval_count")
        except Exception:
            v = None
        self._cache[text] = v
        return v


def aggregate(seeds: List[Dict], turns: int, density: float) -> Dict:
    by_method = defaultdict(list)
    for s in seeds:
        for m, e in s["methods"].items():
            by_method[m].append(e)
    agg = {}
    keys = ["E1_token_efficiency", "E2_memory_accuracy", "E4_needle_in_haystack"]
    from experiments import statistics  # noqa: F401
    pools: Dict[str, Dict[str, list]] = defaultdict(lambda: {"pa": [], "pb": []})
    for m, evals in by_method.items():
        e1 = [x["E1_token_efficiency"] for x in evals]
        e2 = [x["E2_memory_accuracy"] for x in evals]
        e4 = [x["E4_needle_in_haystack"] for x in evals]
        e6 = [x["E6_power_check"] for x in evals]
        for x in evals:
            pools[m]["pa"].extend(x["stats"]["proposed_series"])
            pools[m]["pb"].extend(x["stats"]["baseline_series"])
        pa, pb = pools[m]["pa"], pools[m]["pb"]
        agg[m] = {
            "seeds": len(evals),
            "mean_E1_proposed_tokens": round(float(np.mean([x["proposed_mean_tokens"] for x in e1])), 2),
            "mean_E1_baseline_tokens": round(float(np.mean([x["baseline_mean_tokens"] for x in e1])), 2),
            "E1_proposed_pooled_ci95": statistics.bootstrap_mean_ci(pa),
            "E1_diff_vs_baseline_ci95": statistics.bootstrap_ci_mean_diff(pa, pb),
            "cohens_d_paired_vs_baseline": statistics.cohens_d_paired(pa, pb),
            "mean_E2_proposed_recall": round(float(np.mean([x["proposed_positive_recall"] for x in e2])), 3),
            "mean_E2_baseline_recall": round(float(np.mean([x["baseline_positive_recall"] for x in e2])), 3),
            "mean_E2_proposed_forgetting_precision": round(
                float(np.mean([x["proposed_forgetting_precision"] for x in e2])), 3),
            "E6": {k: [x[k] for x in e6 if x.get(k) is not None] for k in ("observed_cohens_d", "required_n_per_group")},
        }

    mc_raw: Dict[str, float] = {}
    for m in by_method:
        pa, pb = pools[m]["pa"], pools[m]["pb"]
        from experiments.statistics import wilcoxon_paired
        p = wilcoxon_paired(pa, pb)
        mc_raw[m] = max(float(p), 1e-9) if p is not None else None
    ms = list(mc_raw)
    order = sorted([m for m in ms if mc_raw[m] is not None], key=lambda m: mc_raw[m])
    raw_ordered = [mc_raw[m] for m in order]
    holm = statistics.holms_correct(raw_ordered)
    bonf = statistics.bonferroni_correct(raw_ordered)
    agg["multiple_comparisons"] = {
        "paired_wilcoxon_vs_baseline": {m: mc_raw[m] for m in ms},
        "holms_adjusted": {m: (holm[order.index(m)] if m in order else None) for m in ms},
        "bonferroni_adjusted": {m: (bonf[order.index(m)] if m in order else None) for m in ms},
        "comparisons": len(order),
        "note": "E1 per-turn proposed-vs-baseline token series, paired Wilcoxon over pooled seeded traces; "
                "family-wise corrected over methods. Repeated turns within a seed are not independent.",
    }
    return agg


def load_settings(overrides: Optional[Dict] = None) -> Dict:
    import yaml
    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
    s = {
        "system": cfg["system"],
        "max_context_tokens": int(cfg["system"]["max_context_tokens"]),
        "top_k": int(cfg["retrieval"]["top_k"]),
        "similarity_threshold": float(cfg["retrieval"]["similarity_threshold"]),
        "pruning": cfg["pruning"],
        "decay_lambdas": cfg["decay_lambdas"],
        "scoring_weights": cfg["scoring_weights"],
        "compression": cfg["compression"],
        "injection_token_limit": int(cfg["system"].get("injection_token_limit", 0)),
    }
    if overrides:
        if overrides.get("max_context_tokens") is not None:
            s["max_context_tokens"] = int(overrides["max_context_tokens"])
        if overrides.get("top_k") is not None:
            s["top_k"] = int(overrides["top_k"])
        if overrides.get("similarity_threshold") is not None:
            s["similarity_threshold"] = float(overrides["similarity_threshold"])
    return s


def run_paper(turns: int, density: float, seeds: List[int], methods: List[str],
              write: str, measured: bool, model: str, ollama_endpoint: str,
              embedding_model: str, label: str = "",
              conflict_density: float = 0.0, negation_density: float = 0.0,
              overrides: Optional[Dict] = None) -> Dict:
    settings = load_settings(overrides)
    seed_results = []
    for seed in seeds:
        print(f"  seed {seed} (write={write}, measured={measured}) ...")
        sr = run_seed(seed, turns, density, settings, methods, write=write, measured=measured,
                      model=model, ollama_endpoint=ollama_endpoint, embedding_model=embedding_model,
                      conflict_density=conflict_density, negation_density=negation_density)
        seed_results.append(sr)
    agg = aggregate(seed_results, turns, density)
    config = {
        "turns": turns, "density": density, "seeds": seeds, "methods": methods,
        "write": write, "measured": measured, "model": model,
        "embedding_model": embedding_model, "label": label,
        "conflict_density": conflict_density, "negation_density": negation_density,
        "max_context_tokens": settings["max_context_tokens"],
        "top_k": settings["top_k"],
        "injection_token_limit": settings["injection_token_limit"],
    }
    token_source = "measured" if measured else "estimated(word-count)"
    return {
        "pipeline": "paper",
        "generated_at": time.time(),
        "config": config,
        "token_source": token_source,
        "metric_source": (
            "offline seeded replay; facts=" + write +
            (" (LLM extracted)" if write == "extract" else " (oracle ground-truth, writer held fixed)") +
            "; E1 token series " + token_source +
            "; E3 latency not measured in offline mode"
        ),
        "results": seed_results,
        "aggregate": agg,
    }


def write_manifest(payload: Dict) -> str:
    base = Path(__file__).resolve().parent / "results"
    base.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha1(json.dumps(payload["config"], sort_keys=True, default=str).encode()).hexdigest()[:8]
    run_id = f"h{sha}"
    path = base / f"manifest_{run_id}.json"
    path.write_text(json.dumps(_sanitize(payload), indent=2, default=str))
    (base / "latest_manifest.json").write_text(json.dumps(_sanitize(payload), indent=2, default=str))
    return str(path), run_id
