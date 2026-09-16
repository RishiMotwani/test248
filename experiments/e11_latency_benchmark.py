"""e11 - Live latency benchmark (Phase 7, Part 8).

Measured, real-LLM per-stage latency of the post-fix adaptive pipeline
(server._ingest_turn equivalents) over a small seeded conversation, repeated
several times so the report is not a single-shot artifact.

Methodology
-----------
* Extraction (the shared writer cost every method pays) uses the real Ollama
  model (llama3.1:8b). It dominates the budget and is method-independent
  (brain.md D9): we measure it, then report the *adaptive write path* (scoring,
  token measurement, dedupe/compression, decay, budget evict, retrieval) as the
  marginal cost the adaptive policy adds on top.
* Each repetition ingests the same 10-turn stream end-to-end through the same
  AdaptiveMemoryPipeline used by server._ingest_turn, recording per-stage wall
  time via on_stage.
* Retrieval is measured in both lexical fallback and embedding (nomic-embed-text)
  modes; embedding is the documented active store path.
* Latency *optimization* is explicitly NOT a goal of this phase (see the phase
  brief) — the benchmark exists to (a) verify E3's method can attribute the new
  write-time salience cost, and (b) produce reproducible numbers for the report.

Output: experiments/results/e11_latency_benchmark.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

# NOTE: the stdlib `statistics` module cannot be imported here — the script's own
# directory (experiments/) shadows it with the project's experiments/statistics.py
# when this file is run as `python experiments/e11_latency_benchmark.py`.
def _median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.paper import load_settings  # noqa: E402
from memory_optimizer.compression import MemoryCompressor  # noqa: E402
from memory_optimizer.decay import CategoryDecayEngine  # noqa: E402
from memory_optimizer.extraction import FactExtractor  # noqa: E402
from memory_optimizer.pipeline import AdaptiveMemoryPipeline  # noqa: E402
from memory_optimizer.retrieval import MemoryRetriever  # noqa: E402
from memory_optimizer.scoring import ImportanceScorer  # noqa: E402

N_REPEATS = 5
TURNS = 10


def _build_stream():
    from data.synthetic_generator import SyntheticConversationGenerator
    gen = SyntheticConversationGenerator(seed=42)
    convs, _gt = gen.generate_conversation(num_turns=TURNS, signal_density=0.5)
    return [{"turn_id": t["turn_id"], "user": t["user"]} for t in convs]


def _measure_round(settings: Dict, extractor: FactExtractor, use_embeddings: bool,
                   token_mode: str = "word_count") -> Dict:
    stream = _build_stream()
    embed_fn = None
    if use_embeddings:
        from memory_optimizer.embeddings import embed_ollama
        embed_fn = lambda texts: embed_ollama(texts, model="nomic-embed-text",
                                              endpoint="http://localhost:11434")

    from experiments.paper import TokenMeasurer
    if token_mode == "measured":
        fact_tokens = TokenMeasurer(model="llama3.1:8b",
                                    endpoint="http://localhost:11434").measure
    else:
        fact_tokens = lambda text: max(1, len(str(text).split()))

    scorer = ImportanceScorer(weights=settings["scoring_weights"])
    decay = CategoryDecayEngine(lambdas=settings["decay_lambdas"],
                                pruning_threshold=float(settings["pruning"]["threshold"]))
    retriever = MemoryRetriever(top_k=int(settings["top_k"]),
                                sim_threshold=float(settings["similarity_threshold"]),
                                embed_fn=embed_fn, embedding_model="nomic-embed-text")
    compressor = MemoryCompressor()

    lat: Dict[str, List[float]] = {}
    pipe = AdaptiveMemoryPipeline(settings, scorer, decay, retriever, compressor,
                                  on_stage=lambda op, ms: lat.setdefault(op, []).append(ms))

    extraction_ms = []
    for turn in stream:
        t0 = time.perf_counter()
        facts, meta = extractor.extract_facts(turn["turn_id"], turn["user"])
        extraction_ms.append((time.perf_counter() - t0) * 1000)
        pipe.ingest(turn["turn_id"], turn["user"], facts,
                    fact_tokens=fact_tokens, embed_fn=embed_fn)
    med = {k: sorted(v)[len(v) // 2] for k, v in lat.items()}
    return {
        "extraction": round(float(_median(extraction_ms)), 2),
        "adaptive_write_path": med,
        "token_mode": token_mode,
        "round_seconds": round(float(time.perf_counter() - t0), 2),
    }


def run_benchmark(repeats: int = N_REPEATS, model: str = "llama3.1:8b",
                  enable_embeddings: bool = True) -> Dict:
    settings = load_settings()
    extractor = FactExtractor(model=model, endpoint="http://localhost:11434")
    # warm-up
    extractor.extract_facts(0, "User prefers PostgreSQL for the primary database.")

    results: Dict[str, List[Dict]] = {"lexical_word_count": [], "lexical_measured": [],
                                      "embedding_word_count": [], "embedding_measured": []}
    for _ in range(repeats):
        results["lexical_word_count"].append(
            _measure_round(settings, extractor, use_embeddings=False, token_mode="word_count"))
    if enable_embeddings:
        for _ in range(repeats):
            results["embedding_word_count"].append(
                _measure_round(settings, extractor, use_embeddings=True, token_mode="word_count"))
    # measured-token modes: fewer repeats because each round makes many LLM
    # token-measurement calls per fact (fact_token_measurement stage).
    mrepeats = max(2, repeats // 2)
    for _ in range(mrepeats):
        results["lexical_measured"].append(
            _measure_round(settings, extractor, use_embeddings=False, token_mode="measured"))
    if enable_embeddings:
        for _ in range(mrepeats):
            results["embedding_measured"].append(
                _measure_round(settings, extractor, use_embeddings=True, token_mode="measured"))

    def summarize(rows: List[Dict]) -> Dict:
        def med(key):
            xs = [r[key] for r in rows]
            return round(float(_median(xs)), 2) if xs else None
        stages = set()
        for r in rows:
            stages.update(r["adaptive_write_path"].keys())
        out = {"extraction_ms_median": med("extraction"),
               "round_seconds_median": med("round_seconds"),
               "adaptive_write_path_ms_median": {
                   s: (round(float(_median([r["adaptive_write_path"].get(s) for r in rows
                                               if r["adaptive_write_path"].get(s) is not None])), 3)
                       if any(r["adaptive_write_path"].get(s) is not None for r in rows) else None)
                   for s in sorted(stages)},
               "n": len(rows)}
        return out

    return {
        "experiment": "e11_latency_benchmark",
        "model": model,
        "turns_per_round": TURNS,
        "repeats": repeats,
        "measured_token_repeats": max(2, repeats // 2),
        "config": {
            "max_context_tokens": settings["max_context_tokens"],
            "top_k": settings["top_k"],
            "similarity_threshold": settings["similarity_threshold"],
        },
        "results": results,
        "summary": {k: summarize(v) for k, v in results.items()},
        "note": (
            "Per-stage medians over repeated end-to-end rounds of a 10-turn conversation "
            "using the real Ollama model for extraction. extraction = shared writer cost "
            "(method-independent, brain.md D9); adaptive_write_path = the marginal per-turn "
            "cost scored->measure->dedupe->decay->evict->retrieve the adaptive policy adds. "
            "word_count modes measure that path with in-process token estimation (no LLM inside "
            "the write path); measured modes additionally pay a real LLM token-measurement call "
            "per fact (the fact_token_measurement stage). Latency optimization is out of scope "
            "for this phase; the benchmark exists to verify E3 attribution and give reproducible "
            "numbers."
        ),
    }


if __name__ == "__main__":
    payload = run_benchmark()
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "e11_latency_benchmark.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {path}")
    print(json.dumps(payload["summary"], indent=2))