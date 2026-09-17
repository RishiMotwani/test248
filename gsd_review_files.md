# gsd_review_files.md — File Index for External Review

This file provides a concise index of important source files, experiment scripts,
result manifests, and generated artifacts. Use this as a starting point for review.

## Important Source Files

### Core Pipeline
- `memory_optimizer/pipeline.py` — AdaptiveMemoryPipeline orchestration (ingest, decay,
  budget eviction, retrieval, latency staging). Stages: scoring→token_measurement→compression→decay→evict→retrieve.
- `memory_optimizer/scoring.py` — ImportanceScorer with weights w1_relevance,w2_utility,w3_recency,w4_frequency; compute_score(memory, query_relevance, current_turn).
- `memory_optimizer/decay.py` — CategoryDecayEngine step_decay_and_prune(memories, current_turn) → (active, pruned) with lambda per category.
- `memory_optimizer/budget.py` — token_budget_evict(memories, budget, fact_tokens) → (kept, evicted, used_tokens).
- `memory_optimizer/retrieval.py` — MemoryRetriever._score_all(query, memories) → ranked list; retrieve(query, memories, current_turn, token_limit, fact_tokens).
- `memory_optimizer/compression.py` — dedupe_incremental(memories, new_facts, embed_fn) — incremental dedupe preserving correction records.

### Experiments
- `experiments/e12_coding_benchmark.py` — E12 coding-context usefulness benchmark: replays `data/coding_workload.py` per method at identical active-context budget; scores query-time injected context by exact required/forbidden token presence; reports context/store/long-range/correction recall, obsolete retention, mean context tokens, recall-per-1k-tokens. Production retrieval path (nomic-embed-text via Ollama).
- `experiments/paper.py` — run_seed(seed, turns, density, settings, methods, write, measured, token_mode, model, embedding_model, conflict_density, negation_density); evaluate_method(name, per_turn, store_facts, pruned_ids, gt, window_tokens, baseline_included_ids, fact_tokens); _needle_by_distance; _wrongly_retained; _power_check; _per_category_recall; _recall_of.
- `experiments/e10_context_pressure.py` — New context-pressure grid: 48 cells (4 budgets × 4 turns × 3 seeds × 5 methods); probe_recall final-store strict entity match; budget_stressed flag; store_tokens; AUC curve.
- `experiments/e11_latency_benchmark.py` — Latency benchmark: extraction ~350ms/turn (writer cost); adaptive write path marginal stages: scoring~0.05ms, compression~0.01ms, decay~0.007ms, retrieval~0.04ms, budget_evict~0.005ms; fact_token_measurement ~12ms only in measured mode (was 42ms anomaly caused by LLM call inside write path).
- `experiments/e0_extraction_quality.py` — Extraction quality experiment (E0).
- `experiments/e7_sensitivity_sweep.py` — Sensitivity sweep (E7).
- `experiments/e8_external_benchmark.py` — External benchmark (E8).
- `experiments/e9_correction_isolation.py` — Correction isolation (E9).
- `experiments/statistics.py` — Statistical utilities (selftest, bootstrap, cohens_d, power analysis).

### Baselines
- `baselines/baseline_runner.py` — BaseBaseline class (observe, held_facts, retrieve, reset, tokens, token_source, entry_tokens, fit_entries).
- `baselines/sliding_window.py` — SlidingWindowBaseline (window_size=10, turn-budget bounded, no retrieval/score).
- `baselines/vanilla_rag.py` — VanillaRAGBaseline (static vector store, no eviction/decay).
- `baselines/memgpt_style.py` — MemGPTStyleBaseline (context compression + retrieval).
- `baselines/summarization_only.py` — SummarizationOnlyBaseline (global summary concatenation).

### Data & Generators
- `data/coding_workload.py` — E12 deterministic coding-session generator: 12 hand-written facts + 24 distinct predicate-family templates, corrections/obsolete facts, filler, ground-truth queries; facts spread across the session; `self_test()`.
- `data/synthetic_generator.py` — SyntheticConversationGenerator (project_context key_N, technical_preference port_5432+N, transient coffee cup number N; conflict/negation/correction handling).
- `data/seed_utils.py` — Seed utilities.

### Evaluation
- `experiments/results/` — Result manifests (all committed to the repo so the reviewer can read the full data; the directory is normally gitignored, these files were force-added):
  - `manifest_h3795b2da.json` — 3-seed post-fix paper validation (E1/E2/E4)
  - `manifest_h892fac01.json` — Quick gate validation (correction_recall≥0.5, wrongly_retained≤0.5)
  - `e10_context_pressure.json` — Full 48-cell context-pressure grid
  - `e11_latency_benchmark.json` — Latency benchmark results
  - `e12_coding_benchmark.json` — **E12 full coding-context grid (Phase 8)**
  - `e0_extraction_quality.json` — E0 results
  - `e7_sweep.json` — E7 sweep results
  - `e8_e8-v1.json` — E8 results
  - `e9_correction_isolation.json` — E9 isolation results
- `experiments/results/latest_manifest.json` — Latest (post-fix) live manifest
- `experiments/results/live_manifest.json` — Live (pre-fix, stale) manifest

### Configuration & Planning
- `.planning/PROJECT.md` — Project reference
- `.planning/STATE.md` — Project state
- `brain.md` — Design decisions D1–D27
- `config.yaml` — Global configuration
- `requirements.txt` — Python dependencies

### Tests
- `tests/test_coding_benchmark.py` — 9 tests for E12: workload determinism, scale growth, ground-truth coverage, corrections, spread-across-session fairness, harness shape/bounds, deterministic cell, budget binds.
- `tests/test_compression_supersession.py` — 10 tests for correction/negation supersession detection
- `tests/test_retrieval_ranking.py` — 6 tests for retrieval ranking contract, corrected-fact priority, reinforcement
- `tests/test_scoring_salience.py` — 8 tests for write-time salience, embedding cosine, pipeline ingest

### Key Output Artifacts
- `experiments/results/e12_coding_benchmark.json` — **E12 full grid** (config + 9 cells with per-turn `injected` arrays + per-budget aggregate + retrieval-ablation context), 332K
- `experiments/results/e10_context_pressure.json` — 48-cell context-pressure grid (primary new experiment)
- `experiments/results/e11_latency_benchmark.json` — Latency benchmark results
- `experiments/results/manifest_h3795b2da.json` — 3-seed post-fix paper validation
- `experiments/results/manifest_h892fac01.json` — Quick gate validation
- `gsd_metrics.json` — Machine-readable experiment metrics (this file)
- `gsd_handoff.md` — Comprehensive review handoff (this file's counterpart)
- `gsd_review_files.md` — This file

## Files Modified in This Phase

- `data/coding_workload.py` — **NEW:** E12 coding-session workload (diverse predicate families, spread schedule, ground truth)
- `experiments/e12_coding_benchmark.py` — **NEW:** E12 coding-context usefulness benchmark
- `tests/test_coding_benchmark.py` — **NEW:** 9 E12 regression tests
- `brain.md` — D28: E12 negative result + retrieval-calibration finding
- `.planning/STATE.md` — E12 results recorded
- `gsd_handoff.md` — Phase 8 (E12) section prepended
- `gsd_metrics.json` — `e12_coding_benchmark` metrics block added
- `experiments/paper.py` — token_mode word_count support, store/injected summaries, durable recall (E2/E4), _tok_of helper
- `experiments/e10_context_pressure.py` — Context-pressure grid (P2); strict entity-match probe recall; final-store matching; budget-stressed flag
- `experiments/e11_latency_benchmark.py` — Latency benchmark (P8); word-count vs measured token modes; _median helper (fix stdlib shadowing)
- `brain.md` — D27: context-pressure probe recall design
- `.planning/STATE.md` — Project state update (P2 complete, key metrics, instrumentation gaps, known issues)
- `gsd_handoff.md` — Comprehensive review handoff
- `gsd_metrics.json` — Machine-readable experiment metrics
- `gsd_review_files.md` — This file index

## Generated Artifacts (run via venv/bin/python)

- `experiments/results/e12_coding_benchmark.json` — E12 full grid output (committed)
- `experiments/results/e10_context_pressure.json` — 48-cell grid output
- `experiments/results/e11_latency_benchmark.json` — Latency benchmark output
- `experiments/results/manifest_h3795b2da.json` — 3-seed validation
- `experiments/results/manifest_h892fac01.json` — Quick gate validation

## Git

- Current branch: master
- Recent commits (Phase 8 → Phase 1):
  - 60de0c5 research: validate coding-context usefulness (E12, negative result)
  - ba56924 Phase 7: context-pressure experiment + latency benchmark + revalidation
  - bf581ba Phase 6: revalidation, docs, and final audit
  - 91fa6b4 test: stage E9 isolation experiment and retrieval ranking regression tests
  - f011d2c feat: quick-gate correction validation, retrieval ranking regression test, E9 correction isolation probe, dashboard caveats
  - 134d5b5 fix(eval): forgetting precision over evicted turns, horizon notes, manifest v2
  - 70866c3 feat: write-time salience replaces hardcoded query_relevance=0.8
  - be75897 feat: correction supersession detection and correctly-hardened retention metrics
  - 0880f1b refactor: remove dead compression methods and deprecated e1-e6 stubs; add pytest scaffolding
- Working tree clean at 60de0c5

## How to reproduce the E12 result (for the reviewing LLM)

```bash
# 1. install deps (venv already in repo)
python -m venv venv && venv/bin/pip install -r requirements.txt

# 2. Ollama must be running with the embedding model used by the production path
ollama pull nomic-embed-text        # default endpoint http://localhost:11434

# 3. workload self-test (no pipeline, no LLM)
venv/bin/python -m data.coding_workload

# 4. full E12 grid -> experiments/results/e12_coding_benchmark.json (~25s)
venv/bin/python -m experiments.e12_coding_benchmark

# 5. test suite (33 tests; E12 tests run lexical so Ollama is not required)
venv/bin/python -m pytest tests/ -q
```

The retrieval ablation (production 0.167 / pure-similarity 0.421 / pure-importance 0.115)
is a read-only diagnostic: replay the adaptive pipeline at budget 512, then rank the
same store with three different scoring functions. No pipeline code is modified.

## Review Checklist

- [ ] Verify E12 workload fairness (facts spread across session; diverse predicate families; natural store 2–8× budget)
- [ ] Verify E12 negative result is from the pipeline, not the workload (vanilla_rag reaches 0.905 on the same facts)
- [ ] Verify the retrieval-calibration claim: importance span 0.47 vs nomic similarity span 0.069; ablation production 0.167 / pure-sim 0.421 / pure-imp 0.115
- [ ] Confirm retrieval weights were intentionally left untuned (directive: do not optimise adaptive to beat the benchmark)
- [ ] Verify e10_context_pressure.py probe recall semantics (final-store strict entity match)
- [ ] Verify e11_latency_benchmark.py latency numbers (word-count modes vs measured LLM modes)
- [ ] Verify paper.py evaluate_method store/injected summaries and durable recall metrics
- [ ] Verify E4 distance-25 diagnosis (metric artifact, not pipeline bug)
- [ ] Verify correction_recall=1.0, wrongly_retained_fraction=0.0 (lexical) / 0.2 (embed edge)
- [ ] Verify budget-stressed flag correctness (raw_tokens > budget)
- [ ] Verify no new memory mechanisms introduced (context efficiency only)
- [ ] Verify dashboard consistency with manifest values
- [ ] Verify all 24 pytest tests pass
- [ ] Verify quick gate pass (correction_recall≥0.5, wrongly_retained≤0.5)
- [ ] Check for stale pre-fix live manifest on dashboard
- [ ] Verify embed-path dependence documented (lexical vs embed differing per seed)
- [ ] Verify default 50-turn config does not stress 4096 budget (raw 616 < 4096)
