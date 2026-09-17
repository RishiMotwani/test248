# Roadmap: Adaptive Memory Manager — Audit Fixes

**Mode:** standard
**Phases:** 12
**Requirements:** 23 mapped

### Phase 1: Foundation — Dead Code Cleanup + Test Infrastructure
**Goal:** Remove unused code, establish pytest infrastructure, and run existing self-tests to confirm baseline
**Success Criteria:**
1. compression.py `dedupe()` and `compress_cluster()` deleted; grep confirms zero callers
2. Deprecated e1-e6 stubs removed (verified unused by import check)
3. `tests/` directory created with conftest.py (shared fixtures)
4. `pytest` added to requirements-dev.txt
5. `python experiments/statistics.py` and `python baselines/baseline_runner.py` still pass
6. `pytest tests/` passes (initial empty suite)

### Phase 2: Correction Supersession Fix
**Goal:** Make dedupe_incremental detect and correctly handle correction/contradiction facts
**Mode:** mvp
**Success Criteria:**
1. `_is_supersession()` heuristic correctly classifies correction markers
2. dedupe_incremental replaces stale fact text with corrected text on supersession
3. `superseded_prior_fact` stored, confidence from new fact, reinforcement preserved
4. `tests/test_compression_supersession.py` passes with both supersession and true-duplicate cases
5. `python run_experiments.py --quick` shows correction_recall > 0.0 and wrongly_retained fraction < 1.0

### Phase 3: Write-time Salience + Offline Sync
**Goal:** Replace hardcoded query_relevance=0.8 with real write-time salience; unify live/offline scoring
**Mode:** mvp
**Success Criteria:**
1. `_write_time_salience()` helper computes cosine or overlap salience
2. pipeline.ingest uses write-time salience when no explicit override supplied
3. live.py _e5_replay goes through pipeline (no standalone scorer.compute_score with 0.8)
4. `grep -rn "query_relevance=0.8" --include="*.py" .` returns zero matches
5. `tests/test_scoring_salience.py` passes

### Phase 4: Evaluation Fixes + Manifest Versioning
**Goal:** Fix E2 forgetting precision, add horizon note, version manifests
**Mode:** mvp
**Success Criteria:**
1. live.py _e2 forgetting_precision computed over evicted turns
2. paper.py evaluate_method includes forgetting_horizon_note
3. pipeline_fix_version = 2 added to manifests
4. _paper_board exposes scoring_and_correction_fix_applied
5. Live manifest computation still works

### Phase 5: Dashboard + Quick Gate + Regression Tests
**Goal:** Update dashboard UI, add quick gate validation, complete test suite
**Mode:** mvp
**Success Criteria:**
1. E2 dashboard row shows forgetting horizon caveat
2. Historical manifest banner visible when version < 2
3. run_experiments.py --quick fails with non-zero exit if correction_recall invalid
4. Retrieval ranking regression test passes
5. Full test suite passes

### Phase 6: Experiment Revalidation + Documentation + Final Audit
**Goal:** Run full experiments, update docs, perform end-to-end audit
**Mode:** mvp
**Success Criteria:**
1. 3-seed experiment passes: correction_recall >= 0.5, wrongly_retained <= 0.5
2. Token efficiency regression < 15%
3. `python experiments/make_paper_artifacts.py --check` passes
4. README.md updated with actual post-fix results
5. brain.md D5/D14 updated
6. Final end-to-end audit confirms all layers agree

### Phase 7: E12 Coding-Context Usefulness Benchmark
**Goal:** Build a deterministic coding-context benchmark to measure task usefulness at equal active-context budget
**Mode:** mvp
**Success Criteria:**
1. `data/coding_workload.py` generates deterministic coding sessions with ground truth
2. `experiments/e12_coding_benchmark.py` scores query-time injected context by exact token presence
3. Workload includes requirements, architecture, constraints, implementation, bugs, corrections, obsolete, filler
4. Budget genuinely binding (natural store 2–8× budget)
5. Production retrieval path (nomic-embed-text via Ollama)
6. Results show adaptive context_recall 0.15–0.17 vs vanilla_rag 0.905 at same budget
4. Root causes documented: bounded store + decay + importance-dominant retrieval

### Phase 8: E12 Architectural Fixes — Separate Store/Active + Query-First Retrieval
**Goal:** Fix the two structural issues diagnosed in E12: coupled store/active budget + importance-dominant retrieval
**Mode:** mvp
**Success Criteria:**
1. Separate `memory_store_token_budget` (long-term) from `max_context_tokens` (active) + `injection_token_limit`
2. Retrieval defaults: `imp_weight=0.15, sim_weight=0.85, cat_bonus=0.02`; formula `sim*sim + imp*imp + bonus`
3. E12 uses `memory_store_token_budget=max(4096, budget*4)` + `injection_token_limit=budget`
4. Retrieval diagnostics: `store_recall`, `retrieval_loss`, `mean_context_utilization`
5. POST-FIX E12: adaptive ctx_recall 0.71/0.80/0.92 (vs 0.15/0.17/0.17 pre-fix)
6. Retrieval loss decreases 0.075→0.056→0.00; correction_recall 0.5→1.0
4. Store uses independent 4096 budget; active budget fully utilized (utilization ~0.99)
5. All tests pass (40 total)

### Phase 9: E13 Generalization + Causal Ablations
**Goal:** Validate Phase-8 improvements generalize; isolate causal drivers (store separation vs retrieval policy)
**Mode:** mvp
**Success Criteria:**
1. E13 generalization: 5 seeds × 5 budgets × 5 methods with Phase-8 defaults
2. Causal Ablation A: 4 store policies (coupled/matched/separated_4x/unbounded) on same workload
3. Causal Ablation B: 4 retrieval profiles (phase8/legacy/pure_sim/pure_imp) on separated store
4. Embedding vs lexical sensitivity (embeddings vs lexical fallback)
4. Query-family breakdown by qtype (requirements, architecture, constraints, etc.)
5. Strict regression tests: token_limit > top_k, token_limit never exceeded, store independent of active
4. All 40 tests pass
5. **Findings documented:**
   - Store separation necessary: coupled ctx_recall 0.18 vs separated_4x 0.71 at 128 budget
   - Query-first retrieval primary driver: legacy 0.27 vs phase8 0.71 vs pure_sim 0.86 at 128
   - Embeddings outperform lexical: 0.70 vs 0.57 ctx_recall at 128
   - Phase-8 improvements **generalized** across seeds, budgets, query families
   - **Both store separation AND query-first retrieval are causal; retrieval is larger contributor**

### Phase 10: E14 Retention Diagnosis
**Goal:** Causally diagnose why useful facts disappear from the long-term store (pure diagnosis, NO policy change)
**Mode:** mvp
**Success Criteria:**
1. Store-pressure instrumentation (`last_store_pressure`) + `write_time_salience` stamping in pipeline (additive only)
2. 9-state lifecycle classifier (PRESENT/RETRIEVED, decay, store-budget, dedupe, superseded, never-stored, other)
3. `experiments/e14_retention_diagnosis.py` replaying 400 turns × 4 budgets × 5 seeds deterministically
4. Ablation A (no decay), B (no prune threshold), C (score components), D (write-time salience) — experimental toggles only
5. Correction/obsolete safety gate in every run (obsolete_retention = 0 everywhere; correction_recall preserved in ablations)
6. **Finding documented:** decay→prune is the dominant store loss (31/20/11/0 facts at 64/128/256/512); store-budget eviction = 0; budget affects loss only via retrieval-reinforcement feeding decay
7. No production policy changed; all defaults verified unchanged (weights 0.15/0.85/0.02, threshold 0.2)
8. All 54 tests pass (+14 new)

### Phase 11: E15 Retention-Policy Separation — Activation vs Survival
**Goal:** Separate dynamic activation (`current_importance`, retrieval ranking) from long-term survival (`retention_priority`, stable evidence score, store eviction); advance the winning policy via the OBSERVED-based decision rule
**Mode:** mvp
**Success Criteria:**
1. `retention: {mode, eviction_priority}` in config; default advanced to `dual_score` / `retention_priority`; backward-compatible fallback to `hard_threshold`
2. `calculate_retention_priority() = min(1.0, base_score*(1+access_gain*(access_count-1)))`; decay no longer deletes; unknown mode → `ValueError`
3. Primary grid (4 policies × 4 budgets × 5 seeds) + stress grid (scale 27, 1200 turns, natural store ~8272 tok; store 1024/2048/4096): dual_score beats soft at tightest store (ctx 0.368–0.503 vs 0.322–0.484; correction 0.94 vs 0.28)
4. Retrieval path FROZEN during grid (causal isolation returns PARITY)
5. obsolete_retention = 0 under EVERY policy/budget; correction gate never regresses
6. Verdict auto-classified + reviewer-confirmable: separation SUPPORTED, retention-priority eviction SUPPORTED, staleness-tradeoff NOT SUPPORTED
7. All 80 tests pass (+26 new)

### Phase 12: E16 Retention Selectivity Under Hard Pressure
**Goal:** Determine whether the system can selectively retain task-relevant facts under genuine store pressure; harden Phase-11 evaluation (identity-safe metrics + E15 reconciliation); test future-blind `task_affinity` against `random`/`base_score_only` and offline `oracle_future_use` ceiling
**Mode:** mvp
**Success Criteria:**
1. Identity-safe metrics: `fact_id = category:qtype:source_turn:SHA256(text)[:16]`; identity context/store/correction recall + obsolete_retention (token-collision immune)
2. Access separation: `retrieval_access_count` (retriever only) vs `ingest_reinforcement_count` (compressor only); retrieval-Feedback loop diagnosable
3. `TaskStateTracker(window=32)` — future-blind causal `task_affinity`; oracle stamped offline only with no-future-leakage assertions
4. Grid: 5 policies × STORE[256,512,1024,2048] × active[64,128,256] × seeds[42..46] = 300 cells, 1200 turns, scale 27, genuine pressure; no-pressure + window diagnostic cells
5. E15 reconciliation from raw JSON (81 cells): SUPERSEDED_INCORRECTLY = 0 everywhere; residual loss = new corrected fact evicted (now guarded by `protect_corrections`)
6. 8-criteria verdict: **task_affinity REJECTED** (0/8; c1/c6/c7/c8 FAIL; oracle ≈ dual at grid store range); config UNCHANGED (dual_score + retention_priority); no post-result tuning
7. All 117 tests pass (+37 new); results JSON + 18-section report committed to `experiments/results/`
