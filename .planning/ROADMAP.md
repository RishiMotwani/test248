# Roadmap: Adaptive Memory Manager — Audit Fixes

**Mode:** standard
**Phases:** 6
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
