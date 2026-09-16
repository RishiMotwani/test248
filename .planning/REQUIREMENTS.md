# Requirements: Adaptive Memory Manager — Audit Fixes

**Defined:** 2026-09-16
**Core Value:** Correction handling must work; write-time salience must be real, not constant

## v1 Requirements

### Correction Handling

- [x] **CORR-01**: dedupe_incremental detects supersessions via revision/negation markers + lexical overlap
- [x] **CORR-02**: Superseded fact text is replaced with corrected text (not merged away)
- [x] **CORR-03**: Superseded prior fact stored in `superseded_prior_fact` field
- [x] **CORR-04**: Confidence set from new fact (not max of old and new)
- [x] **CORR-05**: access_count and last_access_turn reinforced from correction

### Scoring

- [x] **SCOR-01**: pipeline.ingest computes write-time salience from user_message ↔ fact cosine or overlap
- [x] **SCOR-02**: No hardcoded query_relevance=0.8 remains in any Python file
- [x] **SCOR-03**: live.py E5 _e5_replay uses pipeline ingest (not independent scorer.compute_score)

### Evaluation

- [x] **EVAL-01**: live.py forgetting_precision computed over evicted turns (not in-window turns)
- [x] **EVAL-02**: paper.py evaluate_method includes forgetting_horizon_note field
- [x] **EVAL-03**: Manifests carry pipeline_fix_version = 2
- [x] **EVAL-04**: _paper_board exposes scoring_and_correction_fix_applied field

### Dashboard

- [x] **DASH-01**: E2 forgetting precision row shows conditional caveat when horizon is short
- [x] **DASH-02**: Historical manifests (version < 2) show visible banner

### Quality

- [x] **TEST-01**: pytest suite with correction supersession tests
- [x] **TEST-02**: pytest suite with write-time salience tests
- [x] **TEST-03**: Retrieval ranking regression test
- [x] **TEST-04**: quick gate fails if correction_recall is invalid
- [x] **TEST-05**: e9_correction_isolation.py standalone experiment

### Cleanup

- [ ] **CLEAN-01**: compression.py dedupe() and compress_cluster() deleted
- [ ] **CLEAN-02**: Deprecated e1-e6 stubs removed (if confirmed unused)

### Revalidation

- [x] **REVAL-01**: 3-seed experiment run with conflict/negation density
- [x] **REVAL-02**: correction_recall >= 0.5 in post-fix run (1.0)
- [x] **REVAL-03**: wrongly_retained_after_correction.fraction <= 0.5 (0.2)
- [x] **REVAL-04**: Token efficiency regression < 15% relative (7.9% vs vanilla)

## v2 Requirements

(None yet)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Overlap primitive unification (5 implementations) | Documented in concerns; separate redesign phase needed |
| h2h grader negation false-positive fix | Requires semantic matching redesign; out of scope per spec |
| Thread-safety of global state dict | Separate concern; documented but not part of this audit |
| Ollama num_ctx / keep_alive tuning | Documented concern; separate phase |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CORR-01..05 | Phase 2 | Validated |
| SCOR-01..03 | Phase 3 | Validated |
| EVAL-01..04 | Phase 4 | Validated |
| DASH-01..02 | Phase 5 | Implemented |
| TEST-01 | Phase 2 | Implemented |
| TEST-02 | Phase 3 | Implemented |
| TEST-03..05 | Phase 5 | Implemented |
| CLEAN-01..02 | Phase 1 | Done |
| REVAL-01..04 | Phase 6 | Validated |

**Coverage:**
- v1 requirements: 23 total
- Mapped to phases: 23
- Unmapped: 0 ✓

---
*Requirements defined: 2026-09-16*
*Last updated: 2026-09-16 after Phase 6 — all 23 v1 requirements implemented/validated*
