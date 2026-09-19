# Roadmap: Adaptive Memory Manager — Audit Fixes

**Mode:** standard
**Phases:** 21
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

### Phase 13: E17 Long-Horizon Coding Capability
**Goal:** Determine whether the adaptive memory system helps a real coding model complete long-running repository-editing tasks (hidden-test pass) better than simpler historical-context managers when the historical context is capped at a fixed budget. Outcome = produced patch passing deterministic hidden tests; recall is diagnostic only; do NOT tune the adaptive algorithm
**Mode:** mvp
**Success Criteria:**
1. `data/coding_task_suite.py` + 4 real tasks under `data/coding_tasks/*` (~600-turn transcripts; buried constraints; deterministic hidden tests; gold patches verified to pass; no gold/test leakage asserted per run)
2. Shared harness `experiments/coding_benchmark.py`: one path for all arms; complete-file edit blocks applied verbatim (diff fallback); 2 attempts max; shared word-count tokenizer
3. Methods (history-only difference): `raw_clipped`, `sliding_window`, `llm_summarization` (same-model running summary), `vanilla_rag` (static cosine), `adaptive` (production dual_score + protect_corrections); diagnostics `no_history` / `full_context` / `direct_history` (oracle)
4. Fixed budgets 256/512/1024; adaptive `injection_token_limit=budget`, `memory_store_token_budget=max(4096, budget*4)`
5. All 9 gates pass on pilot AND full grid: budget pressure, workspace independence, methods differ, real summarization, adaptive production path, no leakage, history dependence (oracle-based), test determinism, patch determinism
6. Full grid 180 cells (4 tasks × 3 seeds × 3 budgets × 5 methods): success rates flat across arms (0.72–0.75); every adaptive-vs-baseline paired 95% CI lower bound = 0 → **adaptive_advances NOT SUPPORTED**; honest verdict written, no production change (`config.yaml` untouched)
7. All 134 tests pass (+17 new); `experiments/results/e17_coding_capability.json` + 24-section `_report.md` committed (force-added)

### Phase 14: E18 Correction-Safe Memory, Targeted Validation
**Goal:** Fix the correction/supersession failure E17 surfaced at the memory-consolidation boundary (correction with new semantic wording was stored as an *additional* memory because the supersession check ran only inside `sim >= 0.90`), add regression tests, validate with an offline identity gate + targeted coding grid, report honestly.
**Mode:** mvp
**Success Criteria:**
1. `dedupe_incremental` resolves an explicit supersession (`_is_supersession`) BEFORE generic semantic-deduplication thresholds; `_replace_with_supersession` shared helper used by `supersedes_turn` and the generic loop; `_is_supersession()` semantics unchanged (no weakening); duplicate-merge gain line preserved (existing correction-safety tests keep passing)
2. 138 tests pass (+2 compression supersession, +1 e2e retrieval authoritativeness, +1 E17 hidden-test strictness); `data/coding_tasks/cache_readonly/hidden/test_hidden.py` rewritten to a spy on `CacheCoordinator.set` (delegates to the original so the gold patch still passes) asserting `calls == [("beta","2")]`
3. `experiments/e18_correction_safety.py`: 36-cell offline identity gate (4 tasks × 3 seeds × 3 budgets) ALL correction-bearing cells PASS (`correction_recall 1.0`, `obsolete_exposure 0.0`)
4. 72-run targeted coding grid (`user_ids` + `validation_pure`, llama3.1:8b): adaptive 17/18 (94.4%, correction_recall 1.0) vs vanilla_rag 9/18 (100% OBSOLETE_INFORMATION_USED) / llm_summarization 9/18 (MEMORY_MISS) / raw_clipped 8/18 (RETRIEVAL_MISS); verdict rule honest
5. `config.yaml` untouched; E17 JSON/report untouched; E18 artifacts committed at `experiments/results/e18_correction_safety.json` + `_report.md` (force-added)
- verdict adaptive_advances: False — the pilot verdict rule is honest (gate B + gate C history dependence work; but the broader-set advantage is NOT established at this model tier)
- config.yaml UNTOUCHED; E17/E18 artifacts immutable; report is pilot-only, reviewer-confirmable; tests 143 passing (+9)

### Phase 15: E19 Coding Generalization, Honest Pilot (PILOT ONLY; gate C failed)
**Goal:** Test whether the corrected (D36/D37/D39) adaptive memory generalizes beyond the two E18 tasks to a broader set of genuinely history-dependent long-horizon coding tasks (new fixtures for new tasks; fixed historical-context budgets; one shared harness), under an honest deterministic-hidden-test pilot; only if all gates pass, run a full grid.
**Mode:** mvp
**Success Criteria:**
1. Task suite: 4 new fixtures (calibrated through multiple restate/delete designs this phase), ≥4 topic families, ≥2 critical facts, ≥1 constraint, ≥1 fact >300 turns old, ≥1 distractor thread, identity-safe fact IDs, no marker language / no hidden-test leakage; 3 history-gated primary tasks + 2 negative controls
2. Shared harness `experiments/e19_coding_generalization.py`: one arm path; complete-file edit blocks applied verbatim; gold/gold-identity checks; diagnostic draws for reproducibility on the LLM path
3. Gates A–J with multiple diagnostic draws (DIAG_DRAWS=3): no_history must fail for every primary (gate C); adaptive-vs-baseline paired CI condition
4. Pilot grid 60 cells (3 primary × 2 seeds × 2 budgets × 5 methods), then full grid 225 — llama3.1:8b
5. Honest verdict + 24-section report at `experiments/results/e19_coding_generalization.json` + `_report.md`

**Outcome — HONEST, PILOT-ONLY (gate C failed):**
- Pilot grid run to completion (60 grid + 27 offline + 12 diagnostics); **gate C (no_history must fail for every primary) FAILED** over honest 3-draw measurement at llama3.1:8b: `user_ids` no_history 2/3, `transaction_atomicity` 1/3; only `validation_pure` cleanly gated (0/3). `config_contract` EXCLUDED as fixture-only (oracle probes 0/3 — task nuance beyond llama3.1:8b), retained as a calibration finding.
- Paired adaptive-vs-baseline 95% CI: only validation_pure adaptive 4/4 clean; broader-set advantage NOT established at this model tier → verdict **adaptive_advances = False** (gate A/B/C history-dependence honesty works, but the broader-set advantage is NOT established)
- Full grid NOT run; report is pilot-only, reviewer-confirmable; `config.yaml` untouched; E17/E18 artifacts immutable; all 143 tests pass (+9)
- verdict adaptive_advances: False — the pilot verdict rule is honest (gates all pass except gate C); config UNTOUCHED; report is pilot-only, reviewer-confirmable, tests 143 passing (+9)

### Phase 16: E20 Counterfactual-History Benchmark Calibration (INVALID at llama3.1:8b)
**Goal:** resolve E19 gate C's open question (is the benchmark genuinely history dependent?) with counterfactual history pairs — per group, byte-identical workspace+prompt, different history → different hidden test + gold patch — and gate each group on whether the correct variant's history (direct_history oracle) lifts solvability while no_history does not. NO adaptive-memory change; E19 full grid never auto-run.
**Mode:** mvp
**Success Criteria:**
1. `data/counterfactual_task_suite.py` + `data/counterfactual_tasks/{identifier_policy,retry_policy,serialization_policy}/`: 3 groups × 2 variants, workspace+prompt byte-identical across A/B; 600-turn histories committed (decision >300 turns old, byte-reproducible); variant-specific hidden tests + gold.patch; hashes + `build_variant()` + self-test
2. `CodingTask.gold_patch_path` added; `e19.check_gold` honors it with fallback (no behavior change elsewhere)
3. `experiments/e20_counterfactual_history.py`: 54 cells (3×2×3×3) @ budget 1024; no_history / direct_history (oracle) / full_context (overflow diagnostic); Hard Gate 1 (offline gold-patch validity on all 6 variants) → STOP if fails; pair-integrity gates → STOP if fails; Hard Gate 2 per group (/6: dh≥5, nh≤3, sep≥2) → verdict history_dependence_benchmark VALID/INVALID + e19_full_grid_eligible; 7-section report; resumable JSON
4. Non-trivial fixtures: gold.patch applies cleanly AND base (unmodified workspace) fails hidden tests for every variant; no history/prompt/test leakage
5. **Outcome — HONEST INVALID:** Gate 1 PASS (all 6 gold patches valid); pair-integrity PASS; Gate 2 fails on `identifier_policy` (dh 6/6 vs nh 6/6, sep 0 — solvable from the workspace alone at llama3.1:8b); `retry_policy` PASS (6/6 vs 3/6, sep 3); `serialization_policy` PASS (5/6 vs 0/6, sep 5) → **history_dependence_benchmark = INVALID, e19_full_grid_eligible = FALSE**; identifier_policy-family fixtures (and E19's `user_ids`) must be redesigned so the required behavior is underivable from workspace alone
6. All 174 tests pass (+31 in `tests/test_counterfactual_history.py`); E19 report presentation-only fix (sections 14/15/18 derive from cfg mode/budgets); E20 JSON + report + fixtures force-committed

### Phase 17: E20 Counterfactual Fixture Repair and Revalidation
**Goal:** fix the single failing E20 counterfactual family (identifier_policy — solvable from the workspace alone) with a genuinely history-dependent replacement (routing_policy), harden the offline fixture-validity gate, and re-run the E20 calibration as a repair run. No adaptive-memory change; original E20 artifacts immutable.
**Mode:** mvp
**Success Criteria:**
1. `data/counterfactual_task_suite.py`: `identifier_policy` replaced by `routing_policy` (opaque op→lane routing; mapping present ONLY in variant hidden tests / gold patches / 600-turn histories; workspace + prompt byte-identical across A/B); group list exactly `["routing_policy","retry_policy","serialization_policy"]`; `data/counterfactual_tasks/identifier_policy/` deleted, `routing_policy/` materialized via `python -m data.counterfactual_task_suite --force` with retry/serialization byte-reproducible
2. `experiments/e20_counterfactual_history.py`: `compute_integrity_gates()` uses real per-variant `workspace_sha`/`prompt_sha` A==B comparisons (no hardcoded True); `run_gold_checks()` adds strict offline `base_hidden_test_pass` (unmodified workspace must FAIL hidden tests) + gold-applies-and-passes, required by `gold_gate_passed()` and reported in a base-failing column
3. `experiments/e21_counterfactual_history_repair.py` (new): thin wrapper over `e20.main` swapping `OUT_JSON`/`OUT_REPORT` to `e20_counterfactual_history_repair.{json,_report.md}` — no duplicated benchmark logic
4. All 177 tests pass (+3: unpatched-workspace-fails-hidden-tests across all 6 variants; routing workspace contains all ops+lanes with no op→lane assignment pair; routing pairs differ only by history/hidden/gold contract)
5. **Outcome — VALID:** offline Gate 1 PASS (all 6 variants: base fails, gold passes); pair-integrity PASS (real hashes); Hard Gate 2 ALL PASS — routing_policy dh 6/6 v nh 0/6 (sep 6), retry_policy dh 5/6 v nh 3/6 (sep 5), serialization dh 6/6 v nh 1/6 (sep 6) → **history_dependence_benchmark = VALID, e19_full_grid_eligible = TRUE** (E19 full grid NOT auto-run — reviewer's decision)
6. Original `e20_counterfactual_history.json` + `_report.md` byte-identical (sha256 re-verified); repair artifacts `e20_counterfactual_history_repair.json` + `_repair_report.md` force-committed; `config.yaml` untouched; commit `research: repair E20 counterfactual fixture and revalidate`

### Phase 18: E19 aligned to the E20-validated benchmark + full grid
**Goal:** E19's primary tasks were re-anchored onto the three E20-validated counterfactual families (routing_policy, retry_policy, serialization_policy — genuinely history-dependent per D39, fixed Variant A), with history dependence certified by the E20 calibration instead of stochastic no-history draws, then the 225-cell full grid was run and reported honestly.
**Mode:** standard
**Success Criteria:**
1. `experiments/e19_coding_generalization.py`: PRIMARY_TASKS = the three E20 families; `E19_COUNTERFACTUAL_VARIANTS` maps each to Variant A; `build_e19_task()` builds the counterfactual variant (fallback `build_task`) and normalises `task_id` to the family name; `run_one`/`check_gold`/`run_offline_cell` use `build_e19_task`; Gate C = `check_e20_calibration()` (frozen E20 repair certification: revalidation, groups match, validation/pair-integrity/history-dependence all pass, verdict VALID + eligible); `persist()` stores `e20_calibration`
2. Report alignment: section 3 lists the three families; section 5 describes the E20-validated history-dependence design; section 12 → "## 12. Experiment Gates A-J" with Gate C explained as "E20 counterfactual history calibration"; section 13 title dynamic (Pilot vs Full-Grid); task-suite loop uses `build_e19_task`; per-task prose for the three families; diagnostics section 20 descriptive only
3. `experiments/e22_e19_full_grid.py` (new): thin runner swapping `e19.OUT_JSON`/`OUT_REPORT` to `e19_coding_generalization_full.{json,_report.md}`, delegating to `e19.main(argv)`, restoring paths in `finally` — no duplicated benchmark logic
4. `tests/test_e19_full_grid_alignment.py` (new, 8 offline tests, no Ollama): exact PRIMARY_TASKS; Variant-A mapping; `check_e20_calibration()["passed"] is True` and groups == PRIMARY_TASKS; `build_e19_task` counterfactual metadata (variant A, group == task_id), gold/hidden exist, 600-turn history, task_id normalised; negative controls not counterfactual; full-grid paths distinct from pilot; 225 dimensions (135 + 90, seeds [1,2,3], budgets [256,512,1024], methods); Gate C is the E20 certification
5. Full grid: 225/225 records (no user_ids/validation_pure/transaction_atomicity as primary records); 9/10 gates pass; offline correction_identity (Gate B) FAILS in all 27 correction-bearing cells (counterfactual correction facts don't restate the prior text with full word coverage under the locked `_is_supersession` heuristic) → **gates_all_passed = False → adaptive_advances = False** (locked verdict logic; no gate weakening, no tuning); E20-certified Gate C passes
6. Original E19/E20 artifacts + `config.yaml`/`memory_optimizer/`/`server.py` untouched; new artifacts `e19_coding_generalization_full.json` + `_full_report.md` force-committed; all 185 tests pass; commit `research: align E19 with validated history benchmark and run full grid`

### Phase 19: Counterfactual Correction Metadata Repair and E19 Revalidation
**Goal:** Restore the existing oracle-pre-extracted correction metadata in the counterfactual history adapter so the existing production supersession path receives the intended `supersedes_turn` relationship, then rerun E19 from fresh state.
**Mode:** standard
**Success Criteria:**
1. `data/counterfactual_task_suite.py`: `build_variant()` enriches `history[*]["facts"]` with explicit correction metadata (`supersedes_turn`, `is_correction_target`, `is_current_correction`, `superseded_fact`, `superseded_prior_fact_id` on current fact; `superseded_by` on obsolete fact) without changing history text, workspace, prompt, hidden tests, or gold patches
2. `experiments/e23_e19_correction_identity_repair.py` (new): thin runner with `--preflight` (27-cell offline correction/embedding gate) and `--full` (fresh 225-cell grid), delegating to `e19.main()`, no duplicated benchmark logic
3. `experiments/e24_missing_negative_control.py` (new): narrow closure utility (`--identify` prints missing cell, `--run` reruns it) for the one timed-out negative-control cell
4. `tests/test_phase19_closure.py` (new, 10 offline tests): grid dimensions, missing cell identity, recovery artifact schema, historical artifact immutability, primary grid completeness, GSD handoff/checkpoint verification
5. Full grid: 225/225 cells (135 primary + 90 negative); 10/10 gates PASS; offline correction_identity = 27/27 PASS (explicit `supersedes_turn` enables production supersession); E20-certified Gate C PASS; **gates_all_passed = True → adaptive_advances = False** (adaptive ties vanilla_rag at 27/27; locked verdict logic; no gate weakening, no tuning)
6. Original E19/E20 artifacts + `config.yaml`/`memory_optimizer/`/`server.py` untouched; new artifacts `e19_coding_generalization_full_repaired.json` + `_full_report.md` + `e24_missing_negative_control.json` force-committed; all 189 tests pass; commit `research: close phase 19 experiment accounting`

### Phase 20: E25 E19 Baseline Ceiling and Budget-Geometry Audit
**Goal:** Run a deterministic, offline audit of the locked E19 result artifact that separates *why* `adaptive_advances = false` (budget pressure not binding the recall methods vs. contamination/saturation) and sets input requirements for any next benchmark. No LLM, no embedding calls, no production memory code, no winner, no ranking change.
**Mode:** standard
**Success Criteria:**
1. `experiments/e25_e19_baseline_ceiling_audit.py` (new): pure artifact analysis of `experiments/results/e19_coding_generalization_full_repaired.json` + in-repo counterfactual fixtures; validates the primary grid is exactly 135 unique, complete cells with non-null `context_sha` (STOP/hard error, write nothing, if not); budget geometry (tokens/utilization/headroom/binding per method x budget, 0.90 binding threshold diagnostic), per-track budget elasticity (SHA-based, never token-count), correction-state classification of every cell (`(corr_recall, obsolete_exposure)` exact mapping, unknown pair → hard error), correction-pair verification from fixtures + offline gate cells, task x budget success table, context-hash analysis (adaptive vs vanilla equality + cross-budget SHA stability), 4 diagnostic flags, next-benchmark requirement constraints, conclusion; sems deterministic with only `pathlib`/`json`/`collections.defaultdict`/`statistics.{mean,median}`
2. Outputs `experiments/results/e25_e19_baseline_ceiling_audit.json` + `_report.md` (new, force-added); E19/E20/E24 artifacts byte-identical after the run
3. `tests/test_e25_baseline_ceiling_audit.py` (new, offline-only): never imports `memory_optimizer`/`OllamaCoder`, no requests/ollama, E19 artifact never written; 135-cell grid validation incl. rejection paths; classification mapping + unknown-pair error; SHA-based stability (adaptive/vanilla 9/9 stable, raw/llm budget-sensitive); binding uses 0.90 threshold; correction-state distribution matches locked E19 (adaptive 27x CLEAN_CURRENT, vanilla 27x OBSOLETE_ONLY); the single vanilla failure = serialization_policy/seed2/1024/OBSOLETE_INFORMATION_USED; flags all boolean and all four True; report sections/tables rendered; JSON top-level keys present
4. All flags found True (observations, not claims to fix E19): vanilla OBSOLETE_ONLY success 0.963 >= 0.80; adaptive/vanilla mean utilization @256 (0.359/0.427) < 0.50 with 9/9 SHA-stable tracks each; vanilla primary success 0.963 >= 0.90; `adaptive_advances` false → next benchmark must make obsolete-only evidence unsafe, bind both recall methods at the low end (observed thresholds ~84-99 adaptive, ~102-116 vanilla tokens), add discrimination above the vanilla ceiling, and must not read E19 as adaptive superiority
5. Reports exactly the observed threshold range and does NOT choose final numerical budgets; declares no winner, changes no ranking, runs zero LLM/embedding calls, imports no production pipeline
6. GSD docs updated (STATE.md Phase 21 block + D43 in Notes + known issue #23, ROADMAP Phase 21, gsd_handoff self-contained with locked result + remaining questions, brain.md decision D43 after D42); `.planning/checkpoints/phase-21-e26-discriminative-benchmark-complete.md` created (phase stopped at the failed pilot per the hard-stop rule); original E19/E20/E24/E25 + `memory_optimizer/**` + `baselines/**` + `config.yaml` + `server.py` untouched; all tests pass (279 via `python -m pytest tests/ -q`); commits `research: add E26 discriminative task suite fixtures` then `research: run E26 discriminative pilot and stop at failed gates`

### Phase 21: E26 Discriminative Adaptive-vs-Vanilla RAG Benchmark (pilot gates failed; full grid not legal)
**Goal:** Build the first benchmark that answers D42's question — can adaptive memory be *discriminated* from vanilla RAG above the E19 ceiling — by making obsolete-only evidence unsafe (contradiction-sensitive hidden tests), binding both recall methods below the E25-observed thresholds (budgets 64/96/128), and predeclaring a paired-CI verdict (`adaptive_beats_vanilla`), with a 48-cell pilot whose five gates gate the 54-cell full grid. Any pilot gate failure is a hard stop.
**Mode:** standard
**Success Criteria:**
1. `data/discriminative_coding_suite.py` + `data/discriminative_coding_tasks/{release_adapter,invoice_adapter,message_adapter}/`: 3 groups × 2 variants (A/B behaviors differ only by history contract); workspace+prompt byte-identical across variants; 4 current correction facts (turns 420–540, explicit `supersedes_turn` metadata), 4 obsolete policy facts (turns 70–180, >300 turns old), 4 distractors; every fact 18–28 shared-word tokens; probe strings are substrings of fact text; offline fixture gate on every variant (base workspace FAILS hidden, `gold.patch` PASSES, `obsolete.patch` FAILS, StaticCoder, 0 LLM calls)
2. `experiments/e26_discriminative_coding_benchmark.py`: run_one (per-cell record: context ids, correction states, budget ratio, context_sha, leakage), gold_check, five pilot gates (history_dependence per group dh≥3/4 + nh≤1/4 + diff≥2; contradiction_state adaptive corr_recall≥0.75 + obs≤0.25 vs vanilla obs≥0.75 pooled over 12 cells per method; budget_binding ≥0.75×96; context_difference ≥80% of the 12 paired cells differ; no_leakage 0), grid completeness, paired bootstrap-CI comparison (27 pairs), verdict `adaptive_beats_vanilla = pilot AND full AND fixtures AND ci_lower_bound > 0`, pilot/full report generators; full mode is hard-gated on a recorded all-gates-pass pilot artifact
3. Offline tests (62 new: `tests/test_discriminative_coding_suite.py` + `tests/test_e26_discriminative_coding_benchmark.py`) — suite determinism/placement/metadata/variant-identity/forbidden-terms/as_coding_task mapping/offline gates, runner grid geometry/gate logic/pairing/verdict/gold checks/render, no requests, no writes to immutables; all hermetic (no Ollama/network)
4. **Pilot (48 cells, llama3.1:8b @ 96, seeds 1–2): budget/context/no_leakage gates PASS; history_dependence FAIL (release_adapter direct_history 0.0/4, message_adapter 0.5/4, invoice_adapter 0.75/4 vs no_history 0.0 everywhere) and contradiction_state FAIL (adaptive corr_recall 0.9375/obs 0.0, vanilla_rag obs 0.2917 < 0.75)** → hard stop; 54-cell full grid NOT run; `adaptive_beats_vanilla = False`; no retuning, no gate weakening
5. `experiments/results/e26_discriminative_coding_benchmark_pilot.json` + `_pilot_report.md` force-committed; `memory_optimizer/`, `baselines/`, `config.yaml`, `server.py`, E19/E20/E24/E25 artifacts untouched; commits `research: add E26 discriminative task suite fixtures` and `research: run E26 discriminative pilot and stop at failed gates`

### Phase 22: E27 Counterfactual Calibration Repair (single-policy calibration; pilot gates failed; head-to-head not eligible)
**Goal:** Repair the failed E26 calibration surface (Phase 21) by reducing every task to exactly ONE decisive historical policy fact vs ONE current correction with four distractors per variant at a single 96-token budget, so the two sides of the conflict never both fit and the obsolete-only contradiction signal is small enough for llama3.1:8b to carry. E27 calibrates a future head-to-head surface; it does NOT run a head-to-head grid, declares no winner, and claims no adaptive-vs-RAG result.
**Mode:** standard
**Success Criteria:**
1. `data/e27_calibration_suite.py` + `data/e27_calibration_tasks/{route_contract,serialization_contract,retry_contract}/`: 3 groups × 2 variants; workspace+prompt byte-identical across variants; exactly 1 obsolete policy fact (turns 80–150, >300 turns old, 50–56 shared-word tokens), 1 current correction (turns 470–530, explicit `supersedes_turn`, 50–56 tokens), 4 distractors (18–24 tokens); obsolete+current ratio >96 (all pairs 103–110) so the conflict never fits; forbidden contract terms absent from workspace/prompt; offline fixture gate on every variant (base workspace FAILS hidden, gold.patch PASSES, obsolete.patch FAILS, StaticCoder, 0 LLM calls)
2. `experiments/e27_calibration.py`: 48-cell pilot only (3×2×2×4, seeds 1–2, budget 96; no `--full`); run_one records (context ids, correction states, budget ratio, context_sha, leakage); five locked gates G1–G5 (history_dependence per group direct≥0.75 + no_history≤0.25 + diff≥2; contradiction_state adaptive corr_recall≥0.90 + obs≤0.10 vs vanilla obs≥0.75; budget_binding ≥0.75×96; context_difference ≥80% of the 12 paired cells differ; no_leakage 0); verdict `calibration_valid = grid_complete AND fixture_gates AND pilot_gates`, `full_head_to_head_eligible` key; 15-section report with Tables A–D; `--validate`/`--pilot`/`--resume`/`--report-only`; results `experiments/results/e27_calibration.json` + `_report.md`
3. Offline tests (+74 new: `tests/test_e27_calibration_suite.py` (13) + `tests/test_e27_calibration.py` (61)) — suite determinism/geometry/placement/metadata/variant-identity/forbidden-terms/oracle-patch-validity/offline gates, runner grid geometry/gate thresholds/verdict/report/gold checks, no requests, no writes to immutables; all hermetic (no Ollama/network)
4. **Pilot (48 cells, llama3.1:8b @ 96, seeds 1–2): budget_binding (both methods 1.0 bound), context_difference (12/12 differ), no_leakage (0) PASS; history_dependence FAIL on all 3 groups (route_contract and serialization_contract direct 1.0 vs no_history 0.75 — cap is ≤0.25 — so diff collapses to 1; retry_contract direct 0.0 vs no_history 0.0) and contradiction_state FAIL (adaptive corr_recall 1.0 / obs_exposure 0.0 clean side, vanilla_rag obs_exposure 0.6667 < 0.75)** → **calibration_valid = False, full_head_to_head_eligible = False**; per-method pilot success adaptive 0.67 / direct_history 0.67 / no_history 0.50 / vanilla_rag 0.17; hard stop, no tuning, no gate weakening, no winner, no head-to-head
5. `experiments/results/e27_calibration.json` + `_calibration_report.md` force-committed; `memory_optimizer/`, `baselines/`, `config.yaml`, `server.py`, E26 suite/runner/results and all E19/E20/E24/E25 artifacts untouched; commits `research: add E27 calibration task suite fixtures` and `research: run E27 calibration pilot and stop at failed gates`

### Phase 23: E28 Model-Tier Certification (frozen E27 surface re-certified at qwen2.5:7b; still invalid — surface-level explanation)
**Goal:** answer D44's open question whether the E27 calibration failure is a property of the benchmark surface or of the llama3.1:8b model tier, by re-running the byte-identical frozen 48-cell E27 procedure at exactly one other tier (`qwen2.5:7b`) with the same fixtures, budget, methods, embeddings and locked gate thresholds. E28 never redesigns the surface, never re-runs E26/E19, never runs a head-to-head and never ranks models.
**Mode:** standard
**Success Criteria:**
1. `experiments/e28_model_tier_certification.py` (new): thin output-rebinding wrapper (E22/E23 pattern) that delegates the whole grid + gate computation to `e27_calibration.main` unchanged (`--model qwen2.5:7b`), then wraps the E27-shaped payload into the E28 result shape (`experiment`, `model`, `source_calibration`, `config`, `records`, `pilot_gates`, `aggregate`, `model_comparison`, `verdict` with `gates_passed`/`head_to_head_eligible`/`rationale`); `--run`/`--resume`/`--force`/`--report-only`/`--dry-run` (dry-run builds the four contexts with no LLM); E28 15-section report with Tables A-E (gate comparison, per-method, per-group, contradiction, budget/context structure); no threshold duplicated — wraps `e27.compute_pilot_gates`/`compute_verdict`
2. Outputs `experiments/results/e28_model_tier_certification.json` + `_report.md` (new, force-added); `model_comparison` has no `no_model_ranking=False`-style ranking and `head_to_head_eligible = calibration_valid` but no head-to-head is run
3. `tests/test_e28_model_tier_certification.py` (new, offline-only): grid surface identical to E27, output plumbing, delegation with `--model qwen2.5:7b` + rebinding restored in `finally` (even on error), no LLM coder in dry-run, E28 top-level shape + verdict forwarding, model-comparison gate table, 15 sections + Tables A-E, both model names and no ranking/wins language, no redefinition of locked thresholds, no `requests`/`server` imports; hermetic (no Ollama/network, no writes to immutable artifacts)
4. **Run (48 cells, qwen2.5:7b @ 96, seeds 1–2): offline fixture gates PASS on all 6 variants, 48/48 complete; gate pattern identical to E27@llama3.1:8b — budget_binding (both 1.0 bound), context_difference (12/12), no_leakage (0) PASS; history_dependence FAIL (route_contract/serialization_contract direct 1.0 AND no_history 1.0 → diff 0; retry_contract direct 0.25 vs no_history 0.50, diff -1) and contradiction_state FAIL (adaptive corr_recall 1.0 / obs 0.0, vanilla obs_exposure 0.667 < 0.75**) → **calibration_valid = False, head_to_head_eligible = False at qwen2.5:7b too**; per-method success no_history 0.833 / direct_history 0.75 / adaptive 0.667 / vanilla_rag 0.417; the E27 failure does NOT isolate to the llama3.1:8b tier — the surface remains the leading explanation; no redesign, no winner, no head-to-head
5. Docs updated (STATE.md Phase 23 block + D45 in Notes + known issue #25, ROADMAP Phase 23, gsd_handoff, brain.md D45); `.planning/checkpoints/phase-23-e28-model-tier-certification-complete.md` created; `memory_optimizer/**`, `baselines/**`, `config.yaml`, `server.py` and every E27/E26/E19/E20/E24/E25 artifact untouched; commits `research: certify E27 calibration at qwen2.5 7b` then `docs: record phase 23 model-tier certification`
