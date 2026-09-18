# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-16)

**Core value:** Correction handling must work; write-time salience must be real; context-pressure experiments validate recall-per-token under genuine budget stress
**Current phase:** Phase 14 (E18 correction-safety validation) complete — offline identity gate PASS, targeted coding validation PASS, correction authoritativeness established

## Completed Phases

- **Phase 1** — Foundation (dead code cleanup + pytest infra) · commit `0880f1b`
- **Phase 2** — Correction supersession fix · commit `be75897`
- **Phase 3** — Write-time salience + offline sync · commit `70866c3`
- **Phase 4** — Evaluation fixes + manifest versioning · commit `134d5b5`
- **Phase 5** — Dashboard + quick gate + regression tests · commits `f011d2c`, `91fa6b4`
- **Phase 6** — Revalidation + documentation + final audit · commit `bf581ba`
- **Phase 7** — E12 coding-context benchmark (negative result) · commit `60de0c5`
- **Phase 8** — E12 architectural fixes (store/active separation, query-first retrieval) · commit `4f429d0`
- **Phase 9** — E13 generalization + causal ablations (store capacity + retrieval policy) · commit `f6da170`
- **Phase 10** — E14 retention diagnosis (causal analysis of store losses) · commit `7a026b9`
- **Phase 11** — E15 retention-policy separation (activation vs survival) · commit `4889f2b`
- **Phase 12** — E16 retention selectivity under hard pressure (task_affinity rejected) · commit `c971553` (+docs `6b25a80`)
- **Phase 13** — E17 long-horizon coding capability (no adaptive advantage) · commit `0025589`
- **Phase 14** — E18 correction-safe memory, targeted validation (retrieval-safe supersession) · commit `TBD_AFTER_COMMIT`

## Key Metrics (Post-fix, 3-seed validation — `manifest_h3795b2da.json`)

- correction_recall: 1.0 (pre-fix 0.0)
- wrongly_retained_after_correction.fraction: 0.2 (pre-fix 1.0)
- negation_recall: 1.0, trap_recall: 1.0
- E1 adaptive: 44.0 tok/turn vs baseline 323.0 (pre-fix ~19): +7.9% vs vanilla (< 15% gate)
- E2 proposed positive recall: 0.760 (pre-fix 0.72)
- E10 context-pressure grid: 48 cells × 4 budgets × 4 turns × 3 seeds, word-count tokens, lexical embedding model
  - Probe recall (point-of-need, strict entity match, final-store): adaptive 8.4% @ 340 store-tok, summarization 10.5%→84.8% (budget-proportional), memgpt/vanilla 100% @ ~4875 toks
  - Efficiency (recall per store token): adaptive ~2.5e-4, summarization ~2.1e-4 linear with budget, unbounded baselines ~2.0e-4
  - Budget-stressed cells (raw tokens > budget): 500 turns at 512 budget stressed; 4096 budget never stressed at 50 turns
  - store recall (E2 durable): 0.76 adaptive, 1.0 baseline (blended FP ceiling)
- E4 hard-case: dist 10 adaptive 0.75/baseline 1.0; dist 25 adaptive 0.0/baseline 1.0 (transient coffee facts intentionally forgotten + degenerate unstressed baseline)

## E12 Coding-Context Usefulness Benchmark (D28) — NEGATIVE for adaptive (PRE-FIX)

- `data/coding_workload.py` + `experiments/e12_coding_benchmark.py`; 84 diverse facts, 400 turns, budgets {128,256,512}, 3 seeds, nomic-embed-text (production retrieval path). Budget genuinely binding (natural store 1024 tok = 2–8× budget).
- 3-seed mean query-time injected context_recall (pre-fix: active=store budget):
  - adaptive: 0.155 / 0.167 / 0.167 (flat across budgets despite store_recall 0.19→0.35→0.53)
  - vanilla_rag: 0.905 @ ~60 injected tok; memgpt_style 0.905 @ ~121 tok; summarization_only 0.179→0.333→0.583; sliding_window 0.048
- Root causes: (1) bounded store + decay retains only ~37% of facts at top budget; (2) importance-dominant retrieval `0.6·imp+0.4·sim` — for a query, importance spans 0.47 vs similarity span 0.069, so injection is query-insensitive. Ablation @512: production 0.167, pure-similarity 0.421, pure-importance 0.115.
- Retrieval weights NOT tuned (directive: do not optimise adaptive to beat the benchmark). D3 "calibrate" flagged as remediation.
- Fairness fixes (not tuning): facts spread across whole session (was first-half + decay-prune artifact); distinct predicate families replace near-duplicate templates (dedupe retains ~86–88%, was 332→189).
- Tests: 33 passing (`tests/test_coding_benchmark.py` 9).

## E12 Architectural Fixes (D29) — Phase 8 Results (POST-FIX)

**Changes implemented:**
1. Separate `memory_store_token_budget` (long-term) from `max_context_tokens` (active context) + `injection_token_limit` (explicit override)
2. Retrieval defaults: `imp_weight=0.15, sim_weight=0.85, cat_bonus=0.02`; formula: `sim_weight*sim + imp_weight*imp + bonus`
3. E12 uses `memory_store_token_budget=max(4096, budget*4)` + `injection_token_limit=budget`
4. Retrieval diagnostics: `store_recall`, `retrieval_loss`, `mean_context_utilization`

**Phase 8 Results (E12 POST-FIX, 3 seeds):**
- adaptive ctx_recall: 0.71 / 0.80 / 0.92 at 128/256/512 (vs 0.15/0.17/0.17 pre-fix)
- adaptive store_recall: 0.79 / 0.85 / 0.92 (vs 0.19/0.35/0.53 pre-fix)
- retrieval_loss: 0.075 → 0.056 → 0.000 (decreases with budget)
- correction_recall: 0.83 / 1.0 / 1.0 (vs 0.5 pre-fix)
- store_tokens: 751 / 869 / 983 (independent 4096 store budget)
- mean_context_utilization: 0.98+ at all budgets (active budget fully utilized)
- vanilla_rag unchanged at 0.905; summarization scales 0.18→0.33→0.58

## E13 Generalization + Causal Ablations (D30) — Phase 9 Results

**E13 Generalization (5 seeds × 5 budgets × 5 methods):**
- adaptive ctx_recall: 0.58 / 0.71 / 0.80 / 0.92 / 0.92 at 64/128/256/512/1024
- adaptive store_recall: 0.67 / 0.77 / 0.85 / 0.92 / 0.92
- retrieval_loss: 0.10 → 0.07 → 0.05 → 0.00 → 0.00 (decreases with budget)
- correction_recall: 0.50 / 0.70 / 1.0 / 1.0 / 1.0
- mean_context_utilization: 0.96+ at all budgets; zero budget violations
- vanilla_rag stable at 0.905; summarization scales 0.08→0.18→0.33→0.58→0.89

**Ablation A — Store-Capacity Separation (3 seeds × 4 policies × 5 budgets):**
- coupled_old_behavior (store=active): ctx_recall 0.18 at 128
- matched_budget (store=active): ctx_recall 0.18 at 128  
- separated_4x (Phase-8): ctx_recall 0.71 at 128, store 735 tok
- unbounded (0): ctx_recall 0.71 at 128, store 735 tok
- *Store separation is necessary for high retention; 4x ≈ unbounded at this workload scale*

**Ablation B — Retrieval Policy (3 seeds × 4 profiles × 3 budgets):**
- phase8 (0.15/0.85): ctx_recall 0.71, loss 0.07
- legacy_phase7 (0.6/0.4): ctx_recall 0.27, loss 0.37
- pure_similarity (0/1): ctx_recall 0.86, loss 0.01
- pure_importance (1/0): ctx_recall 0.18, loss 0.42
- *Query-first retrieval is the primary driver; pure similarity slightly outperforms Phase-8*

**Embedding vs Lexical (2 seeds × 3 budgets):**
- embeddings: ctx_recall 0.70, loss 0.08
- lexical: ctx_recall 0.57, loss 0.20
- *Phase-8 benefit partially depends on embedding similarity discrimination*

**Tests:** 40 passing (11 retrieval + 9 coding + 8 compression + 6 scoring + 7 pipeline)

## E14 Retention Diagnosis (D31) — Phase 10

**Objective:** Identify exactly why useful task facts disappear from the long-term store (pure diagnosis, no policy change).

**Implementation:**
- Store-pressure instrumentation in `pipeline.py`: per-ingest `last_store_pressure` (pre/post tokens, eviction count, over-budget delta); `write_time_salience` stamped at ingest
- `experiments/e14_retention_diagnosis.py`: instrumented replay + 9-state lifecycle classifier + ablations
- Lifecycle states: PRESENT_AND_RETRIEVED, PRESENT_BUT_NOT_RETRIEVED, REMOVED_BY_DECAY, REMOVED_BY_STORE_BUDGET, MERGED_BY_DEDUPE, SUPERSEDED_CORRECTLY, SUPERSEDED_INCORRECTLY, NEVER_STORED, OTHER

**Results (5 seeds × 4 budgets, turns=400, scale=3):**
- Dominant store loss = **decay-driven pruning** at low budgets:
  - decay removals: 31.4 / 20.4 / 10.6 / 0.2 facts at 64/128/256/512
  - store-budget eviction: 0.0 everywhere (4096 store never binds at 86 facts)
  - dedupe: 4.0 constant (template collision, by design); supersession: 2 correct / 0 incorrect
- ctx_recall: 0.576 / 0.705 / 0.800 / 0.917; store_recall: 0.672 / 0.774 / 0.852 / 0.917
- Ablation A (no decay): decay losses → 0, store_recall 0.917, correction_recall 1.0 at ALL budgets (vs 0.5 at 64 with decay)
- Ablation B (decay applied, no prune): decay losses → 0, store_recall 0.917; ctx_recall 64 → 0.862 (decayed importance lowers retrieval rank but facts survive)
- Ablation D: write-time salience ≈0.98 for all facts (no discrimination); retrieval_sim ≈0.48; spearman ≈ -0.08
- Correction/obsolete gate never regressed (obsolete_retention = 0 in every run)
- Retrieval loss (PRESENT_BUT_NOT_RETRIEVED) ≈ 5–6 facts/cell, consistent across budgets

**Conclusion:** the mechanism responsible for useful-fact loss is the decay→prune path, not budget eviction and not dedupe/supersession. Budget only matters indirectly: fewer injected facts → less reinforcement → faster decay. No policy change made (Phase 10 is diagnosis-only).

**Tests:** 54 passing (+14 new: lifecycle tracking, classifier, store-pressure eviction, production-defaults guard)

## E15 Retention-Policy Separation (D32) — Phase 11

**Objective:** separate dynamic **activation** (`current_importance`: decays, drives retrieval ranking) from long-term **survival** (`retention_priority`: stable evidence score, drives store eviction). Apply the OBSERVED-based decision rule and advance the winning candidate to the production default.

**Implementation:**
- `config.yaml` `retention: {mode, eviction_priority}` (default advanced to `dual_score` / `retention_priority`); `load_settings()` returns it and honors programmatic overrides
- `decay.py`: `calculate_retention_priority()` = `min(1.0, base_score*(1+access_gain*(access_count-1)))` (deterministic, NO temporal decay); `step_decay_and_prune(..., retention_mode)` — `hard_threshold` (legacy prune), `soft_decay`/`dual_score` (never threshold-prune; flag `retained_below_threshold`); unknown mode → `ValueError`
- `budget.py`: `token_budget_evict(..., priority_key)` — `dual_score → retention_priority`; tie-break unchanged (oldest first)
- `experiments/e15_retention_policy.py`: primary grid (4 policies × 4 budgets × 5 seeds) + stress grid (3 policies × 3 store budgets × 3 active budgets × 3 seeds, scale 27, 1200 turns, natural store ~8272 tok) + lifecycle registry (activation + retention transcripts) + report + auto-classifier (`classify_result`)
- Retrieval path untouched (frozen: ranking uses `current_importance` only) — causal isolation test included

**Primary results (mean context recall / store recall, 5 seeds × 400 turns):**
- hard_threshold (legacy): ctx 0.576/0.705/0.800/0.917; store 0.672/0.774/0.852/0.917; decay removals 31.4/20.4/10.6/0.2
- soft_decay: ctx 0.862/0.883/0.888/0.917; store 0.917 flat; decay removals 0; obsolete 0; correction_recall 1.0 at ALL budgets (vs 0.5/0.7 legacy)
- dual_score: identical to soft_decay in primary (no store-pressure evictions); adds retention-priority eviction under stress
- no_decay: ctx 0.893/0.893/0.893/0.917, store 0.917 (upper-bound diagnostic, shows remaining leakage is retrieval-side)
- Revived facts (would-be-pruned, later retrieved): 30.0/20.4/9.6/0.2 at 64/128/256/512 — proves recovery is real, not mere data preservation

**Stress results (scale 27, ~8272 natural store tokens, 1200 turns):**
- At tightest store budget 1024: dual_score ctx 0.368/0.434/0.503 vs soft 0.322/0.405/0.484; correction_recall 0.94 vs 0.28; dual evicts 430 vs soft 518 (more stable store); archival store_recall −1 pt tradeoff (dual evicts never-retrieved low-retention facts)
- At 2048/4096 store: dual store 0.805/0.897 vs soft 0.775/0.877; hard_threshold loses up to ~409 facts to decay-pruning even before eviction
- obsolete_retention = 0 under EVERY policy/every budget (no stale resurrection; supersession intact)

**Decision rule verdict (auto-classified, reviewer-confirmable):**
- SEPARATING ACTIVATION FROM SURVIVAL IS SUPPORTED → **SUPPORTED** (store +0.113, ctx +0.138, active-context unchanged, safety preserved, store growth bounded)
- RETENTION PRIORITY FOR STORE EVICTION IS SUPPORTED → **SUPPORTED** (under tightest store pressure: context +0.031, correction +0.667; archival store_recall −0.009 tradeoff)
- SOFT RETENTION ... CREATES A STALENESS TRADEOFF → **NOT SUPPORTED** (obsolete 0, correction improved, no stale accumulation)

**Advanced default:** `retention.mode: dual_score`, `retention.eviction_priority: retention_priority` (decay no longer deletes; store eviction prefers low retention). Pipeline fallback for settings lacking the key stays `hard_threshold` (backward-compatible). E14 suite pinned to legacy mode explicitly in its test helper.

**Tests:** 80 passing (+26 new in `tests/test_e15_retention_policy.py`: retention defaults/modes/priority math/eviction priority parity, retrieval isolation, revival hard-vs-soft, supersession safety under soft, E15 cell shape)

## E16 Retention Selectivity (D33) — Phase 12

**Objective:** (bounded) Determine whether the system can selectively retain task-relevant facts under genuine store pressure; harden Phase-11 evaluation; diagnose the retrieval-feedback loop on `retention_priority`; test future-blind `task_affinity` against `random`/`base_score_only` and an offline `oracle_future_use` ceiling. **No post-result tuning; config changes only if task_affinity passes ALL 8 criteria.**

**Implementation:**
- Identity-safe metrics (`data/coding_workload.py`): deterministic `fact_id = category:qtype:source_turn:SHA256(text)[:16]` (SHA-256, not Python hash); queries carry `target_fact_ids`/`forbidden_fact_ids`; identity context/store/correction recall + obsolete_retention (immune to token collisions like `memcache`/`redis`/`100`/`250`)
- Access separation (`retrieval.py`/`compression.py`): `retrieval_access_count` (retriever `_bump` only) vs `ingest_reinforcement_count` (compressor only), isolating feedback loops circling `access_count`
- `memory_optimizer/task_state.py`: `TaskStateTracker(window=32)` FIFO of fact-carrying turn embeddings; `task_affinity = mean(top-4 cosine)`, clamped [0,1], observed at current-turn ingest (causal/future-blind)
- Eviction policies (all `retention.mode=dual_score`, retrieval FROZEN): dual_score (retention_priority), base_score_only, task_affinity (lexicographic `(task_affinity, retention_priority)`), random (seeded lower bound), oracle_future_use (OFFLINE future-peeking upper bound, leak-guard asserted)
- `pipeline.py` `protect_corrections=True` in E16: current correction never evicted while its superseded predecessor exists
- `experiments/e16_retention_selectivity.py`: 300-cell grid (5 policies × STORE[256,512,1024,2048] × active[64,128,256] × seeds[42..46], 1200 turns, scale 27, 662 gt facts — genuine pressure) + no-pressure diagnostic (store_budget=0, fingerprints identical across policies) + window sensitivity 16/32/64 + 18-section auto report + E15 raw-JSON audit reconciliation

**Results (see brain.md D33 for full tables):**
- E15 reconciliation (81 stress cells): token-based obsolete_retention > 0 in 27 cells, correction_recall < 1 in 44, both 11, clean 21; `SUPERSEDED_INCORRECTLY = 0` in ALL cells; real loss = NEW corrected fact missing (dual 0/0/0, hard_threshold 18/14/13, soft 18/13/9 at 1024/2048/4096)
- identity_store_recall (mean): dual 0.057/0.103/0.129/0.253 ≈ task_affinity 0.049/0.093/0.126/0.252 ≈ random 0.048/0.095/0.126/0.250; oracle 0.057/0.103/0.129/0.253 ≈ dual (ceiling reached at grid's store range)
- precision_of_retention = 1.000 for ALL policies; dual identity correction recall 1.0 (0.867 at 1024 — verified real marginal-eviction effect near natural store size ~1043 tok)
- window w16/w64: rr 0.0636/0.0667, precision 1.0 — insensitive

**Decision rule verdict (auto-classified, reviewer-confirmable):**
- TASK_AFFINITY FOR RETENTION → **NOT SUPPORTED** (0/8: c1 FAIL future-use rr 0.0709 vs dual 0.0802; c6 FAIL 0/5 seeds; c7 FAIL vs base_score_only; c8 FAIL oracle store gap task +0.009 vs dual −0.0003)
- RETRIEVAL-FEEDBACK DOMINATES SURVIVAL → **NOT OBSERVED** (inconclusive — oracle ≈ causal at low store; workload cannot discriminate)
- IDENTITY-SAFE EVAL + CORRECTION PROTECTION → **SUPPORTED** (obsolete 0, corr recall 1.0, SUPERSEDED_INCORRECTLY 0 across all 300 cells)

**Production default UNCHANGED:** `retention.mode: dual_score`, `retention.eviction_priority: retention_priority`. No silent flip. `task_affinity` remains opt-in for workloads with concentrated future use. Retrieval stays frozen (0.85 sim + 0.15 imp + 0.02 cat_bonus, top_k 5, sim_threshold 0.35).

**Tests:** 117 passing (+37 new in `tests/test_e16_retention_selectivity.py`: identity-safe API, tracker causality/windowing, access-separation cell fields, evicted-population stamps, oracle isolation/ceiling, SUPERSEDED_INCORRECTLY==0 across cells, correction gate, no-leak flags, hard-budget enforcement, fingerprints, `_cell_key`, workload self-test)

## E17 Long-Horizon Coding Capability (D34) — Phase 13

**Objective:** the downstream validity test — does the adaptive memory system help a real coding model (llama3.1:8b) complete long-running repository-editing tasks better than simpler historical-context managers when the historical context supplied to the model is fixed? Dependent variable = hidden-test pass of the produced edit, not recall.

**Implementation:**
- `data/coding_task_suite.py` + `data/coding_tasks/{cache_readonly,user_ids,write_retry,validation_pure}/{workspace,hidden,gold.patch}`: 4 real packages, ~600-turn transcripts burying constraints among distractors, deterministic hidden tests never seen by the model; gold patches verified (base fails / gold passes)
- `experiments/coding_benchmark.py` (shared harness): method protocol (prepare→retrieve), complete-file edit blocks applied verbatim (+diff fallback), 2 coding attempts, word-count tokenizer, per-run leakage assertion, deterministic failure taxonomy, recall diagnostics (obsolete facts excluded from recall denominators)
- Methods (history mechanism = only difference): raw_clipped, sliding_window, llm_summarization (running summary from same model every 50 turns), vanilla_rag (static cosine store, no decay/importance/eviction), adaptive (production `dual_score` + `protect_corrections=True`); diagnostics no_history / full_context / direct_history (oracle gold facts)
- `experiments/e17_coding_capability.py`: pilot (3×2×2×5=60) + full grid (4×3×3×5=180), 9 gates, paired bootstrap CIs, verdict rule, 24-section report; **incremental JSON → resumable** (`--force`/`--report-only`/`--limit`/`--full` CLI)
- Required harness fix (recorded, not an experiment result): the coding model emits malformed multi-hunk unified diffs that `git apply` applies partially/silently (even the oracle failed) → primary edit format is complete-file blocks, applied deterministically

**Results (full grid: 4×3×3×5 = 180 runs, ≤2 attempts each):**
- All 9 gates pass: budget pressure (full raw history ~7.2k tokens > 1k budget and > 8k window with workspace), workspace independence, methods differ, real summarization (144 update calls), adaptive production path, no leakage, history dependence (user_ids + validation_pure: no-history fails, oracle succeeds), test determinism, edit determinism
- Overall success rate: adaptive/llm_summarization 0.75, raw_clipped/sliding_window/vanilla_rag 0.722 — no separation
- Paired vs adaptive (36 pairs each): diff vs raw +0.028 (CI 0.00–0.08), vs sliding +0.028 (CI 0.00–0.08), vs llm +0.000 (CI 0–0), vs vanilla +0.028 (CI 0.00–0.08) → **no CI excludes zero; adaptive_advances = False**
- First-pass success worst for adaptive (0.53 vs 0.58–0.69); retrieval highlight does not help first-try code correctness at these budgets
- Failure mechanism in failing cells: `correction_recall=0` + `obsolete_fact_exposure=1.0` — both adaptive and vanilla_rag inject the obsolete fact and miss the correction (identical contexts); recall-only metrics would have masked this
- `full_context` diagnostics always overflow the 8k window once the workspace is attached → "show everything" is genuinely unavailable
- cache_readonly/write_retry are solvable from the workspace alone (no-history passes); only 2/4 tasks are strictly history-dependent

**Decision rule verdict (auto-classified, reviewer-confirmable):**
- ADAPTIVE MEMORY IMPROVES LONG-HORIZON CODING AT FIXED BUDGET → **NOT SUPPORTED** (no paired CI excludes zero; all arms cluster at 0.72–0.75)
- All 9 pilot gates PASS → the grid is a valid experiment; the negative verdict is a finding, not a noisy instrument

**Production default UNCHANGED:** `config.yaml` untouched; no memory_optimizer behavior modified by E17 (harness + baselines + data only). Retrieval remains frozen (0.85 sim + 0.15 imp + 0.02 cat_bonus, top_k 5, sim_threshold 0.35).

**Tests:** 134 passing (+17 new in `tests/test_e17_coding_capability.py`: task-suite determinism, gold-patch harness correctness on all 4 tasks, per-method budget enforcement, no-history vs oracle diagnostics, leakage logic-vs-import separation, file-edit parse/apply + path-traversal guard, failure-class precedence, aggregation/verdict shape, 24-section report, artifact paths under repo). Results: `experiments/results/e17_coding_capability.json` + `_report.md` (force-added; /tmp unavailable).

## E18 Correction-Safe Memory, Targeted Validation (D35) — Phase 14

**Objective:** fix the correction/supersession failure that E17 surfaced at the memory-consolidation boundary (obsolete fact injected, correction missed), add regression tests, validate with an offline identity gate + a targeted coding grid, and report honestly.

**Root cause (fixed):** `dedupe_incremental` ran `_is_supersession` only *inside* `if sim >= 0.90`. A correction restating the old fact with genuinely new wording (cosine ~0.7) never reached the supersession branch → stored as an additional memory, obsolete fact never replaced → retrieval saw both facts. Config, retrieval weights, decay, embedding model unchanged.

**Implementation:**
- `memory_optimizer/compression.py`: `_replace_with_supersession(ex, mem, mem_emb)` helper; `supersedes_turn` mutation now calls it; supersession check moved BEFORE any similarity computation in the generic same-category loop; `_is_supersession()` and duplicate-merge unchanged (revision/negation marker + full restatement still required); docstring updated to reflect resolution ordering.
- Tests (134 → 138): `test_compression_supersession.py` +2 (supersession precedes embedding threshold 0.7<0.90; correction with new semantic wording replaces old fact via lexical path), `test_retrieval_ranking.py` +1 (e2e: compressed correction is the only authoritative memory), `test_e17_coding_capability.py` +1 (cache_readonly hidden test strictness).
- `data/coding_tasks/cache_readonly/hidden/test_hidden.py` rewritten: spy wraps original `CacheCoordinator.set` (delegates to keep gold passing), asserts `calls == [("beta","2")]` and `store.get("beta")=="2"` — rejects naive `store.put` solutions.
- `experiments/e18_correction_safety.py`: 36-cell offline identity gate (4 tasks × 3 seeds × 3 budgets) + 72-run coding grid (`user_ids`+`validation_pure`, 4 methods, llama3.1:8b) + 19-section report, all artifacts under `experiments/results/`.

**Offline identity gate:** 36 cells, 27 correction-bearing — ALL PASS (correction_recall 1.0, obsolete_exposure 0.0 in every cell). Current fact is the only authority; obsolete fact absent by identity.

**Targeted coding validation (72 runs, ≤2 attempts each):**
- adaptive: 17/18 (94.4%), correction_recall 1.0, obsolete_exposure 0.0
- vanilla_rag: 9/18 (50%), correction_recall 0.0, obsolete_exposure 1.0, all failures OBSOLETE_INFORMATION_USED
- llm_summarization: 9/18 (50%), correction_recall 0.17–0.67, MEMORY_MISS dominant
- raw_clipped: 8/18 (44.4%), correction_recall 0.0, RETRIEVAL_MISS dominant
- Paired adaptive diffs: +0.4444 vs vanilla_rag, +0.5 vs raw_clipped
- Adaptive's single failure (validation_pure, seed 2, budget 1024): CODING_ERROR — correction was in context; model applied a wrong edit. Model failure, not memory failure.

**Decision rule verdict (auto-classified, reviewer-confirmable):**
- EXPLICIT CORRECTION SUPERSESSION IS RETRIEVAL-SAFE AFTER D35 FIX → **SUPPORTED** (offline identity gate all-PASS + coding grid: correction_recall 1.0, obsolete_exposure 0.0, adaptive 17/18)
- Same-model textual summaries (llm_summarization) can carry a correction, but its propagation across windows is lossy (correction_recall 0.17–0.67) → **NOT** a reliable correction carrier

**Production default UNCHANGED:** `config.yaml` untouched (retention dual_score / retention_priority, top_k 5, sim_threshold 0.35, compression enabled, correction protection enabled). The D35 change is ordering of the supersession check only.

**Tests:** 138 passing. Results: `experiments/results/e18_correction_safety.json` + `_report.md` (19 sections; force-added). Cache_readonly benchmark limitation (E17 known-issue #15) resolved by the stricter hidden test.

## Notes

- Codebase map at `.planning/codebase/` — all 7 docs committed
- `brain.md` governs all memory_optimizer/server.py changes (D24 correction supersession, D25 write-time salience, D26 eval/versioning/dashboard/audit, D27 context-pressure probe recall, D28 coding-context usefulness benchmark, D29 separate store/active + query-first retrieval, D30 Phase 9 generalization + causal ablations, D31 Phase 10 retention diagnosis, D32 Phase 11 activation-vs-survival separation, D33 Phase 12 retention selectivity / task_affinity rejected, D34 Phase 13 long-horizon coding capability / no adaptive advantage, D35 Phase 14 correction-safe memory / retrieval-safe supersession)
- Git: fresh repo at RishiMotwani/test248 (public), branch master
- Session relocated to prototype3
- Two worktrees removed: `prototype3_0880f1b_wt`, `prototype3_pre_reval_wt`
- /tmp cleaned; artifacts on /home filesystem

## Instrumentation gaps (honest list, for reviewers)

- E3 latency unmeasured offline (null in manifest)
- MM multi-model variance never measured
- No budget-stress experiment at raw-history > 4096 (only ~480+ turns stress)
- E4 distance-25 recall 0.0 not yet root-caused
- Negation recall has no quick-gate threshold
- No post-fix live-verified run (only stale pre-fix live manifest)

## Known issues (for reviewers)

1. Post-fix token-vs-correctness tradeoff: +121% vs pre-fix adaptive, gate basis ambiguous (REVAL-04)
2. E4 needle recall drops to 0.0 at 25-turn distance
3. Stale live manifest (pre-fix) can be mistaken for current results
4. Embed-path dependence changes per-seed misuse fraction (0.0 vs 0.2)
5. Default 50-turn config does not stress 4096 budget; barebone fits entirely
6. D27: point-of-need probe recall isolates retained content from budget effects
7. D29: Phase 8 validation complete — E12 POST-FIX metrics confirm fix
8. D30: Phase 9 complete — Phase 8 generalized; query-first retrieval is primary causal driver
9. D31: Phase 10 complete — decay→prune is the dominant store-loss mechanism; no policy change made
10. D32: Phase 11 complete — separation of activation/survival supported; advanced default dual_score + retention-priority eviction
11. D33: Phase 12 complete — E16 task_affinity rejected at 8-criteria gate (c1/c6/c7/c8 FAIL, 0/5 seed wins; oracle ≈ causal at low store, workload cannot discriminate); dual_score retained, config unchanged
12. E16 full grid verified via identity-safe metrics; E15 token-based obsolete_retention reclassifies as metric artifact (real residual loss = new corrected fact evicted, now guarded by correction protection)
13. `/tmp` 100% full during E16 runs — all logs/artifacts written under `experiments/results/` (repo), not /tmp
14. D34: Phase 13 complete — E17 full grid (180 runs) valid, all gates pass, but adaptive shows NO advantage at fixed historical budgets (all arms 0.72–0.75, every paired CI lower bound 0); failing cells = obsolete fact injected + correction missed (both adaptive and vanilla_rag); harness fixed to complete-file edit format (model's diffs were malformed — applied partially/silently)
15. E17 coding task `cache_readonly` hidden test only asserts `put_count==1`, which a naive `store.put` also satisfies — the CellCoordinator/CacheCoordinator constraint is not strictly enforced (documented as a suite limitation)
16. D35: Phase 14 complete — E18 offline identity gate (36 cells, 27 correction-bearing) ALL PASS, targeted coding validation adaptive 17/18 vs vanilla_rag 9/18 (all OBSOLETE_INFORMATION_USED) / llm_summarization 9/18 / raw_clipped 8/18 (all RETRIEVAL_MISS); cache_readonly benchmark limitation (#15) RESOLVED by spy-based hidden test enforcing CacheCoordinator + `calls == [("beta","2")]`

---

State initialized: 2026-09-16
Last updated: 2026-09-18 after Phase 14 E18 correction-safe memory — offline identity gate PASS, adaptive 17/18 vs vanilla 9/18 on targeted coding validation