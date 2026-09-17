# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-16)

**Core value:** Correction handling must work; write-time salience must be real; context-pressure experiments validate recall-per-token under genuine budget stress
**Current phase:** Phase 11 (E15 retention-policy separation: activation vs survival) complete

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
- **Phase 11** — E15 retention-policy separation (activation vs survival) · commit `TBD`

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

## Notes

- Codebase map at `.planning/codebase/` — all 7 docs committed
- `brain.md` governs all memory_optimizer/server.py changes (D24 correction supersession, D25 write-time salience, D26 eval/versioning/dashboard/audit, D27 context-pressure probe recall, D28 coding-context usefulness benchmark, D29 separate store/active + query-first retrieval, D30 Phase 9 generalization + causal ablations, D31 Phase 10 retention diagnosis, D32 Phase 11 activation-vs-survival separation)
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

---

State initialized: 2026-09-16
Last updated: 2026-09-17 after Phase 11 E15 retention-policy separation