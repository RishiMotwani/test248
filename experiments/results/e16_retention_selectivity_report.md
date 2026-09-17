# E16 Retention Selectivity Report (Phase 12)

Selective retention of task-relevant memories under hard store pressure, measured with **identity-safe** fact-level metrics; diagnosis of the retrieval-feedback loop on `retention_priority`; causal baselines and an offline oracle upper bound for one future-blind `task_affinity` candidate.

## 1. Purpose & Hypothesis

Under genuine store pressure the survival policy decides which facts the long-term store keeps. `retention_priority` (E15) is fed by `access_count`, which increments on retrieval — so a fact that is *retrieved* becomes *more likely to survive*, independently of task relevance. E16 asks whether (a) this loop actually dominates survival, and (b) a future-blind task-similarity signal (`task_affinity`) can retain task-relevant facts better than the causal baselines (`random`, `base_score_only`) and close part of the gap to an offline `oracle_future_use` upper bound — without resurrecting superseded facts or sacrificing correction safety.

## 2. Phase 11 Audit & E15 Raw-JSON Reconciliation

- Surveyed **81** E15 stress cells (1200 turns, scale 27).
- Cells with token-based `obsolete_retention > 0`: **27**; `correction_recall < 1`: **44**; both: 11; clean: 21.
- Fact-level reconciliation: `SUPERSEDED_INCORRECTLY` appears in **0** of the cells (total **0** occurrences) — the store never resurrects a corrected fact as an authoritative memory. The E15 report's blanket 'obsolete_retention = 0 under every policy/budget' therefore overclaims ONLY at the token level, where the forbidden token legitimately occurs in unrelated facts.
- Token-level collision census (scale 27, authoritative gt):
  - `memcache`: 3 authoritative fact(s) contain it.
  - `redis`: 8 authoritative fact(s) contain it.
  - `100`: 0 authoritative fact(s) contain it.
  - `250`: 1 authoritative fact(s) contain it.
- Correction-content loss is REAL (E15 stress): the newly corrected fact is missing from the final store in many cells:

| policy | store | correction facts | missing from store |
| --- | --- | --- | --- |
| dual_score | 1024 | 18 | 0 |
| dual_score | 2048 | 18 | 0 |
| dual_score | 4096 | 18 | 0 |
| hard_threshold | 1024 | 18 | 18 |
| hard_threshold | 2048 | 18 | 14 |
| hard_threshold | 4096 | 18 | 13 |
| soft_decay | 1024 | 18 | 18 |
| soft_decay | 2048 | 18 | 13 |
| soft_decay | 4096 | 18 | 9 |

Phase 12 addresses this with identity-safe metrics (section 4) and correction protection (`protect_corrections=True`, section 11).

## 3. Methodology

All policies share `retention.mode = dual_score` so decay never deletes; the only causally varying knob is the store-eviction priority. `task_affinity` evicts lexicographically lowest `(task_affinity, retention_priority)`; `base_score_only` evicts lowest `base_score` only (no retrieval/decay/salience feedback); `random` is a seeded lower bound; `oracle_future_use` evicts future-irrelevant facts first (offline-only upper bound). Retrieval ranking is FROZEN (0.85*sim + 0.15*importance + category bonus), so separation acts only through survival.

## 4. Evaluation Definitions

- `fact_id = category:qtype:source_turn:SHA256(precise_fact)[:16]` (deterministic; no Python `hash()`).
- Queries carry `target_fact_ids` (must be retrievable) and `forbidden_fact_ids` (must NOT be); corrections assert the new fact's id present and the old fact's id absent.
- `fact_identity_context_recall` / `fact_identity_store_recall` / `fact_identity_correction_recall` / `fact_identity_obsolete_retention` measure presence/absence on fact ids, immune to token collisions (e.g. `memcache`/`redis`).
- Legacy token metrics are kept for comparability to E13–E15.
- `future_use`: offline label = fact id requested by any query. `retention_recall` = retained future-use facts / all future-use facts. `precision_of_retention` = retained future-use facts / retained authoritative facts. Future labels reach ONLY the offline oracle + metric code (section 15/24 checks).

## 5. Configuration

- **active_budgets**: [64, 128, 256]
- **store_budgets**: [256, 512, 1024, 2048]
- **seeds**: [42, 43, 44, 45, 46]
- **turns**: 1200
- **scale**: 27
- **embedding_model**: nomic-embed-text
- **matching**: identity-safe fact_id (+ legacy token metrics)
- **policies**: ['dual_score', 'base_score_only', 'task_affinity', 'random', 'oracle_future_use']
- **protect_corrections**: True
- **task_context_window_default**: 32
- **active_budget_enforced**: True
- **decision**: 8 criteria (see report section 17); no post-result tuning

### Policies

| Policy | eviction priority |
| --- | --- |
| dual_score | `retention_priority` — Phase-11 default control: evict lowest retention_priority. |
| base_score_only | `base_score_only` — Evict lowest base_score (write-time value only: no retrieval_access_count, no current_importance, no last_access_turn). |
| task_affinity | `task_affinity` — Evict lexicographically lowest (task_affinity, retention_priority); task_affinity = top-4 cosine to the recent run of fact-carrying turns. |
| random | `random` — Random eviction (seeded) — causal lower bound. |
| oracle_future_use | `oracle_future_use` — Evict lowest future_use (0 before 1) — OFFLINE-ONLY theoretical upper bound. |

## 6. Identity-Safe Recall Grid (policy x store, mean over active/seeds)

### fact_identity_store_recall

| Policy | 256 | 512 | 1024 | 2048 |
| --- | --- | --- | --- | --- |
| dual_score | 0.057 | 0.103 | 0.129 | 0.253 |
| base_score_only | 0.045 | 0.088 | 0.120 | 0.241 |
| task_affinity | 0.049 | 0.093 | 0.126 | 0.252 |
| random | 0.048 | 0.095 | 0.126 | 0.250 |
| oracle_future_use | 0.057 | 0.103 | 0.129 | 0.253 |

### fact_identity_context_recall

| Policy | 256 | 512 | 1024 | 2048 |
| --- | --- | --- | --- | --- |
| dual_score | 0.057 | 0.103 | 0.128 | 0.240 |
| base_score_only | 0.045 | 0.088 | 0.119 | 0.232 |
| task_affinity | 0.049 | 0.093 | 0.122 | 0.234 |
| random | 0.048 | 0.095 | 0.125 | 0.234 |
| oracle_future_use | 0.057 | 0.103 | 0.128 | 0.240 |

### legacy context_recall (comparability)

| Policy | 256 | 512 | 1024 | 2048 |
| --- | --- | --- | --- | --- |
| dual_score | 0.219 | 0.355 | 0.436 | 0.519 |
| base_score_only | 0.223 | 0.280 | 0.382 | 0.505 |
| task_affinity | 0.231 | 0.345 | 0.386 | 0.473 |
| random | 0.235 | 0.311 | 0.392 | 0.495 |
| oracle_future_use | 0.219 | 0.355 | 0.436 | 0.519 |

## 7. Hard-Pressure Grid (store x active x policy)

*This grid IS the pressured grid (store 256–2048 tokens, natural store far above; see section 5). Identity store recall by cell:*

| Policy | store | active | identity_store_recall_mean |
| --- | --- | --- | --- |
| dual_score | 256 | 64 | 0.039 |
| dual_score | 256 | 128 | 0.093 |
| dual_score | 256 | 256 | 0.039 |
| dual_score | 512 | 64 | 0.071 |
| dual_score | 512 | 128 | 0.166 |
| dual_score | 512 | 256 | 0.071 |
| dual_score | 1024 | 64 | 0.129 |
| dual_score | 1024 | 128 | 0.129 |
| dual_score | 1024 | 256 | 0.129 |
| dual_score | 2048 | 64 | 0.253 |
| dual_score | 2048 | 128 | 0.253 |
| dual_score | 2048 | 256 | 0.253 |
| base_score_only | 256 | 64 | 0.030 |
| base_score_only | 256 | 128 | 0.074 |
| base_score_only | 256 | 256 | 0.030 |
| base_score_only | 512 | 64 | 0.061 |
| base_score_only | 512 | 128 | 0.142 |
| base_score_only | 512 | 256 | 0.061 |
| base_score_only | 1024 | 64 | 0.120 |
| base_score_only | 1024 | 128 | 0.120 |
| base_score_only | 1024 | 256 | 0.120 |
| base_score_only | 2048 | 64 | 0.241 |
| base_score_only | 2048 | 128 | 0.241 |
| base_score_only | 2048 | 256 | 0.241 |
| task_affinity | 256 | 64 | 0.035 |
| task_affinity | 256 | 128 | 0.076 |
| task_affinity | 256 | 256 | 0.035 |
| task_affinity | 512 | 64 | 0.065 |
| task_affinity | 512 | 128 | 0.150 |
| task_affinity | 512 | 256 | 0.065 |
| task_affinity | 1024 | 64 | 0.126 |
| task_affinity | 1024 | 128 | 0.126 |
| task_affinity | 1024 | 256 | 0.126 |
| task_affinity | 2048 | 64 | 0.252 |
| task_affinity | 2048 | 128 | 0.252 |
| task_affinity | 2048 | 256 | 0.252 |
| random | 256 | 64 | 0.033 |
| random | 256 | 128 | 0.079 |
| random | 256 | 256 | 0.033 |
| random | 512 | 64 | 0.065 |
| random | 512 | 128 | 0.154 |
| random | 512 | 256 | 0.065 |
| random | 1024 | 64 | 0.126 |
| random | 1024 | 128 | 0.126 |
| random | 1024 | 256 | 0.126 |
| random | 2048 | 64 | 0.250 |
| random | 2048 | 128 | 0.250 |
| random | 2048 | 256 | 0.250 |
| oracle_future_use | 256 | 64 | 0.039 |
| oracle_future_use | 256 | 128 | 0.093 |
| oracle_future_use | 256 | 256 | 0.039 |
| oracle_future_use | 512 | 64 | 0.071 |
| oracle_future_use | 512 | 128 | 0.166 |
| oracle_future_use | 512 | 256 | 0.071 |
| oracle_future_use | 1024 | 64 | 0.129 |
| oracle_future_use | 1024 | 128 | 0.129 |
| oracle_future_use | 1024 | 256 | 0.129 |
| oracle_future_use | 2048 | 64 | 0.253 |
| oracle_future_use | 2048 | 128 | 0.253 |
| oracle_future_use | 2048 | 256 | 0.253 |

## 8. Future-Use Analysis

### future-use retention_recall (policy x store, mean over active/seeds)

| Policy | 256 | 512 | 1024 | 2048 |
| --- | --- | --- | --- | --- |
| dual_score | 0.057 | 0.103 | 0.129 | 0.253 |
| base_score_only | 0.045 | 0.087 | 0.120 | 0.241 |
| task_affinity | 0.048 | 0.093 | 0.126 | 0.252 |
| random | 0.048 | 0.094 | 0.126 | 0.250 |
| oracle_future_use | 0.057 | 0.103 | 0.129 | 0.253 |

### precision_of_retention

| Policy | 256 | 512 | 1024 | 2048 |
| --- | --- | --- | --- | --- |
| dual_score | 1.000 | 1.000 | 1.000 | 1.000 |
| base_score_only | 1.000 | 1.000 | 1.000 | 1.000 |
| task_affinity | 1.000 | 1.000 | 1.000 | 1.000 |
| random | 1.000 | 1.000 | 1.000 | 1.000 |
| oracle_future_use | 1.000 | 1.000 | 1.000 | 1.000 |

No-pressure controls retained/future counts are in the JSON; the report focuses on the pressured grid where survival selection is measurable.

## 9. Feedback-Loop Diagnosis

`retention_priority` is fed by `access_count` (retrievals + ingest reinforcements). The question: does survival under `dual_score` merely echo 'was retrieved', so that retained future-use facts are explained by retrieval rather than by policy? Evidence in `diagnostics.spearman_vs_future_use` per cell; the JSON holds the full correlation matrix for `retrieval_access_count`, `ingest_reinforcement_count`, `retention_priority`, `task_affinity`, `base_score` vs the offline future-use label. Distinct counters (§5) allow the loop to be separated into retrieval-feedback and ingest-feedback components.

## 10. Saturation Diagnostics

Per-population distributions (min/max/mean/median/p10/p90) of `base_score`, `current_importance`, `retention_priority`, `task_affinity`, `retrieval_access_count`, `ingest_reinforcement_count`, `duplicates` over the buckets retrieved-needed / stored-not-retrieved / evicted / never-retrieved / corrections / obsolete are recorded in every cell (`diagnostics.by_population`). Population sizes confirm whether retained sets are outcomes of the policy or of saturation.

## 11. Correction & Obsolete Safety (identity-safe)

| Policy | store | identity corr. recall | identity obs. retention | new-corr missing turns |
| --- | --- | --- | --- | --- |
| dual_score | 256 | 1.000 | 0.000 | 0.00 |
| dual_score | 512 | 1.000 | 0.000 | 0.00 |
| dual_score | 1024 | 0.867 | 0.000 | 0.00 |
| dual_score | 2048 | 0.967 | 0.000 | 0.00 |
| base_score_only | 256 | 1.000 | 0.000 | 0.00 |
| base_score_only | 512 | 1.000 | 0.000 | 0.00 |
| base_score_only | 1024 | 1.000 | 0.000 | 0.00 |
| base_score_only | 2048 | 0.933 | 0.000 | 0.00 |
| task_affinity | 256 | 1.000 | 0.000 | 0.00 |
| task_affinity | 512 | 1.000 | 0.000 | 0.00 |
| task_affinity | 1024 | 1.000 | 0.000 | 0.00 |
| task_affinity | 2048 | 1.000 | 0.000 | 0.00 |
| random | 256 | 1.000 | 0.000 | 0.00 |
| random | 512 | 1.000 | 0.000 | 0.00 |
| random | 1024 | 1.000 | 0.000 | 0.00 |
| random | 2048 | 1.000 | 0.000 | 0.00 |
| oracle_future_use | 256 | 1.000 | 0.000 | 0.00 |
| oracle_future_use | 512 | 1.000 | 0.000 | 0.00 |
| oracle_future_use | 1024 | 0.867 | 0.000 | 0.00 |
| oracle_future_use | 2048 | 0.967 | 0.000 | 0.00 |

Every E16 cell runs `protect_corrections=True`: a current correction is never evicted while its superseded predecessor still exists, and `protected_capacity_conflict` records when the store budget is too tight to honour that. `identity_obsolete_retention = 0` is the acceptance gate (criterion c4).

## 12. Age Analysis

Future-use retention by age bucket (age = query_turn - source_turn) is stored per cell (`age_analysis`), as are the eviction ages of lost future-use facts (`evicted_age_analysis`). Buckets: 0-99 … 1000+. A policy that only survives young/recent facts will show collapsing retention in late buckets.

## 13. Query-Family Analysis

Identity context/store recall per query family is recorded per cell (`family_metrics`); the JSON holds the full breakdown for every (policy, store, active, seed) cell.

## 14. Oracle Results & Gap

| Policy | store | identity store recall (oracle) | gap vs oracle (low store) |
| --- | --- | --- | --- |
| dual_score | 256 | 0.057 | -0.000 |
| base_score_only | 256 | 0.045 | 0.014 |
| task_affinity | 256 | 0.049 | 0.009 |
| random | 256 | 0.048 | 0.009 |
| oracle_future_use | all | 0.057 | 0.103 | 0.129 | 0.253 | -0.000 |

oracle_gap(policy) = oracle identity store recall − policy future-use retention recall at store 256/512 (low pressure). The oracle is an offline, future-peeking upper bound: a policy is compared on how much of that theoretical headroom it recovers without future information.

## 15. No-Pressure Diagnostic

For store_budget=0, dual_score / task_affinity / oracle_future_use must produce IDENTICAL final store contents (eviction never fires). Fingerprints identical over 3 runs: **True**
- Causal isolation: without store pressure the eviction policy cannot change what is stored; any recall difference in the pressured grid is therefore attributable to survival, not to retrieval/writing.

## 16. Window Sensitivity (task_affinity)

| window | future-use retention rec. | precision | identity store rec. |
| --- | --- | --- | --- |
| 16 | 0.064 | 1.000 | 0.064 |
| 64 | 0.067 | 1.000 | 0.067 |

Per-seed values are in the JSON. 32 is the default window; 16 and 64 are sensitivity only, not production candidates.

## 17. Decision Rule Application

task_affinity advances ONLY if all eight criteria hold vs `dual_score` (the current production default). Numbers below come from the JSON, not from hand-typed values.

- **c1_beats_dual_future_use_recall_j_n_0.03_at_low_store**: FAIL — {'task_affinity': 0.0709, 'dual_score': 0.0802, 'ok': False}
- **c2_precision_does_not_collapse_vs_dual_n_0.15**: PASS — {'task_affinity': 1.0, 'dual_score': 1.0, 'ok': True}
- **c3_identity_correction_recall_j_0.95**: PASS — {'value': 1.0, 'ok': True}
- **c4_identity_obsolete_retention_==_0**: PASS — {'mean': 0.0, 'max': 0, 'ok': True}
- **c5_active_context_cost_unchanged**: PASS — {'task_affinity_ctx': 144.9657, 'dual_ctx': 146.513, 'ok': True}
- **c6_holds_across_seeds_4_of_5**: FAIL — {'seeds_winning': 0, 'seeds_total': 5, 'ok': False}
- **c7_beats_base_score_only**: FAIL — {'task_affinity': 0.0709, 'base_score_only': 0.0662, 'ok': False}
- **c8_closes_oracle_gap**: FAIL — {'gaps': {'dual_score': -0.0003, 'task_affinity': 0.009, 'base_score_only': 0.0137, 'random': 0.0086, 'oracle_future_use': -0.0003}, 'ok': False}

**task_affinity advances: NO.**

rationale: task_affinity vs dual_score future-use retention recall at low store ([256, 512]): 0.0709 vs 0.0802; precision 1.0 vs 1.0; per-seed wins 0/5; oracle store gap task 0.009 vs dual -0.0003.

## 18. Production Default

Retention default stays `dual_score` (`eviction_priority: retention_priority`). `task_affinity` is reported as a failed candidate with its failure mode documented; no silent flip happens.

— Numbers in this report are generated from `e16_retention_selectivity.json` (and, for section 2, from the E15 raw JSON) by `generate_report()`; no hand-typed figures.