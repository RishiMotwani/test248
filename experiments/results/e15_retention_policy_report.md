# E15 Retention Policy Report (Phase 11)

Separating memory **activation** (`current_importance`, decays, retrieval ranking) from long-term **survival** (`retention_priority`, stable, eviction).

## Configuration

- **budgets**: [64, 128, 256, 512]
- **seeds**: [42, 43, 44, 45, 46]
- **turns**: 400
- **scale**: 3
- **store_budget**: 4096
- **embedding_model**: nomic-embed-text
- **matching**: exact answer-token presence
- **policies**: ['hard_threshold', 'soft_decay', 'dual_score', 'no_decay']
- **active_budget_enforced**: True

### Policies

| Policy | retention_mode / eviction_priority |
| --- | --- |
| hard_threshold | `hard_threshold` / `current_importance` |
| soft_decay | `soft_decay` / `current_importance` |
| dual_score | `dual_score` / `retention_priority` |
| no_decay | `hard_threshold` / `current_importance` |

## 1. Lifecycle Cause-of-Loss (primary grid, per budget)

| Policy | Budget | stored+retr | stored/not | decay | budget | dedupe | super_ok | super_bad | corr_recall | obs_ret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hard_threshold | 64 | 43.40 | 5.20 | 31.40 | 0.00 | 4.00 | 2.00 | 0.00 | 0.50 | 0.00 |
| hard_threshold | 128 | 54.60 | 5.00 | 20.40 | 0.00 | 4.00 | 2.00 | 0.00 | 0.70 | 0.00 |
| hard_threshold | 256 | 64.40 | 5.00 | 10.60 | 0.00 | 4.00 | 2.00 | 0.00 | 1.00 | 0.00 |
| hard_threshold | 512 | 73.80 | 6.00 | 0.20 | 0.00 | 4.00 | 2.00 | 0.00 | 1.00 | 0.00 |
| soft_decay | 64 | 72.40 | 7.60 | 0.00 | 0.00 | 4.00 | 2.00 | 0.00 | 1.00 | 0.00 |
| soft_decay | 128 | 74.00 | 6.00 | 0.00 | 0.00 | 4.00 | 2.00 | 0.00 | 1.00 | 0.00 |
| soft_decay | 256 | 74.00 | 6.00 | 0.00 | 0.00 | 4.00 | 2.00 | 0.00 | 1.00 | 0.00 |
| soft_decay | 512 | 74.00 | 6.00 | 0.00 | 0.00 | 4.00 | 2.00 | 0.00 | 1.00 | 0.00 |
| dual_score | 64 | 72.40 | 7.60 | 0.00 | 0.00 | 4.00 | 2.00 | 0.00 | 1.00 | 0.00 |
| dual_score | 128 | 74.00 | 6.00 | 0.00 | 0.00 | 4.00 | 2.00 | 0.00 | 1.00 | 0.00 |
| dual_score | 256 | 74.00 | 6.00 | 0.00 | 0.00 | 4.00 | 2.00 | 0.00 | 1.00 | 0.00 |
| dual_score | 512 | 74.00 | 6.00 | 0.00 | 0.00 | 4.00 | 2.00 | 0.00 | 1.00 | 0.00 |
| no_decay | 64 | 74.00 | 6.00 | 0.00 | 0.00 | 4.00 | 2.00 | 0.00 | 1.00 | 0.00 |
| no_decay | 128 | 74.00 | 6.00 | 0.00 | 0.00 | 4.00 | 2.00 | 0.00 | 1.00 | 0.00 |
| no_decay | 256 | 74.00 | 6.00 | 0.00 | 0.00 | 4.00 | 2.00 | 0.00 | 1.00 | 0.00 |
| no_decay | 512 | 74.00 | 6.00 | 0.00 | 0.00 | 4.00 | 2.00 | 0.00 | 1.00 | 0.00 |

## 2. Final Comparison Table (Policy x Budget)

| Policy | B | Store Recall | Context Recall | Retrieval Loss | Store Tokens | Obs. Retention | Corr. Recall | Mean Ctx Tokens | Below Thresh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hard_threshold | 64 | 0.672 | 0.576 | 0.096 | 613.000 | 0.000 | 0.500 | 61.108 | 0.000 |
| hard_threshold | 128 | 0.774 | 0.705 | 0.069 | 743.600 | 0.000 | 0.700 | 125.702 | 0.000 |
| hard_threshold | 256 | 0.852 | 0.800 | 0.052 | 852.800 | 0.000 | 1.000 | 253.406 | 0.000 |
| hard_threshold | 512 | 0.917 | 0.917 | 0.000 | 981.400 | 0.000 | 1.000 | 509.460 | 0.000 |
| soft_decay | 64 | 0.917 | 0.862 | 0.055 | 983.000 | 0.000 | 1.000 | 61.190 | 0.258 |
| soft_decay | 128 | 0.917 | 0.883 | 0.034 | 983.000 | 0.000 | 1.000 | 125.388 | 0.163 |
| soft_decay | 256 | 0.917 | 0.888 | 0.029 | 983.000 | 0.000 | 1.000 | 253.416 | 0.068 |
| soft_decay | 512 | 0.917 | 0.917 | 0.000 | 983.000 | 0.000 | 1.000 | 509.474 | 0.003 |
| dual_score | 64 | 0.917 | 0.862 | 0.055 | 983.000 | 0.000 | 1.000 | 61.190 | 0.258 |
| dual_score | 128 | 0.917 | 0.883 | 0.034 | 983.000 | 0.000 | 1.000 | 125.388 | 0.163 |
| dual_score | 256 | 0.917 | 0.888 | 0.029 | 983.000 | 0.000 | 1.000 | 253.416 | 0.068 |
| dual_score | 512 | 0.917 | 0.917 | 0.000 | 983.000 | 0.000 | 1.000 | 509.474 | 0.003 |
| no_decay | 64 | 0.917 | 0.893 | 0.024 | 983.000 | 0.000 | 1.000 | 60.950 | 0.000 |
| no_decay | 128 | 0.917 | 0.893 | 0.024 | 983.000 | 0.000 | 1.000 | 125.480 | 0.000 |
| no_decay | 256 | 0.917 | 0.893 | 0.024 | 983.000 | 0.000 | 1.000 | 253.310 | 0.000 |
| no_decay | 512 | 0.917 | 0.917 | 0.000 | 983.000 | 0.000 | 1.000 | 509.230 | 0.000 |

## 3. Activation / Survival summary

| Policy | B | decay_rm | budget_ev | mean_imp | mean_rp | below_thr | strong_rel_keep | cand | sel | revived |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hard_threshold | 64 | 31.400 | 0.000 | 0.524 | 0.987 | 0.000 | 0.000 | 48.600 | 48.400 | 0.000 |
| hard_threshold | 128 | 20.400 | 0.000 | 0.631 | 0.993 | 0.000 | 0.000 | 59.600 | 59.600 | 0.000 |
| hard_threshold | 256 | 10.600 | 0.000 | 0.768 | 0.998 | 0.000 | 0.000 | 69.400 | 69.400 | 0.000 |
| hard_threshold | 512 | 0.200 | 0.000 | 0.925 | 1.000 | 0.000 | 0.000 | 79.800 | 79.800 | 0.000 |
| soft_decay | 64 | 0.000 | 0.000 | 0.538 | 0.990 | 0.258 | 32.200 | 80.000 | 77.600 | 30.000 |
| soft_decay | 128 | 0.000 | 0.000 | 0.639 | 0.995 | 0.163 | 21.400 | 80.000 | 80.000 | 20.400 |
| soft_decay | 256 | 0.000 | 0.000 | 0.783 | 0.999 | 0.068 | 10.600 | 80.000 | 80.000 | 9.600 |
| soft_decay | 512 | 0.000 | 0.000 | 0.925 | 1.000 | 0.003 | 0.200 | 80.000 | 80.000 | 0.200 |
| dual_score | 64 | 0.000 | 0.000 | 0.538 | 0.990 | 0.258 | 32.200 | 80.000 | 77.600 | 30.000 |
| dual_score | 128 | 0.000 | 0.000 | 0.639 | 0.995 | 0.163 | 21.400 | 80.000 | 80.000 | 20.400 |
| dual_score | 256 | 0.000 | 0.000 | 0.783 | 0.999 | 0.068 | 10.600 | 80.000 | 80.000 | 9.600 |
| dual_score | 512 | 0.000 | 0.000 | 0.925 | 1.000 | 0.003 | 0.200 | 80.000 | 80.000 | 0.200 |
| no_decay | 64 | 0.000 | 0.000 | 0.995 | 0.995 | 0.000 | 0.000 | 80.000 | 80.000 | 0.000 |
| no_decay | 128 | 0.000 | 0.000 | 0.998 | 0.998 | 0.000 | 0.000 | 80.000 | 80.000 | 0.000 |
| no_decay | 256 | 0.000 | 0.000 | 0.999 | 0.999 | 0.000 | 0.000 | 80.000 | 80.000 | 0.000 |
| no_decay | 512 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 80.000 | 80.000 | 0.000 |

## 4. Age-bucket survival (budget 128 reference)

*See JSON for every budget. Table below uses budget 128.*

| Policy | bucket | expected | store_present | retrieved |
| --- | --- | --- | --- | --- |
| hard_threshold | 0-49 | 11.4 | 1.0 | 0.7912 |
| hard_threshold | 50-99 | 12.6 | 0.857 | 0.812 |
| hard_threshold | 100-149 | 8.8 | 0.6172 | 0.6018 |
| hard_threshold | 150-199 | 8.0 | 0.6052 | 0.5004 |
| hard_threshold | 200-249 | 12.2 | 0.5566 | 0.5566 |
| hard_threshold | 250-299 | 8.2 | 0.9428 | 0.848 |
| hard_threshold | 300-349 | 11.8 | 0.4982 | 0.4848 |
| hard_threshold | 350-399 | 11.0 | 0.6202 | 0.6202 |
| soft_decay | 0-49 | 11.4 | 1.0 | 0.7912 |
| soft_decay | 50-99 | 12.6 | 0.9184 | 0.8738 |
| soft_decay | 100-149 | 8.8 | 1.0 | 0.9846 |
| soft_decay | 150-199 | 8.0 | 1.0 | 0.867 |
| soft_decay | 200-249 | 12.2 | 1.0 | 0.9364 |
| soft_decay | 250-299 | 8.2 | 1.0 | 0.9052 |
| soft_decay | 300-349 | 11.8 | 0.7732 | 0.7598 |
| soft_decay | 350-399 | 11.0 | 0.9684 | 0.9684 |
| dual_score | 0-49 | 11.4 | 1.0 | 0.7912 |
| dual_score | 50-99 | 12.6 | 0.9184 | 0.8738 |
| dual_score | 100-149 | 8.8 | 1.0 | 0.9846 |
| dual_score | 150-199 | 8.0 | 1.0 | 0.867 |
| dual_score | 200-249 | 12.2 | 1.0 | 0.9364 |
| dual_score | 250-299 | 8.2 | 1.0 | 0.9052 |
| dual_score | 300-349 | 11.8 | 0.7732 | 0.7598 |
| dual_score | 350-399 | 11.0 | 0.9684 | 0.9684 |
| no_decay | 0-49 | 11.4 | 1.0 | 0.7912 |
| no_decay | 50-99 | 12.6 | 0.9184 | 0.8738 |
| no_decay | 100-149 | 8.8 | 1.0 | 0.9846 |
| no_decay | 150-199 | 8.0 | 1.0 | 0.867 |
| no_decay | 200-249 | 12.2 | 1.0 | 0.9364 |
| no_decay | 250-299 | 8.2 | 1.0 | 0.9052 |
| no_decay | 300-349 | 11.8 | 0.7732 | 0.7598 |
| no_decay | 350-399 | 11.0 | 0.9684 | 0.9684 |

## 5. Stress grid (genuine store pressure)

Scale 27, natural store tokens 8272, turns 1200.

| Policy | store_budget | active | ctx_recall | store_recall | retr_loss | actual_store_tokens | eviction_count | decay_rm | corr | obs_ret | below_thr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Stress comparison table as specified:

| Policy | store_budget | natural_store_tokens | actual_store_tokens | eviction_count |
| --- | --- | --- | --- | --- |
| soft_decay | 1024 | 8272 | 1019.8889 | 519.0 |
| soft_decay | 2048 | 8272 | 2042.8889 | 336.1111 |
| soft_decay | 4096 | 8272 | 4088.4444 | 32.4444 |
| dual_score | 1024 | 8272 | 1022.0 | 430.0 |
| dual_score | 2048 | 8272 | 2048.0 | 247.0 |
| dual_score | 4096 | 8272 | 4088.0 | 29.0 |
| hard_threshold | 1024 | 8272 | 1019.8889 | 518.2222 |
| hard_threshold | 2048 | 8272 | 1900.2222 | 199.1111 |
| hard_threshold | 4096 | 8272 | 2443.1111 | 0.0 |

## 6. Answers to the Nine Evaluation Questions

### Q1 — Does separating activation from survival recover useful long-range facts?

soft_decay store_recall means: B64=0.917, B128=0.917, B256=0.917, B512=0.917
hard_threshold store_recall means: B64=0.672, B128=0.774, B256=0.852, B512=0.917
Delta store_recall (soft - hard): B64=+0.245, B128=+0.143, B256=+0.065, B512=+0.000
long_range_recall soft vs hard: B64=0.880/0.580, B128=0.902/0.715, B256=0.908/0.815, B512=0.938/0.938

### Q2 — Does soft decay improve low-budget (64/128) recall?

Budget 64: store_recall soft=0.917 vs hard=0.672 (delta +0.245); context_recall soft=0.862 vs hard=0.576 (delta +0.286).
Budget 128: store_recall soft=0.917 vs hard=0.774 (delta +0.143); context_recall soft=0.883 vs hard=0.705 (delta +0.179).

### Q3 — Does dual_score add anything beyond soft_decay?

Under genuine store pressure (natural tokens 8272), compare soft_decay vs dual_score for store_recall, eviction count and obsolete retention in the stress table above.

### Q4 — Store-size cost?

Store tokens means (soft vs hard): B64=983.000/613.000, B128=983.000/743.600, B256=983.000/852.800, B512=983.000/981.400. fraction_below_pruning_threshold soft vs hard: B64=0.258/0.000, B128=0.163/0.000, B256=0.068/0.000, B512=0.003/0.000

### Q5 — Obsolete retention controlled?

obsolete_retention soft: B64=0.000, B128=0.000, B256=0.000, B512=0.000. hard: B64=0.000, B128=0.000, B256=0.000, B512=0.000

### Q6 — Corrections correct?

correction_recall soft: B64=1.000, B128=1.000, B256=1.000, B512=1.000. hard: B64=0.500, B128=0.700, B256=1.000, B512=1.000

### Q7 — Active-context cost unchanged?

mean_context_tokens soft: B64=61.190, B128=125.388, B256=253.416, B512=509.474. hard: B64=61.108, B128=125.702, B256=253.406, B512=509.460. Context budget is enforced by construction; utilization reported in JSON.

### Q8 — Where does retrieval loss remain?

retrieval_loss soft: B64=0.055, B128=0.034, B256=0.029, B512=0.000. hard: B64=0.096, B128=0.069, B256=0.052, B512=0.000. PRESENT_BUT_NOT_RETRIEVED counts in lifecycle table itemize it.

### Q9 — Real improvement or mere data preservation?

revived-facts (retained-below-threshold AND later retrieved): soft B64=30.000, B128=20.400, B256=9.600, B512=0.200; candidate_count soft B64=80.000, B128=80.000, B256=80.000, B512=80.000. A policy that merely keeps everything scores store_recall high without improving context_recall; compare both columns in section 2.


## 7. Decision Rule Application

**Labels from the observed grid (auto-classified, reviewer-confirmable):**

- `SEPARATING ACTIVATION FROM SURVIVAL IS SUPPORTED`: **SUPPORTED**
    - store_recall_delta: 0.1133
    - context_recall_delta: 0.1382
    - active_context_cost_unchanged: True
    - safety_preserved: True
    - store_growth_manageable: True
    - rationale: soft_decay store_recall=0.917, hard=0.8037; context_recall soft=0.8876, hard=0.7494; mean_context_tokens soft=237.367, hard=237.419; obsolete soft=0.0, hard=0.0; correction soft=1.0, hard=0.8; store tokens soft=983.0, hard=797.7.

- `RETENTION PRIORITY FOR STORE EVICTION IS SUPPORTED`: **SUPPORTED**
    - dual_minus_soft_primary: 0.0
    - dual_minus_soft_tightest_stress_store: -0.0094
    - dual_minus_soft_tightest_stress_context: 0.0308
    - dual_minus_soft_tightest_stress_correction: 0.6666
    - rationale: dual_score store_recall=0.917 vs soft_decay=0.917; under tightest store pressure dual gives store +-0.0094, context +0.0308, correction +0.6666 (archival store_recall tradeoff documented in the stress table).

- `SOFT RETENTION RECOVERS FACTS BUT CREATES A STALENESS TRADEOFF`: **NOT SUPPORTED**
    - obsolete_retention_soft: 0.0
    - obsolete_retention_hard: 0.0
    - rationale: soft retention must not resurrect superseded values; if obsolete_retention grows beyond the hard baseline the tradeoff is real.


*A candidate is made the leading option only if it improves recall (context + store) without inflating active context, preserves correction/obsolete safety, and keeps store growth manageable under genuine pressure. Higher recall alone is not a win.*
