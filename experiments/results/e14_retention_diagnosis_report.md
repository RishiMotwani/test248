# E14 Retention Diagnosis Report

Pure diagnosis of why useful facts leave the long-term store. No memory-policy change.

## Configuration

- budgets: [64, 128, 256, 512]
- seeds: [42, 43, 44, 45, 46]
- turns: 400
- scale: 3
- embedding: nomic-embed-text

## 1. Lifecycle Cause-of-Loss (per budget)

| Budget | gt | stored+retr | stored/not | decay | budget | dedupe | super_ok | super_bad | never | other | corr_recall | obs_ret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 86.00 | 43.40 | 5.20 | 31.40 | 0.00 | 4.00 | 2.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.00 |
| 128 | 86.00 | 54.60 | 5.00 | 20.40 | 0.00 | 4.00 | 2.00 | 0.00 | 0.00 | 0.00 | 0.70 | 0.00 |
| 256 | 86.00 | 64.40 | 5.00 | 10.60 | 0.00 | 4.00 | 2.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| 512 | 86.00 | 73.80 | 6.00 | 0.20 | 0.00 | 4.00 | 2.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |

## 2. Dominant Store Loss

| Budget | Dominant store loss | Decay | Store budget | Dedupe | Super bad | Retrieval loss |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | REMOVED_BY_DECAY | 0.365 | 0.000 | 0.046 | 0.000 | 0.0605 |
| 128 | REMOVED_BY_DECAY | 0.237 | 0.000 | 0.046 | 0.000 | 0.0581 |
| 256 | REMOVED_BY_DECAY | 0.123 | 0.000 | 0.046 | 0.000 | 0.0581 |
| 512 | MERGED_BY_DEDUPE | 0.002 | 0.000 | 0.046 | 0.000 | 0.0698 |

## 3. Category Survival (budget 128 reference)

*See JSON for all budgets; the per-category table below uses the first budget bucket.*

## 4. Ablations

### Ablation: no_decay

| Budget | ctx_recall | store_recall | retrieval_loss | corr_recall | obs_ret | decay | budget_evict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 0.893 | 0.917 | 0.024 | 1.0 | 0.0 | 0.0 | 0.0 |
| 128 | 0.893 | 0.917 | 0.024 | 1.0 | 0.0 | 0.0 | 0.0 |
| 256 | 0.893 | 0.917 | 0.024 | 1.0 | 0.0 | 0.0 | 0.0 |
| 512 | 0.917 | 0.917 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 |

### Ablation: no_prune_threshold

| Budget | ctx_recall | store_recall | retrieval_loss | corr_recall | obs_ret | decay | budget_evict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 0.8618 | 0.917 | 0.0552 | 1.0 | 0.0 | 0.0 | 0.0 |
| 128 | 0.8834 | 0.917 | 0.0336 | 1.0 | 0.0 | 0.0 | 0.0 |
| 256 | 0.8882 | 0.917 | 0.0288 | 1.0 | 0.0 | 0.0 | 0.0 |
| 512 | 0.917 | 0.917 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 |

## 5. Write-time Salience vs Query-time Similarity (Ablation D)

| Budget | sal_retrieved | sal_missed | sim_retrieved | sim_missed | spearman |
| --- | --- | --- | --- | --- | --- |
| 64 | 0.9837 | 0.9822 | 0.4864 | 0.4732 | -0.0611 |
| 128 | 0.9834 | 0.9826 | 0.4868 | 0.4722 | -0.0886 |
| 256 | 0.9829 | 0.9826 | 0.4837 | 0.4722 | -0.1019 |
| 512 | 0.9829 | 0.9828 | 0.4791 | 0.4666 | -0.0802 |

## 6. Limitations / Unresolved

1. Lifecycle state is derived from exact fact-text presence; a fact merged into another is classified MERGED_BY_DEDUPE even though its information may still be retrievable in merged form.
2. Store budget is intentionally 4x the active budget (>=4096); store pressure is therefore not expected to bind at this workload scale.
3. Retrieval loss (PRESENT_BUT_NOT_RETRIEVED) is not a store loss and is reported separately.
