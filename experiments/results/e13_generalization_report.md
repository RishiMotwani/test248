# E13 Generalization and Ablation Report

## Experiment Configuration

- Turns: 400
- Scale: 3
- Budgets: [64, 128, 256, 512, 1024]
- Seeds: [42, 43, 44, 45, 46]
- Embedding: nomic-embed-text
- Methods: N/A

## 1. Generalization Results (5 seeds × 5 budgets)

### Generalization (Phase-8 defaults)

| Budget | Method | ctx_recall | store_recall | retrieval_loss | mean_ctx_tok | max_ctx_tok | util | max_util | store_tok | budget_viol | corr_recall | obs_ret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | adaptive | 0.576 | 0.672 | 0.095 | 61.108 | 64.000 | 0.955 | 1.000 | 613.000 | 0.000 | 0.500 | 0.000 |
| 64 | sliding_window | 0.048 | 0.048 | N/A | 53.000 | 53.000 | 0.828 | 0.828 | 53.000 | 0.000 | 0.000 | 0.000 |
| 64 | memgpt_style | 0.083 | 0.905 | N/A | 64.000 | 64.000 | 1.000 | 1.000 | 1041.000 | 0.000 | 0.000 | 0.000 |
| 64 | summarization_only | 0.083 | 0.083 | N/A | 64.000 | 64.000 | 1.000 | 1.000 | 64.000 | 0.000 | 0.000 | 0.000 |
| 64 | vanilla_rag | 0.905 | 0.905 | N/A | 57.170 | 64.000 | 0.893 | 1.000 | 1041.000 | 0.000 | 0.000 | 1.000 |
| 128 | adaptive | 0.705 | 0.774 | 0.069 | 125.702 | 128.000 | 0.982 | 1.000 | 743.600 | 0.000 | 0.700 | 0.000 |
| 128 | sliding_window | 0.048 | 0.048 | N/A | 123.200 | 123.200 | 0.963 | 0.963 | 123.200 | 0.000 | 0.000 | 0.000 |
| 128 | memgpt_style | 0.905 | 0.905 | N/A | 120.630 | 128.000 | 0.942 | 1.000 | 1041.000 | 0.000 | 0.000 | 1.000 |
| 128 | summarization_only | 0.179 | 0.179 | N/A | 128.000 | 128.000 | 1.000 | 1.000 | 128.000 | 0.000 | 0.000 | 0.000 |
| 128 | vanilla_rag | 0.905 | 0.905 | N/A | 59.950 | 77.000 | 0.468 | 0.602 | 1041.000 | 0.000 | 0.000 | 1.000 |
| 256 | adaptive | 0.800 | 0.852 | 0.053 | 253.406 | 256.000 | 0.990 | 1.000 | 852.800 | 0.000 | 1.000 | 0.000 |
| 256 | sliding_window | 0.048 | 0.048 | N/A | 128.400 | 128.400 | 0.502 | 0.502 | 128.400 | 0.000 | 0.000 | 0.000 |
| 256 | memgpt_style | 0.905 | 0.905 | N/A | 123.630 | 141.000 | 0.483 | 0.551 | 1041.000 | 0.000 | 0.000 | 1.000 |
| 256 | summarization_only | 0.333 | 0.333 | N/A | 255.000 | 255.000 | 0.996 | 0.996 | 255.000 | 0.000 | 0.000 | 0.000 |
| 256 | vanilla_rag | 0.905 | 0.905 | N/A | 59.950 | 77.000 | 0.234 | 0.301 | 1041.000 | 0.000 | 0.000 | 1.000 |
| 512 | adaptive | 0.917 | 0.917 | 0.000 | 509.460 | 512.000 | 0.995 | 1.000 | 981.400 | 0.000 | 1.000 | 0.000 |
| 512 | sliding_window | 0.048 | 0.048 | N/A | 128.400 | 128.400 | 0.251 | 0.251 | 128.400 | 0.000 | 0.000 | 0.000 |
| 512 | memgpt_style | 0.905 | 0.905 | N/A | 123.630 | 141.000 | 0.241 | 0.275 | 1041.000 | 0.000 | 0.000 | 1.000 |
| 512 | summarization_only | 0.583 | 0.583 | N/A | 511.000 | 511.000 | 0.998 | 0.998 | 511.000 | 0.000 | 0.000 | 0.000 |
| 512 | vanilla_rag | 0.905 | 0.905 | N/A | 59.950 | 77.000 | 0.117 | 0.150 | 1041.000 | 0.000 | 0.000 | 1.000 |
| 1024 | adaptive | 0.917 | 0.917 | 0.000 | 983.000 | 983.000 | 0.960 | 0.960 | 983.000 | 0.000 | 1.000 | 0.000 |
| 1024 | sliding_window | 0.048 | 0.048 | N/A | 128.400 | 128.400 | 0.125 | 0.125 | 128.400 | 0.000 | 0.000 | 0.000 |
| 1024 | memgpt_style | 0.905 | 0.905 | N/A | 123.630 | 141.000 | 0.121 | 0.138 | 1041.000 | 0.000 | 0.000 | 1.000 |
| 1024 | summarization_only | 0.893 | 0.893 | N/A | 1019.000 | 1019.000 | 0.995 | 0.995 | 1019.000 | 0.000 | 0.000 | 1.000 |
| 1024 | vanilla_rag | 0.905 | 0.905 | N/A | 59.950 | 77.000 | 0.059 | 0.075 | 1041.000 | 0.000 | 0.000 | 1.000 |


## 2. Store-Capacity Ablation (Ablation A)

### Store-Capacity Policy Ablation

*See detailed JSON for full breakdown.*

- **Coupled/Matched** (store=active): Store capped at active budget.
- **Separated 4x** (Phase-8): Store up to 4x active budget.
- **Unbounded** (0): No hard store cap.


## 3. Retrieval-Policy Ablation (Ablation B)

### Retrieval Profile Ablation

*See detailed JSON for full breakdown.*

- **Phase-8** (0.15/0.85): Query-first.
- **Legacy Phase-7** (0.6/0.4): Importance-dominant.
- **Pure Similarity** (0/1.0): Query-only.
- **Pure Importance** (1.0/0): History-only.


## 4. Embedding vs Lexical Sensitivity

### Retrieval Mode Ablation

*See detailed JSON for full breakdown.*

- **Embeddings**: Production path; higher similarity discrimination.
- **Lexical**: Fallback path; lower discrimination.


## 5. Limitations and Unresolved Questions

1. **Workload scale**: Current coding workload (scale=3, ~84 facts, ~1024 natural store tokens) may not create genuine store pressure at 4x store budget (4096+). The unbounded policy may not differ from 4x if natural store < 4096.
2. **Embedding variance**: Only nomic-embed-text tested. Other embedding models may yield different similarity distributions.
3. **Query coverage**: Query families derived from existing qtypes; may not cover all realistic coding question types.
4. **Decay not varied**: Decay policy held fixed per Phase 9 directive. Store pressure interacts with decay.
5. **Budget range**: 64-token budget may be too small for meaningful retrieval; 1024 may exceed workload needs.
