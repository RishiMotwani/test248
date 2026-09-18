# E17 Coding Capability Report (Phase 13)

Does adaptive memory help a real coding model complete long-horizon repository-editing tasks better than simpler historical-context managers when the historical context is capped at a fixed budget? Primary outcome: **hidden-test patch pass**.

## 1. Purpose & Research Question

Every earlier experiment measured memory recall or injected-context answerability. E17 measures the downstream quantity the memory system exists to support: whether a coding model produces, in a real workspace, a patch that passes deterministic hidden tests after a ~600-turn conversation whose constraints no longer fit any small context window. Historical-fact recall is diagnostic only and never the dependent variable.

## 2. Hypothesis & Falsification Criteria

H1: at a fixed historical-context budget, the adaptive pipeline (`dual_score` retention + similarity/importance retrieval + correction-aware compression) yields a higher hidden-test pass rate than `raw_clipped`, `sliding_window`, `llm_summarization` and `vanilla_rag`. Falsified if any paired 95% bootstrap CI on the mean success difference has a lower bound <= 0, or if the pilot gates do not all pass.

## 3. Methodology / Controlled Comparison

One code path runs every arm (`experiments/coding_benchmark.py`): the method builds a historical memory representation, retrieves a context string bounded by the budget, and the harness assembles the prompt, calls the coding model, writes the model's complete-file edit blocks into a fresh workspace copy (a unified-diff fallback is kept for robustness) and runs the hidden tests. The model, prompt scaffold, temperature (0.1), retry policy (initial + one repair) and tokenizer are identical across arms; only the history mechanism differs.

## 4. Task Suite Description

| task | title | critical facts | corrections | negative constraints |
| --- | --- | --- | --- | --- |
| cache_readonly | Implement a cache write helper (early architectural constraint) | 2 | 1 | 1 |
| user_ids | Implement a user storage key (correction / supersession) | 2 | 1 | 0 |
| write_retry | Implement a record submitter (negative constraint) | 2 | 0 | 2 |
| validation_pure | Add strict config validation (long-range purity constraint) | 3 | 1 | 1 |

Ingestion mode: `oracle_pre_extracted` (oracle-pre-extracted): E17 measures retention + retrieval + context allocation, not extraction. Each task's workspace is a small real package; its hidden test fails if a buried constraint is violated and is never seen by the model.

## 5. Historical Dependence Design

Each task spreads critical facts across a ~600-turn transcript among distractors; >300-turn-old facts and explicit corrections are present. The `no_history` diagnostic (section 20) quantifies whether the task is genuinely history-dependent. Hidden-test pass requires the model to respect constraints that the current workspace alone does not make obvious.

## 6. Memory Methods Under Test

| method | historical-context mechanism |
| --- | --- |
| raw_clipped | newest raw turns that fit the budget; no retrieval |
| sliding_window | last 10 raw turns (production `SlidingWindowBaseline`) |
| llm_summarization | running summary by the same model every 50 turns |
| vanilla_rag | static fact store, pure cosine top-k fit to budget |
| adaptive | production pipeline, `dual_score` retention, `protect_corrections=True` |

Diagnostics (not competitors): `no_history`, `full_context` (unconstrained upper bound), `direct_history` (oracle gold-fact context).

## 7. Fixed Budgets & Adaptive Allocation Policy

Historical budgets: [256, 512, 1024] tokens (shared word-count tokenizer `shared_word_count`; no exact tokenizer available offline). For adaptive, `injection_token_limit = budget` and `memory_store_token_budget = max(4096, budget*4)`, mirroring the repo convention; this is documented, not tuned.

## 8. Coding Model, Prompt & Harness Determinism

Coding model: `llama3.1:8b` via Ollama (`temperature=0.1`, `num_ctx=8192`, `keep_alive=30m`, `num_predict=700`). Prompt = instructions + current workspace files (visible files only) + current task + historical engineering context. The model returns complete-file blocks that the harness writes verbatim (the resulting change is captured as a git diff); a diff fallback exists. Workspace reset, edit application and hidden tests are deterministic (gates 8-9).

## 9. Evaluation Metrics & Failure Taxonomy

Primary: `final_success` (hidden test passes). Secondary: `first_pass_success`, attempts. Diagnostics: `critical_fact_recall`, `correction_recall`, `negative_constraint_recall`, `long_range_fact_recall`, `obsolete_fact_exposure`. Failure classes (deterministic precedence): MEMORY_MISS, RETRIEVAL_MISS, CONTEXT_OVERFLOW, PATCH_INVALID, OBSOLETE_INFORMATION_USED, CORRECTION_MISSED, CODING_ERROR, HIDDEN_TEST_FAILURE, OTHER.

## 10. Configuration & Reproducibility

- **mode**: full
- **tasks**: ['cache_readonly', 'user_ids', 'write_retry', 'validation_pure']
- **seeds**: [1, 2, 3]
- **budgets**: [256, 512, 1024]
- **methods**: ['raw_clipped', 'sliding_window', 'llm_summarization', 'vanilla_rag', 'adaptive']
- **model**: llama3.1:8b
- **embedding_model**: nomic-embed-text
- **use_embeddings**: True
- **max_output_tokens**: 700
- **temperature**: 0.1
- **tokenizer**: shared_word_count
- **ingestion**: oracle_pre_extracted
- **adaptive**: {'retention_mode': 'dual_score', 'protect_corrections': True, 'injection_token_limit': 'historical_budget', 'memory_store_token_budget': 'max(4096, budget*4)'}
- **generated_at**: 1789715822.7246826

## 11. Results: Success Rate by Method × Budget

| method | 256 | 512 | 1024 | overall |
| --- | --- | --- | --- | --- |
| raw_clipped | 0.750 | 0.667 | 0.750 | 0.722 |
| sliding_window | 0.667 | 0.750 | 0.750 | 0.722 |
| llm_summarization | 0.750 | 0.750 | 0.750 | 0.750 |
| vanilla_rag | 0.667 | 0.750 | 0.750 | 0.722 |
| adaptive | 0.750 | 0.750 | 0.750 | 0.750 |
| no_history | N/A | N/A | N/A | N/A |
| full_context | N/A | N/A | N/A | N/A |
| direct_history | N/A | N/A | N/A | N/A |

## 12. Results: First-Pass vs Final Success

| method | budget | first-pass | final | repair recovery | runs |
| --- | --- | --- | --- | --- | --- |
| raw_clipped | 256 | 0.583 | 0.750 | 0.167 | 12 |
| raw_clipped | 512 | 0.667 | 0.667 | 0.000 | 12 |
| raw_clipped | 1024 | 0.750 | 0.750 | 0.000 | 12 |
| sliding_window | 256 | 0.667 | 0.667 | 0.000 | 12 |
| sliding_window | 512 | 0.750 | 0.750 | 0.000 | 12 |
| sliding_window | 1024 | 0.667 | 0.750 | 0.083 | 12 |
| llm_summarization | 256 | 0.750 | 0.750 | 0.000 | 12 |
| llm_summarization | 512 | 0.583 | 0.750 | 0.167 | 12 |
| llm_summarization | 1024 | 0.667 | 0.750 | 0.083 | 12 |
| vanilla_rag | 256 | 0.667 | 0.667 | 0.000 | 12 |
| vanilla_rag | 512 | 0.500 | 0.750 | 0.250 | 12 |
| vanilla_rag | 1024 | 0.583 | 0.750 | 0.167 | 12 |
| adaptive | 256 | 0.500 | 0.750 | 0.250 | 12 |
| adaptive | 512 | 0.500 | 0.750 | 0.250 | 12 |
| adaptive | 1024 | 0.583 | 0.750 | 0.167 | 12 |

## 13. Paired Comparisons & Effect Sizes (adaptive − other)

| vs | pairs | adaptive | other | mean diff | diff 95% CI | adaptive 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| raw_clipped | 36 | 0.750 | 0.722 | 0.028 | [0.0, 0.083] | [0.611, 0.889] |
| sliding_window | 36 | 0.750 | 0.722 | 0.028 | [0.0, 0.083] | [0.611, 0.889] |
| llm_summarization | 36 | 0.750 | 0.750 | 0.000 | [0.0, 0.0] | [0.611, 0.889] |
| vanilla_rag | 36 | 0.750 | 0.722 | 0.028 | [0.0, 0.083] | [0.611, 0.889] |

## 14. Historical-Context Usage & Budget Pressure

| method | budget | mean historical tokens | mean total prompt tokens |
| --- | --- | --- | --- |
| raw_clipped | 256 | 248.500 | 532.250 |
| raw_clipped | 512 | 507.083 | 790.833 |
| raw_clipped | 1024 | 1018.667 | 1302.417 |
| sliding_window | 256 | 120.083 | 403.833 |
| sliding_window | 512 | 120.083 | 403.833 |
| sliding_window | 1024 | 120.083 | 403.833 |
| llm_summarization | 256 | 187.000 | 470.750 |
| llm_summarization | 512 | 295.083 | 578.833 |
| llm_summarization | 1024 | 413.583 | 697.333 |
| vanilla_rag | 256 | 96.500 | 380.250 |
| vanilla_rag | 512 | 96.500 | 380.250 |
| vanilla_rag | 1024 | 96.500 | 380.250 |
| adaptive | 256 | 96.500 | 380.250 |
| adaptive | 512 | 96.500 | 380.250 |
| adaptive | 1024 | 96.500 | 380.250 |

## 15. Diagnostic: Critical-Fact Recall

| method | 256 | 512 | 1024 |
| --- | --- | --- | --- |
| adaptive | 1.000 | 1.000 | 1.000 |
| llm_summarization | 0.500 | 0.375 | 0.250 |
| raw_clipped | 0.000 | 0.000 | 0.000 |
| sliding_window | 0.000 | 0.000 | 0.000 |
| vanilla_rag | 1.000 | 1.000 | 1.000 |

## 16. Diagnostic: Correction & Negative-Constraint Recall

### correction_recall

| method | 256 | 512 | 1024 |
| --- | --- | --- | --- |
| adaptive | 0.000 | 0.000 | 0.000 |
| llm_summarization | 0.333 | 0.333 | 0.111 |
| raw_clipped | 0.000 | 0.000 | 0.000 |
| sliding_window | 0.000 | 0.000 | 0.000 |
| vanilla_rag | 0.000 | 0.000 | 0.000 |

### negative_constraint_recall

| method | 256 | 512 | 1024 |
| --- | --- | --- | --- |
| adaptive | 1.000 | 1.000 | 1.000 |
| llm_summarization | 0.611 | 0.333 | 0.389 |
| raw_clipped | 0.000 | 0.000 | 0.000 |
| sliding_window | 0.000 | 0.000 | 0.000 |
| vanilla_rag | 1.000 | 1.000 | 1.000 |

## 17. Diagnostic: Long-Range Recall & Obsolete Exposure

### long_range_fact_recall

| method | 256 | 512 | 1024 |
| --- | --- | --- | --- |
| adaptive | 1.000 | 1.000 | 1.000 |
| llm_summarization | 0.583 | 0.500 | 0.333 |
| raw_clipped | 0.000 | 0.000 | 0.000 |
| sliding_window | 0.000 | 0.000 | 0.000 |
| vanilla_rag | 1.000 | 1.000 | 1.000 |

### obsolete_fact_exposure

| method | 256 | 512 | 1024 |
| --- | --- | --- | --- |
| adaptive | 1.000 | 1.000 | 1.000 |
| llm_summarization | 0.000 | 0.111 | 0.111 |
| raw_clipped | 0.000 | 0.000 | 0.000 |
| sliding_window | 0.000 | 0.000 | 0.000 |
| vanilla_rag | 1.000 | 1.000 | 1.000 |

## 18. Failure Taxonomy Distribution

| method | budget | failure classes |
| --- | --- | --- |
| raw_clipped | 256 | {'RETRIEVAL_MISS': 3, 'SUCCESS': 9} |
| raw_clipped | 512 | {'RETRIEVAL_MISS': 4, 'SUCCESS': 8} |
| raw_clipped | 1024 | {'RETRIEVAL_MISS': 3, 'SUCCESS': 9} |
| sliding_window | 256 | {'MEMORY_MISS': 4, 'SUCCESS': 8} |
| sliding_window | 512 | {'MEMORY_MISS': 3, 'SUCCESS': 9} |
| sliding_window | 1024 | {'MEMORY_MISS': 3, 'SUCCESS': 9} |
| llm_summarization | 256 | {'MEMORY_MISS': 3, 'SUCCESS': 9} |
| llm_summarization | 512 | {'MEMORY_MISS': 3, 'SUCCESS': 9} |
| llm_summarization | 1024 | {'MEMORY_MISS': 3, 'SUCCESS': 9} |
| vanilla_rag | 256 | {'OBSOLETE_INFORMATION_USED': 4, 'SUCCESS': 8} |
| vanilla_rag | 512 | {'OBSOLETE_INFORMATION_USED': 3, 'SUCCESS': 9} |
| vanilla_rag | 1024 | {'OBSOLETE_INFORMATION_USED': 3, 'SUCCESS': 9} |
| adaptive | 256 | {'OBSOLETE_INFORMATION_USED': 3, 'SUCCESS': 9} |
| adaptive | 512 | {'OBSOLETE_INFORMATION_USED': 3, 'SUCCESS': 9} |
| adaptive | 1024 | {'OBSOLETE_INFORMATION_USED': 3, 'SUCCESS': 9} |

## 19. Context-Overflow / Patch-Valid / Failure Modes

- Runs with `CONTEXT_OVERFLOW`: **0**.
- Runs whose model output contained no valid patch: **0**.
- Full per-run records (including the produced patch) are in the JSON.

## 20. No-History & Full/Direct-Context Diagnostics

| task | no_history | full_context | direct_history |
| --- | --- | --- | --- |
| cache_readonly | True | False | True |
| user_ids | False | False | True |
| write_retry | True | False | True |
| validation_pure | False | False | True |

Gold-patch harness sanity: 4/4 tasks pass when the gold patch is supplied.

## 21. Pilot Gates (9/9 required)

| gate | passed | detail |
| --- | --- | --- |
| budget_pressure | PASS | {"growth": {"raw_clipped": {"tokens_lo": 248.5, "tokens_hi": 1018.6667}, "adaptive": {"tokens_lo": 96.5, "tokens_hi": 96.5}, "sliding_window": {"tokens_lo": 120.0833, "tokens_hi": 120.0833}, "llm_summarization": {"tokens_lo": 187.0, "tokens_hi": 413.5833}, "vanilla_rag": {"tokens_lo": 96.5, "tokens_hi": 96.5}}, "raw_grows": true, "history_exceeds_max_budget": true, "some_run_fills_60pct": true} |
| workspace_independence | PASS | {"fingerprints": {"cache_readonly": "aae91f27cf435925", "user_ids": "2268c6c86e0027c2", "write_retry": "94a6c00e4ed884e9", "validation_pure": "9cb2c5e4ad8d4ce3"}} |
| methods_differ | PASS | {"max_distinct_contexts": 4} |
| real_summarization | PASS | {"summary_update_calls": 432} |
| adaptive_production_path | PASS | {"adaptive_runs": 36} |
| no_leakage | PASS | {"leaking_runs": 0} |
| history_dependence | PASS | {"history_dependent_tasks": 2, "per_task": {"cache_readonly": {"no_history_success": true, "oracle_direct_history_success": true, "history_dependent": false}, "user_ids": {"no_history_success": false, "oracle_direct_history_success": true, "history_dependent": true}, "write_retry": {"no_history_success": true, "oracle_direct_history_success": true, "history_dependent": false}, "validation_pure": { |
| test_determinism | PASS | {"checks": [{"task_id": "cache_readonly", "run1_pass": false, "run2_pass": false, "same_pass_fail": true, "same_output": false, "deterministic": true}, {"task_id": "user_ids", "run1_pass": false, "run2_pass": false, "same_pass_fail": true, "same_output": false, "deterministic": true}, {"task_id": "write_retry", "run1_pass": false, "run2_pass": false, "same_pass_fail": true, "same_output": false, " |
| patch_determinism | PASS | {"check": {"ok": true, "results": [{"applied": true, "passed": true}, {"applied": true, "passed": true}], "task_id": "cache_readonly", "method": "raw_clipped"}} |

**All gates passed: True.**

## 22. Threats to Validity & Limitations

- Word-count tokenizer approximates the model's real tokenization; budgets are therefore approximate (documented in the manifest).
- Small task count and seeds imply very wide success-rate CIs; absence of significance is not evidence of equivalence.
- Task suite is synthetic-but-real-code; findings may not transfer to large repositories.
- `full_context` is a context-unconstrained diagnostic, not a fixed-budget competitor.
- Ollama-served `llama3.1:8b` at temperature 0.1 is not deterministic across runs; per-run success is the unit and the grid is resumable.

## 23. Conclusion & Evidence-Based Verdict

- Mode: **full**.
- Gates all passed: **True**.
- adaptive_advances: **False**.

rationale: adaptive_advances requires every paired 95% CI lower bound vs raw_clipped / sliding_window / llm_summarization / vanilla_rag to exceed 0 (adaptive strictly better) AND all pilot gates to pass.

A positive verdict requires every paired adaptive-vs-baseline 95% CI to exclude 0 in adaptive's favour; otherwise the honest conclusion is that adaptive did not demonstrate an advantage on this benchmark.

## 24. Reproduction / Artifacts / Provenance

- JSON: `e17_coding_capability.json` (records include per-run patches).
- generated_at: 1789715822.7246826
- tokenizer: `shared_word_count`; model: `llama3.1:8b`; embedding: `nomic-embed-text`
- work dir: `experiments/results/e17_work/` (gitignored)

— All numbers are generated from the E17 JSON by `generate_report()`; no hand-typed figures.