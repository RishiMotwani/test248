# E19 Coding-Generalization Report (Phase 15)
## 1. Purpose & Research Question

After Phase 14 made explicit supersession authoritative at the consolidation boundary (D35) and Phase 15 fixed a second, adjacent defect (D36: a supersession replaced the fact text but could keep the obsolete predecessor's embedding, so retrieval continued to rank the corrected fact by obsolete semantics), E19 asks: **does the corrected adaptive memory system generalize to a broader set of genuinely history-dependent long-horizon coding tasks better than simpler context-management strategies under the same historical-context budget?** The dependent variable is whether the produced patch passes deterministic hidden tests.
## 2. Causal Chain & What Changed Since E18 (D36 / D37)

E18 re-validated the consolidation boundary after the Phase 14 fix (offline identity gate passed; correction_recall == 1.0, obsolete_exposure == 0.0). Phase 15 hardens the wire between consolidation and retrieval: `_replace_with_supersession` now stores the correcting statement's embedding when it arrives and otherwise drops the stale predecessor embedding, so the surviving memory always embeds its current text (D36). E19 then tests this corrected system on a wider task set with strict primary / negative-control separation (D37): only the three genuinely history-dependent primary tasks may contribute to the adaptive-advances decision. config_contract was calibrated but excluded: probing showed the required explicit-empty-string nuance is not reliably expressible by llama3.1:8b even with oracle history (0/3), so it carries no memory signal at this model; its fixture is retained for future stronger models.
## 3. Design: Primary vs Negative-Control Split (D37)

- **Primary tasks** (history-dependent; the paired/verdict analysis uses only these): user_ids, validation_pure, transaction_atomicity.
- **Negative-control tasks** (harness sanity only; recorded and reported, never used in paired comparisons or the adaptive-advances verdict): cache_readonly, write_retry.
- **Fixture-only task**: config_contract (authored and calibrated this phase, but excluded from the primary set — llama3.1:8b cannot reliably express its explicit-empty-string fact even with oracle history; probing: 0/3 oracle passes across four fixture designs). Its fixture is committed for future stronger models.
- Reasoning: a generalization claim must be earned on tasks whose constraints genuinely require the buried history; tasks that can be solved from the workspace alone would measure coding ability, not memory. The negative controls keep the environment honest (a correctly-built harness passes them with the gold patch).
## 4. Task Suite

| task | group | title | critical facts | corrections | negative constraints |
| --- | --- | --- | --- | --- | --- |
| user_ids | primary | Implement a user storage key (correction / supersession) | 2 | 1 | 0 |
| validation_pure | primary | Add strict config validation (long-range purity constraint) | 3 | 1 | 1 |
| transaction_atomicity | primary | Implement an atomic two-step submit (all-or-nothing + no-retry) | 4 | 1 | 2 |
| cache_readonly | negative-control | Implement a cache write helper (early architectural constraint) | 2 | 1 | 1 |
| write_retry | negative-control | Implement a record submitter (negative constraint) | 2 | 0 | 2 |

## 5. Historical Dependence Design

Each task spreads critical facts across a ~600-turn deterministic transcript among distractors (>=3 topic families), with >=2 critical facts, >=1 correction or negative constraint, >=1 fact >300 turns old, and >=1 distractor thread. user_ids buries the opaque-string-cast constraint (cast ids no longer); validation_pure buries the pure-validation constraint (no audit I/O on the critical path); transaction_atomicity buries the all-or-nothing rollback rule (correction turn 505) and the no-retry rule (turn 310). No fact text uses IMPORTANT/CRITICAL marker language; fact_ids are identity-safe; histories are generated from structured data with no LLM calls. The task prompts never mention the critical historical rules; the no_history diagnostic (section 20) quantifies dependence.
## 6. Memory Methods Under Test

| method | historical-context mechanism |
| --- | --- |
| raw_clipped | newest raw turns that fit the budget; no retrieval |
| sliding_window | last 10 raw turns (production `SlidingWindowBaseline`) |
| llm_summarization | running summary by the same model every 50 turns |
| vanilla_rag | static fact store, pure cosine top-k fit to budget |
| adaptive | the production pipeline, `dual_score` retention, `protect_corrections=True` |

Exactly one adaptive arm (the actual production pipeline); no adaptive_v2 / tuned / plus variants (D37 / spec).Diagnostics (not competitors): no_history, full_context (context-unconstrained upper bound), direct_history (oracle gold-fact context).
## 7. Fixed Budgets & Adaptive Allocation Policy

Historical budgets: [256, 512] tokens (shared word-count tokenizer `shared_word_count`). For adaptive, `injection_token_limit = budget` and `memory_store_token_budget = max(4096, budget*4)`, mirroring the repo convention; this is documented, not tuned. Per-run token accounting separates historical_context_tokens, workspace_context_tokens, task_prompt_tokens and total_prompt_tokens.
## 8. Coding Model, Prompt & Harness

Coding model `llama3.1:8b` via Ollama (`temperature=0.1`, `num_ctx=8192`, `keep_alive=30m`, `num_predict=700`); embeddings `nomic-embed-text` (no silent fallback; model availability is verified before any run). The harness (`experiments/coding_benchmark.py`, unchanged) owns workspace reset, prompt construction, complete-file edit-block application (unified diff fallback), `git apply --check`, the repair-attempt policy (initial + one repair), hidden-test execution, failure classification and token accounting.
## 9. Evaluation Metrics & Failure Taxonomy

Primary: `final_success` and `first_pass_success`. Diagnostics: critical_fact_recall, correction_recall, negative_constraint_recall, long_range_fact_recall, obsolete_fact_exposure. Failure classes (deterministic precedence): MEMORY_MISS, RETRIEVAL_MISS, CONTEXT_OVERFLOW, PATCH_INVALID, CODING_ERROR, HIDDEN_TEST_FAILURE, OBSOLETE_INFORMATION_USED, CORRECTION_MISSED, OTHER (no new class added).
## 10. Configuration & Reproducibility

- **mode**: pilot
- **tasks**: ['user_ids', 'validation_pure', 'transaction_atomicity', 'cache_readonly', 'write_retry']
- **primary_tasks**: ['user_ids', 'validation_pure', 'transaction_atomicity']
- **negative_control_tasks**: ['cache_readonly', 'write_retry']
- **seeds**: [1, 2]
- **budgets**: [256, 512]
- **methods**: ['raw_clipped', 'sliding_window', 'llm_summarization', 'vanilla_rag', 'adaptive']
- **grid_cells**: 60
- **model**: llama3.1:8b
- **endpoint**: http://localhost:11434
- **embedding_model**: nomic-embed-text
- **use_embeddings**: True
- **max_output_tokens**: 700
- **temperature**: 0.1
- **tokenizer**: shared_word_count
- **ingestion**: oracle_pre_extracted
- **adaptive**: {'retention_mode': 'dual_score', 'protect_corrections': True, 'injection_token_limit': 'historical_budget', 'memory_store_token_budget': 'max(4096, budget*4)'}
- **generated_at**: 1789733915.8019342

## 11. Grids: Pilot & Full

- Pilot: 3 primary x 2 seeds x 2 budgets x 5 methods = **60 runs**.
- Full: primary 135 + negative 90 = **225 runs**. Every run is persisted immediately under `task_id:seed:budget:method`; the grid is resumable and completed results are never deleted.
## 12. Pilot Gates A-J

| gate | passed | detail |
| --- | --- | --- |
| embedding_consistency | PASS | {"cells": 27, "embedding_consistent_cells": 27, "failing_cells": []} |
| correction_identity | PASS | {"cells": 27, "identity_ok_cells": 27, "failing_cells": []} |
| history_dependence | FAIL | {"per_task": {"user_ids": {"no_history_success": true, "no_history_successes": 2, "oracle_direct_history_success": true, "oracle_direct_history_successes": 3}, "validation_pure": {"no_history_success": false, "no_history_successes": 0, "oracle_direct_history_success": true, "oracle_direct_history_successes": 3}, "transaction_atomicity": {"no_history_success": true, "no_history_successes": 1, "oracle_direct_history_success": true, "oracle_direct_history_successes": 2}}} |
| method_separation | PASS | {"max_distinct_contexts": 5} |
| no_leakage | PASS | {"leaking_runs": 0} |
| gold_passes | PASS | {"gold_checks": [{"task_id": "user_ids", "final_success": true, "patch_applied": true}, {"task_id": "validation_pure", "final_success": true, "patch_applied": true}, {"task_id": "transaction_atomicity", "final_success": true, "patch_applied": true}, {"task_id": "cache_readonly", "final_success": true, "patch_applied": true}, {"task_id": "write_retry", "final_success": true, "patch_applied": true}]} |
| budget_pressure | PASS | {"history_exceeds_max_budget": true, "raw_grows_with_budget": true, "some_run_fills_60pct": true} |
| real_summarization | PASS | {"summary_update_calls": 144} |
| adaptive_production_path | PASS | {"adaptive_runs": 12} |
| unit_tests_pass | PASS | {"unit_tests": {"exit_code": 0, "passed": 143, "exit_ok": true, "stdout_tail": "........................................................................ [ 50%]\n.......................................................................  [100%]\n143 passed in 12.89s\n"}} |

**All gates passed: False** (verdict section 16).
## 13. Pilot Results: Success Rate by Method x Budget (PILOT)

The full grid has not been run yet; these are PILOT numbers and no production decision is made from them.
| method | 256 | 512 | overall |
| --- | --- | --- | --- |
| raw_clipped | 0.500 | 0.667 | 0.583 |
| sliding_window | 0.167 | 0.333 | 0.250 |
| llm_summarization | 0.667 | 0.500 | 0.583 |
| vanilla_rag | 0.667 | 0.667 | 0.667 |
| adaptive | 1.000 | 1.000 | 1.000 |

## 14. Full Grid: Results by Method x Budget (primary)

Primary-task records only (E19 schema; see JSON).

| method | 256 | 512 | overall |
| --- | --- | --- | --- |
| raw_clipped | 0.500 | 0.667 | 0.583 |
| sliding_window | 0.167 | 0.333 | 0.250 |
| llm_summarization | 0.667 | 0.500 | 0.583 |
| vanilla_rag | 0.667 | 0.667 | 0.667 |
| adaptive | 1.000 | 1.000 | 1.000 |

## 15. Full Grid: First-Pass vs Final Success

| method | budget | first-pass | final | runs |
| --- | --- | --- | --- | --- |
| raw_clipped | 256 | 0.333 | 0.500 | 6 |
| raw_clipped | 512 | 0.333 | 0.667 | 6 |
| sliding_window | 256 | 0.167 | 0.167 | 6 |
| sliding_window | 512 | 0.333 | 0.333 | 6 |
| llm_summarization | 256 | 0.667 | 0.667 | 6 |
| llm_summarization | 512 | 0.333 | 0.500 | 6 |
| vanilla_rag | 256 | 0.333 | 0.667 | 6 |
| vanilla_rag | 512 | 0.500 | 0.667 | 6 |
| adaptive | 256 | 0.833 | 1.000 | 6 |
| adaptive | 512 | 1.000 | 1.000 | 6 |

## 16. Paired Comparisons & Bootstrap CIs (adaptive - baseline)

Paired on (task, seed, budget) over primary records; paired 95% percentile bootstrap CIs on the mean within-pair difference.

| vs | pairs | adaptive | baseline | mean diff | diff 95% CI | adaptive 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| raw_clipped | 12 | 1.000 | 0.583 | 0.417 | [0.167, 0.667] | [1.0, 1.0] |
| sliding_window | 12 | 1.000 | 0.250 | 0.750 | [0.5, 0.917] | [1.0, 1.0] |
| llm_summarization | 12 | 1.000 | 0.583 | 0.417 | [0.167, 0.667] | [1.0, 1.0] |
| vanilla_rag | 12 | 1.000 | 0.667 | 0.333 | [0.083, 0.667] | [1.0, 1.0] |

**Predeclared criterion:** positive paired advantage with the bootstrap 95% CI lower bound > 0 against every baseline; otherwise `adaptive_advances = false`.

- gates_all_passed: **False**
- adaptive_beats_all_baselines: **True**
- **adaptive_advances: False**

rationale: Pilot gates did not all pass; the grid is not a valid experiment and adaptive cannot be said to advance.
## 17. Verdict

**PILOT verdict.** The full grid is not complete; no production conclusion is declared. The pilot gates and paired numbers above are directional only.
## 18. Diagnostics: Recall & Obsolete Exposure (primary)

| method | metric | mean (budget 256) | mean (512) | mean (1024) |
| --- | --- | --- | --- | --- |
| raw_clipped | critical_fact_recall | 0.000 | 0.000 |
| sliding_window | critical_fact_recall | 0.000 | 0.000 |
| llm_summarization | critical_fact_recall | 0.722 | 0.861 |
| vanilla_rag | critical_fact_recall | 1.000 | 1.000 |
| adaptive | critical_fact_recall | 1.000 | 1.000 |
| raw_clipped | correction_recall | 0.000 | 0.000 |
| sliding_window | correction_recall | 0.000 | 0.000 |
| llm_summarization | correction_recall | 0.500 | 0.667 |
| vanilla_rag | correction_recall | 0.000 | 0.000 |
| adaptive | correction_recall | 1.000 | 1.000 |
| raw_clipped | negative_constraint_recall | 0.000 | 0.000 |
| sliding_window | negative_constraint_recall | 0.000 | 0.000 |
| llm_summarization | negative_constraint_recall | 0.875 | 1.000 |
| vanilla_rag | negative_constraint_recall | 1.000 | 1.000 |
| adaptive | negative_constraint_recall | 1.000 | 1.000 |
| raw_clipped | long_range_fact_recall | 0.000 | 0.000 |
| sliding_window | long_range_fact_recall | 0.000 | 0.000 |
| llm_summarization | long_range_fact_recall | 1.000 | 1.000 |
| vanilla_rag | long_range_fact_recall | 1.000 | 1.000 |
| adaptive | long_range_fact_recall | 1.000 | 1.000 |
| raw_clipped | obsolete_fact_exposure | 0.000 | 0.000 |
| sliding_window | obsolete_fact_exposure | 0.000 | 0.000 |
| llm_summarization | obsolete_fact_exposure | 0.000 | 0.000 |
| vanilla_rag | obsolete_fact_exposure | 1.000 | 1.000 |
| adaptive | obsolete_fact_exposure | 0.000 | 0.000 |

## 19. Failure Taxonomy Distribution (primary)

| method | budget | failure classes |
| --- | --- | --- |
| raw_clipped | 256 | {'RETRIEVAL_MISS': 3, 'SUCCESS': 3} |
| raw_clipped | 512 | {'RETRIEVAL_MISS': 2, 'SUCCESS': 4} |
| sliding_window | 256 | {'MEMORY_MISS': 5, 'SUCCESS': 1} |
| sliding_window | 512 | {'MEMORY_MISS': 4, 'SUCCESS': 2} |
| llm_summarization | 256 | {'MEMORY_MISS': 2, 'SUCCESS': 4} |
| llm_summarization | 512 | {'CODING_ERROR': 1, 'MEMORY_MISS': 2, 'SUCCESS': 3} |
| vanilla_rag | 256 | {'OBSOLETE_INFORMATION_USED': 2, 'SUCCESS': 4} |
| vanilla_rag | 512 | {'OBSOLETE_INFORMATION_USED': 2, 'SUCCESS': 4} |
| adaptive | 256 | {'SUCCESS': 6} |
| adaptive | 512 | {'SUCCESS': 6} |

## 20. No-History / Full-Context / Direct-History Diagnostics

no_history and direct_history are stochastic diagnostics and are measured over 3 draws (seed 1, budget 1024); gate C requires *zero* no_history successes across all draws for every primary. Oracle direct_history is a recorded success rate, not a gate.

| task | no_history (x3) | full_context | direct_history oracle (x3) |
| --- | --- | --- | --- |
| user_ids | 2/3 (True) | False | 3/3 (True) |
| validation_pure | 0/3 (False) | False | 3/3 (True) |
| transaction_atomicity | 1/3 (True) | False | 2/3 (True) |

Gold-patch harness sanity: 5/5 tasks pass when the gold patch is supplied.

## 21. Context Usage & Budget Pressure

| method | budget | mean hidden | mean total prompt |
| --- | --- | --- | --- |
| raw_clipped | 256 | 248.500 | 531.500 |
| raw_clipped | 512 | 507.500 | 790.500 |
| sliding_window | 256 | 121.833 | 404.833 |
| sliding_window | 512 | 121.833 | 404.833 |
| llm_summarization | 256 | 180.500 | 463.500 |
| llm_summarization | 512 | 333.167 | 616.167 |
| vanilla_rag | 256 | 131.333 | 414.333 |
| vanilla_rag | 512 | 131.333 | 414.333 |
| adaptive | 256 | 113.333 | 396.333 |
| adaptive | 512 | 113.333 | 396.333 |

- full_context diagnostics max tokens: (7199, 7336)
## 22. Negative-Control Results (harness sanity; NOT part of the verdict)

| method | runs | success_rate | first_pass |
| --- | --- | --- | --- |

These tasks (cache_readonly, write_retry) only sanity-check that the harness/environment solves a task the workspace signals are solvable by the gold patch; they never contribute to the paired comparisons or to `adaptive_advances` (D37).
## 23. Threats to Validity & Limitations

- Word-count tokenizer approximates the model's real tokenization; budgets are approximate (documented in the manifest).
- Small task count and seeds imply wide success-rate CIs; absence of a difference is not evidence of equivalence.
- The task suite is synthetic-but-real-code; findings may not transfer to large repositories.
- `full_context` is a context-unconstrained diagnostic, not a fixed-budget competitor; `direct_history` is an oracle.
- Ollama-served `llama3.1:8b` at temperature 0.1 is not deterministic across runs; per-run success is the unit and the grid is resumable.

## 24. Conclusion

- Mode: **pilot**.
- Primary-task runs completed: **60**.
- Negative-control runs completed: **0**.
- Gates all passed: **False**.
- adaptive_advances: **False**.

At least one pilot gate failed; findings are directional only.

## 25. Reproduction / Artifacts / Provenance

- JSON: `e19_coding_generalization.json` (every record is the E19 schema with experiment='E19'; no fabricated fields).
- generated_at: 1789733915.8019342
- tokenizer: `shared_word_count`; model: `llama3.1:8b`; embedding: `nomic-embed-text`; use_embeddings: True.
- unit-test suite gate: exit_code=0, passed=143.
- work dir: `experiments/results/e19_work/` (gitignored).
- E17/E18 artifacts are immutable and were not regenerated.

— All numbers are generated from the E19 JSON by `generate_report()`; no hand-typed figures.