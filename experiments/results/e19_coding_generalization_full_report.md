# E19 Coding-Generalization Report (Phase 15)
## 1. Purpose & Research Question

After Phase 14 made explicit supersession authoritative at the consolidation boundary (D35) and Phase 15 fixed a second, adjacent defect (D36: a supersession replaced the fact text but could keep the obsolete predecessor's embedding, so retrieval continued to rank the corrected fact by obsolete semantics), E19 asks: **does the corrected adaptive memory system generalize to a broader set of genuinely history-dependent long-horizon coding tasks better than simpler context-management strategies under the same historical-context budget?** The dependent variable is whether the produced patch passes deterministic hidden tests.
## 2. Causal Chain & What Changed Since E18 (D36 / D37)

E18 re-validated the consolidation boundary after the Phase 14 fix (offline identity gate passed; correction_recall == 1.0, obsolete_exposure == 0.0). Phase 15 hardens the wire between consolidation and retrieval: `_replace_with_supersession` now stores the correcting statement's embedding when it arrives and otherwise drops the stale predecessor embedding, so the surviving memory always embeds its current text (D36). E19 then tests this corrected system on a wider task set with strict primary / negative-control separation (D37): only the three genuinely history-dependent primary tasks may contribute to the adaptive-advances decision. config_contract was calibrated but excluded: probing showed the required explicit-empty-string nuance is not reliably expressible by llama3.1:8b even with oracle history (0/3), so it carries no memory signal at this model; its fixture is retained for future stronger models.
## 3. Design: Primary vs Negative-Control Split (D37)

- **Primary tasks** (history-dependent; the paired/verdict analysis uses only these): routing_policy, retry_policy, serialization_policy.
- **Negative-control tasks** (harness sanity only; recorded and reported, never used in paired comparisons or the adaptive-advances verdict): cache_readonly, write_retry.
- **Fixture-only task**: config_contract (authored and calibrated during the Phase 15 pilot, but excluded from the primary set — llama3.1:8b cannot reliably express its explicit-empty-string fact even with oracle history; probing: 0/3 oracle passes across four fixture designs). Its fixture remains committed for future stronger models; it is not part of the Phase 18 primary set.
- Reasoning: a generalization claim must be earned on tasks whose constraints genuinely require the buried history; tasks that can be solved from the workspace alone would measure coding ability, not memory. Each primary task is a fixed Variant-A instance of a counterfactual family independently calibrated by E20, so the history dependence of the benchmark is externally certified. The negative controls keep the environment honest (a correctly-built harness passes them with the gold patch).
## 4. Task Suite

| task | group | title | critical facts | corrections | negative constraints |
| --- | --- | --- | --- | --- | --- |
| routing_policy | primary | counterfactual routing_policy variant A (the current routing contract assigns opaque operations to deployment lanes according to the historical allocation table) | 3 | 1 | 0 |
| retry_policy | primary | counterfactual retry_policy variant A (the write is non-idempotent: submit must NOT retry a failed attempt and must surface the error after a single call) | 3 | 1 | 0 |
| serialization_policy | primary | counterfactual serialization_policy variant A (unknown/future config keys must be preserved through normalization) | 3 | 1 | 0 |
| cache_readonly | negative-control | Implement a cache write helper (early architectural constraint) | 2 | 1 | 1 |
| write_retry | negative-control | Implement a record submitter (negative constraint) | 2 | 0 | 2 |

## 5. Historical Dependence Design

Each primary task is sourced from a Phase-17-validated E20 counterfactual family. The E20 calibration used identical workspace and task prompt across variants, variant-specific historical contracts, variant-specific hidden tests/gold patches, deterministic 600-turn histories, and predeclared success thresholds. E19 uses Variant A of each validated family as the fixed coding task. The A/B counterfactual validation, rather than stochastic E19 no_history draws, is the benchmark-level history-dependence certification.

Task-specific mechanisms: routing_policy uses an opaque operation→lane allocation absent from the workspace; retry_policy uses a historical idempotency/retry contract; serialization_policy uses a historical policy governing unknown configuration keys.

No fact text uses IMPORTANT/CRITICAL marker language; fact_ids are identity-safe; histories are generated from structured data with no LLM calls. The task prompts never mention the critical historical rules; the no_history diagnostic (section 20) quantifies dependence descriptively.
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

Historical budgets: [256, 512, 1024] tokens (shared word-count tokenizer `shared_word_count`). For adaptive, `injection_token_limit = budget` and `memory_store_token_budget = max(4096, budget*4)`, mirroring the repo convention; this is documented, not tuned. Per-run token accounting separates historical_context_tokens, workspace_context_tokens, task_prompt_tokens and total_prompt_tokens.
## 8. Coding Model, Prompt & Harness

Coding model `llama3.1:8b` via Ollama (`temperature=0.1`, `num_ctx=8192`, `keep_alive=30m`, `num_predict=700`); embeddings `nomic-embed-text` (no silent fallback; model availability is verified before any run). The harness (`experiments/coding_benchmark.py`, unchanged) owns workspace reset, prompt construction, complete-file edit-block application (unified diff fallback), `git apply --check`, the repair-attempt policy (initial + one repair), hidden-test execution, failure classification and token accounting.
## 9. Evaluation Metrics & Failure Taxonomy

Primary: `final_success` and `first_pass_success`. Diagnostics: critical_fact_recall, correction_recall, negative_constraint_recall, long_range_fact_recall, obsolete_fact_exposure. Failure classes (deterministic precedence): MEMORY_MISS, RETRIEVAL_MISS, CONTEXT_OVERFLOW, PATCH_INVALID, CODING_ERROR, HIDDEN_TEST_FAILURE, OBSOLETE_INFORMATION_USED, CORRECTION_MISSED, OTHER (no new class added).
## 10. Configuration & Reproducibility

- **mode**: full
- **tasks**: ['routing_policy', 'retry_policy', 'serialization_policy', 'cache_readonly', 'write_retry']
- **primary_tasks**: ['routing_policy', 'retry_policy', 'serialization_policy']
- **negative_control_tasks**: ['cache_readonly', 'write_retry']
- **seeds**: [1, 2, 3]
- **budgets**: [256, 512, 1024]
- **methods**: ['raw_clipped', 'sliding_window', 'llm_summarization', 'vanilla_rag', 'adaptive']
- **grid_cells**: 225
- **model**: llama3.1:8b
- **endpoint**: http://localhost:11434
- **embedding_model**: nomic-embed-text
- **use_embeddings**: True
- **max_output_tokens**: 700
- **temperature**: 0.1
- **tokenizer**: shared_word_count
- **ingestion**: oracle_pre_extracted
- **adaptive**: {'retention_mode': 'dual_score', 'protect_corrections': True, 'injection_token_limit': 'historical_budget', 'memory_store_token_budget': 'max(4096, budget*4)'}
- **generated_at**: 1789754261.350467

## 11. Grids: Pilot & Full

- Pilot: 3 primary x 3 seeds x 3 budgets x 5 methods = **225 runs**.
- Full: primary 135 + negative 90 = **225 runs**. Every run is persisted immediately under `task_id:seed:budget:method`; the grid is resumable and completed results are never deleted.
## 12. Experiment Gates A-J

| gate | passed | detail |
| --- | --- | --- |
| embedding_consistency | PASS | {"cells": 27, "embedding_consistent_cells": 27, "failing_cells": []} |
| correction_identity | FAIL | {"cells": 27, "identity_ok_cells": 0, "failing_cells": ["routing_policy:1:256", "routing_policy:1:512", "routing_policy:1:1024", "routing_policy:2:256", "routing_policy:2:512", "routing_policy:2:1024", "routing_policy:3:256", "routing_policy:3:512", "routing_policy:3:1024", "retry_policy:1:256", "retry_policy:1:512", "retry_policy:1:1024", "retry_policy:2:256", "retry_policy:2:512", "retry_policy:2:1024", "retry_policy:3:256", "retry_policy:3:512", "retry_policy:3:1024", "serialization_policy:1: |
| history_dependence | PASS | {"source": "E20 counterfactual history calibration", "calibration": {"passed": true, "artifact": "/home/goku/prototype/prototype3/experiments/results/e20_counterfactual_history_repair.json", "groups": ["routing_policy", "retry_policy", "serialization_policy"], "expected_groups": ["routing_policy", "retry_policy", "serialization_policy"], "gold_patch_validity": {"passed": true}, "pair_integrity": {"passed": true}, "history_dependence": {"groups": {"routing_policy": {"direct_history": "6/6", "no_h |
| method_separation | PASS | {"max_distinct_contexts": 4} |
| no_leakage | PASS | {"leaking_runs": 0} |
| gold_passes | PASS | {"gold_checks": [{"task_id": "routing_policy", "final_success": true, "patch_applied": true}, {"task_id": "retry_policy", "final_success": true, "patch_applied": true}, {"task_id": "serialization_policy", "final_success": true, "patch_applied": true}, {"task_id": "cache_readonly", "final_success": true, "patch_applied": true}, {"task_id": "write_retry", "final_success": true, "patch_applied": true}]} |
| budget_pressure | PASS | {"history_exceeds_max_budget": true, "raw_grows_with_budget": true, "some_run_fills_60pct": true} |
| real_summarization | PASS | {"summary_update_calls": 324} |
| adaptive_production_path | PASS | {"adaptive_runs": 27} |
| unit_tests_pass | PASS | {"unit_tests": {"exit_code": 0, "passed": 185, "exit_ok": true, "stdout_tail": "........... [ 38%]\n........................................................................ [ 77%]\n.........................................                                [100%]\n185 passed in 17.31s\n"}} |

Gate C (`history_dependence`) uses **E20 counterfactual history calibration**: E20 is the benchmark-validity certification. E19 diagnostics are descriptive and are not used to override the E20 certification.

**All experiment gates passed: False** (verdict section 16).
## 13. Full-Grid Results: Success Rate by Method x Budget

| method | 256 | 512 | 1024 | overall |
| --- | --- | --- | --- | --- |
| raw_clipped | 0.556 | 0.556 | 0.333 | 0.481 |
| sliding_window | 0.333 | 0.222 | 0.444 | 0.333 |
| llm_summarization | 0.111 | 0.667 | 0.778 | 0.518 |
| vanilla_rag | 1.000 | 1.000 | 1.000 | 1.000 |
| adaptive | 1.000 | 1.000 | 1.000 | 1.000 |

## 14. Full Grid: Results by Method x Budget (primary)

Primary-task records only (E19 schema; see JSON). This table aggregates the full grid grid so far (mode=full).

| method | 256 | 512 | 1024 | overall |
| --- | --- | --- | --- | --- |
| raw_clipped | 0.556 | 0.556 | 0.333 | 0.481 |
| sliding_window | 0.333 | 0.222 | 0.444 | 0.333 |
| llm_summarization | 0.111 | 0.667 | 0.778 | 0.518 |
| vanilla_rag | 1.000 | 1.000 | 1.000 | 1.000 |
| adaptive | 1.000 | 1.000 | 1.000 | 1.000 |

## 15. Full Grid: First-Pass vs Final Success

Over the full grid grid aggregated in section 14 (mode=full).
| method | budget | first-pass | final | runs |
| --- | --- | --- | --- | --- |
| raw_clipped | 256 | 0.333 | 0.556 | 9 |
| raw_clipped | 512 | 0.444 | 0.556 | 9 |
| raw_clipped | 1024 | 0.222 | 0.333 | 9 |
| sliding_window | 256 | 0.111 | 0.333 | 9 |
| sliding_window | 512 | 0.111 | 0.222 | 9 |
| sliding_window | 1024 | 0.222 | 0.444 | 9 |
| llm_summarization | 256 | 0.111 | 0.111 | 9 |
| llm_summarization | 512 | 0.667 | 0.667 | 9 |
| llm_summarization | 1024 | 0.556 | 0.778 | 9 |
| vanilla_rag | 256 | 0.889 | 1.000 | 9 |
| vanilla_rag | 512 | 1.000 | 1.000 | 9 |
| vanilla_rag | 1024 | 0.778 | 1.000 | 9 |
| adaptive | 256 | 1.000 | 1.000 | 9 |
| adaptive | 512 | 1.000 | 1.000 | 9 |
| adaptive | 1024 | 1.000 | 1.000 | 9 |

## 16. Paired Comparisons & Bootstrap CIs (adaptive - baseline)

Paired on (task, seed, budget) over primary records; paired 95% percentile bootstrap CIs on the mean within-pair difference.

| vs | pairs | adaptive | baseline | mean diff | diff 95% CI | adaptive 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| raw_clipped | 27 | 1.000 | 0.481 | 0.518 | [0.333, 0.741] | [1.0, 1.0] |
| sliding_window | 27 | 1.000 | 0.333 | 0.667 | [0.481, 0.852] | [1.0, 1.0] |
| llm_summarization | 27 | 1.000 | 0.518 | 0.481 | [0.296, 0.667] | [1.0, 1.0] |
| vanilla_rag | 27 | 1.000 | 1.000 | 0.000 | [0.0, 0.0] | [1.0, 1.0] |

**Predeclared criterion:** positive paired advantage with the bootstrap 95% CI lower bound > 0 against every baseline; otherwise `adaptive_advances = false`.

- gates_all_passed: **False**
- adaptive_beats_all_baselines: **False**
- **adaptive_advances: False**

rationale: Pilot gates did not all pass; the grid is not a valid experiment and adaptive cannot be said to advance.
## 17. Verdict

Full-grid verdict per the predeclared criterion (section 16): adaptive_advances = **False**.
## 18. Diagnostics: Recall & Obsolete Exposure (primary)

| method | metric | mean (256) | mean (512) | mean (1024) |
| --- | --- | --- | --- | --- |
| raw_clipped | critical_fact_recall | 0.000 | 0.000 | 0.000 |
| sliding_window | critical_fact_recall | 0.000 | 0.000 | 0.000 |
| llm_summarization | critical_fact_recall | 0.278 | 0.278 | 0.389 |
| vanilla_rag | critical_fact_recall | 1.000 | 1.000 | 1.000 |
| adaptive | critical_fact_recall | 1.000 | 1.000 | 1.000 |
| raw_clipped | correction_recall | 0.000 | 0.000 | 0.000 |
| sliding_window | correction_recall | 0.000 | 0.000 | 0.000 |
| llm_summarization | correction_recall | 0.222 | 0.222 | 0.333 |
| vanilla_rag | correction_recall | 0.000 | 0.000 | 0.000 |
| adaptive | correction_recall | 0.000 | 0.000 | 0.000 |
| raw_clipped | negative_constraint_recall | N/A | N/A | N/A |
| sliding_window | negative_constraint_recall | N/A | N/A | N/A |
| llm_summarization | negative_constraint_recall | N/A | N/A | N/A |
| vanilla_rag | negative_constraint_recall | N/A | N/A | N/A |
| adaptive | negative_constraint_recall | N/A | N/A | N/A |
| raw_clipped | long_range_fact_recall | 0.000 | 0.000 | 0.000 |
| sliding_window | long_range_fact_recall | 0.000 | 0.000 | 0.000 |
| llm_summarization | long_range_fact_recall | 0.417 | 0.417 | 0.417 |
| vanilla_rag | long_range_fact_recall | 1.000 | 1.000 | 1.000 |
| adaptive | long_range_fact_recall | 1.000 | 1.000 | 1.000 |
| raw_clipped | obsolete_fact_exposure | 0.000 | 0.000 | 0.000 |
| sliding_window | obsolete_fact_exposure | 0.000 | 0.000 | 0.000 |
| llm_summarization | obsolete_fact_exposure | 0.556 | 0.556 | 0.556 |
| vanilla_rag | obsolete_fact_exposure | 1.000 | 1.000 | 1.000 |
| adaptive | obsolete_fact_exposure | 1.000 | 1.000 | 1.000 |

## 19. Failure Taxonomy Distribution (primary)

| method | budget | failure classes |
| --- | --- | --- |
| raw_clipped | 256 | {'RETRIEVAL_MISS': 4, 'SUCCESS': 5} |
| raw_clipped | 512 | {'RETRIEVAL_MISS': 4, 'SUCCESS': 5} |
| raw_clipped | 1024 | {'RETRIEVAL_MISS': 6, 'SUCCESS': 3} |
| sliding_window | 256 | {'MEMORY_MISS': 6, 'SUCCESS': 3} |
| sliding_window | 512 | {'MEMORY_MISS': 7, 'SUCCESS': 2} |
| sliding_window | 1024 | {'MEMORY_MISS': 5, 'SUCCESS': 4} |
| llm_summarization | 256 | {'MEMORY_MISS': 8, 'SUCCESS': 1} |
| llm_summarization | 512 | {'MEMORY_MISS': 3, 'SUCCESS': 6} |
| llm_summarization | 1024 | {'MEMORY_MISS': 2, 'SUCCESS': 7} |
| vanilla_rag | 256 | {'SUCCESS': 9} |
| vanilla_rag | 512 | {'SUCCESS': 9} |
| vanilla_rag | 1024 | {'SUCCESS': 9} |
| adaptive | 256 | {'SUCCESS': 9} |
| adaptive | 512 | {'SUCCESS': 9} |
| adaptive | 1024 | {'SUCCESS': 9} |

## 20. No-History / Full-Context / Direct-History Diagnostics

no_history and direct_history are stochastic diagnostics and are measured over 3 draws (seed 1, budget 1024). These diagnostics are descriptive only and are NOT used to certify history dependence: benchmark-level history dependence is certified by the E20 counterfactual calibration (Gate C, section 12). Oracle direct_history is a recorded success rate, not a gate.

| task | no_history (x3) | full_context | direct_history oracle (x3) |
| --- | --- | --- | --- |
| routing_policy | 0/3 (False) | False | 3/3 (True) |
| retry_policy | 3/3 (True) | False | 3/3 (True) |
| serialization_policy | 2/3 (True) | False | 3/3 (True) |

Gold-patch harness sanity: 5/5 tasks pass when the gold patch is supplied.

## 21. Context Usage & Budget Pressure

| method | budget | mean hidden | mean total prompt |
| --- | --- | --- | --- |
| raw_clipped | 256 | 244.000 | 412.333 |
| raw_clipped | 512 | 509.000 | 677.333 |
| raw_clipped | 1024 | 1017.667 | 1186.000 |
| sliding_window | 256 | 116.000 | 284.333 |
| sliding_window | 512 | 116.000 | 284.333 |
| sliding_window | 1024 | 116.000 | 284.333 |
| llm_summarization | 256 | 192.778 | 361.111 |
| llm_summarization | 512 | 354.333 | 522.667 |
| llm_summarization | 1024 | 435.778 | 604.111 |
| vanilla_rag | 256 | 109.333 | 277.667 |
| vanilla_rag | 512 | 109.333 | 277.667 |
| vanilla_rag | 1024 | 109.333 | 277.667 |
| adaptive | 256 | 109.333 | 277.667 |
| adaptive | 512 | 109.333 | 277.667 |
| adaptive | 1024 | 109.333 | 277.667 |

- full_context diagnostics max tokens: (7226, 7268)
## 22. Negative-Control Results (harness sanity; NOT part of the verdict)

| method | runs | success_rate | first_pass |
| --- | --- | --- | --- |
| adaptive | 18 | 1.000 | 1.000 |
| llm_summarization | 18 | 0.944 | 0.722 |
| raw_clipped | 18 | 1.000 | 0.500 |
| sliding_window | 18 | 1.000 | 0.500 |
| vanilla_rag | 18 | 1.000 | 1.000 |

These tasks (cache_readonly, write_retry) only sanity-check that the harness/environment solves a task the workspace signals are solvable by the gold patch; they never contribute to the paired comparisons or to `adaptive_advances` (D37).
## 23. Threats to Validity & Limitations

- Word-count tokenizer approximates the model's real tokenization; budgets are approximate (documented in the manifest).
- Small task count and seeds imply wide success-rate CIs; absence of a difference is not evidence of equivalence.
- The task suite is synthetic-but-real-code; findings may not transfer to large repositories.
- `full_context` is a context-unconstrained diagnostic, not a fixed-budget competitor; `direct_history` is an oracle.
- Ollama-served `llama3.1:8b` at temperature 0.1 is not deterministic across runs; per-run success is the unit and the grid is resumable.

## 24. Conclusion

- Mode: **full**.
- Primary-task runs completed: **135**.
- Negative-control runs completed: **90**.
- Gates all passed: **False**.
- adaptive_advances: **False**.

At least one pilot gate failed; findings are directional only.

## 25. Reproduction / Artifacts / Provenance

- JSON: `e19_coding_generalization_full.json` (every record is the E19 schema with experiment='E19'; no fabricated fields).
- generated_at: 1789754261.350467
- tokenizer: `shared_word_count`; model: `llama3.1:8b`; embedding: `nomic-embed-text`; use_embeddings: True.
- unit-test suite gate: exit_code=0, passed=185.
- work dir: `experiments/results/e19_work/` (gitignored).
- E17/E18 artifacts are immutable and were not regenerated.

— All numbers are generated from the E19 JSON by `generate_report()`; no hand-typed figures.