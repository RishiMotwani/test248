# E18 Correction-Safety Report (Phase 14)

Did moving explicit supersession ahead of the generic semantic-duplicate threshold make corrections authoritative at the consolidation boundary? Offline identity gate first; targeted coding validation second.

## 1. Purpose & Research Question

A correction/supersession is a **semantic relationship**, not a high-similarity duplicate (D35). Phase 14 makes `_is_supersession` (unchanged) run before the cos>=0.90 / overlap gate in `dedupe_incremental`, so a restated-and-corrected fact replaces the stale fact's memory entry instead of coexisting with it. E18 verifies (a) the store after the production preparation path contains exactly the current fact of every correction pair (identity-based), and (b) that this propagates to coding-task success on the tasks where E17 observed the failure.

## 2. Root Cause & the Fix

In Phase 13 the supersession check sat *inside* `if sim >= 0.90`, so a correction whose restatement used new wording (cosine ~0.7 vs the stored fact) never reached it and was stored as an additional memory. Retrieval ranked the stale predecessor first (E17 failing cells: `correction_recall == 0`, `obsolete_fact_exposure == 1.0`, adaptive context identical to vanilla_rag). The trusted `_replace_with_supersession` mutation is factored out and invoked: (1) for `supersedes_turn` targets, (2) before any similarity computation in the generic loop. The duplicate-merge path is otherwise unchanged; `_is_supersession` still requires a revision marker + full-word restatement (no weakening).

## 3. Offline Identity Gate: Method

Adaptive production preparation (retention `dual_score`, `protect_corrections=True`, store budget `max(4096, budget*4)`, injection budget = historical budget) replayed over ['cache_readonly', 'user_ids', 'write_retry', 'validation_pure'] x [1, 2, 3] seeds x [256, 512, 1024] budgets = 36 cells. Every correction pair is checked **by fact identity** (`fact_id`) in the final store: `current_fact_present` and `obsolete_fact_present`. A cell passes iff `correction_recall == 1.0` and `obsolete_exposure == 0.0`; cells for tasks without corrections are recorded but not gated.

## 4. Offline Identity Gate: Results

| task | seed | budget | n_corr | correction_recall | obsolete_exposure | store_tokens | memory_count | current@id | obsolete@id | cell gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cache_readonly | 1 | 256 | 1 | 1.000 | 0.000 | 90 | 5 | True/False | PASS |
| cache_readonly | 1 | 512 | 1 | 1.000 | 0.000 | 90 | 5 | True/False | PASS |
| cache_readonly | 1 | 1024 | 1 | 1.000 | 0.000 | 90 | 5 | True/False | PASS |
| cache_readonly | 2 | 256 | 1 | 1.000 | 0.000 | 90 | 5 | True/False | PASS |
| cache_readonly | 2 | 512 | 1 | 1.000 | 0.000 | 90 | 5 | True/False | PASS |
| cache_readonly | 2 | 1024 | 1 | 1.000 | 0.000 | 90 | 5 | True/False | PASS |
| cache_readonly | 3 | 256 | 1 | 1.000 | 0.000 | 90 | 5 | True/False | PASS |
| cache_readonly | 3 | 512 | 1 | 1.000 | 0.000 | 90 | 5 | True/False | PASS |
| cache_readonly | 3 | 1024 | 1 | 1.000 | 0.000 | 90 | 5 | True/False | PASS |
| user_ids | 1 | 256 | 1 | 1.000 | 0.000 | 67 | 4 | True/False | PASS |
| user_ids | 1 | 512 | 1 | 1.000 | 0.000 | 67 | 4 | True/False | PASS |
| user_ids | 1 | 1024 | 1 | 1.000 | 0.000 | 67 | 4 | True/False | PASS |
| user_ids | 2 | 256 | 1 | 1.000 | 0.000 | 67 | 4 | True/False | PASS |
| user_ids | 2 | 512 | 1 | 1.000 | 0.000 | 67 | 4 | True/False | PASS |
| user_ids | 2 | 1024 | 1 | 1.000 | 0.000 | 67 | 4 | True/False | PASS |
| user_ids | 3 | 256 | 1 | 1.000 | 0.000 | 67 | 4 | True/False | PASS |
| user_ids | 3 | 512 | 1 | 1.000 | 0.000 | 67 | 4 | True/False | PASS |
| user_ids | 3 | 1024 | 1 | 1.000 | 0.000 | 67 | 4 | True/False | PASS |
| validation_pure | 1 | 256 | 1 | 1.000 | 0.000 | 95 | 5 | True/False | PASS |
| validation_pure | 1 | 512 | 1 | 1.000 | 0.000 | 95 | 5 | True/False | PASS |
| validation_pure | 1 | 1024 | 1 | 1.000 | 0.000 | 95 | 5 | True/False | PASS |
| validation_pure | 2 | 256 | 1 | 1.000 | 0.000 | 95 | 5 | True/False | PASS |
| validation_pure | 2 | 512 | 1 | 1.000 | 0.000 | 95 | 5 | True/False | PASS |
| validation_pure | 2 | 1024 | 1 | 1.000 | 0.000 | 95 | 5 | True/False | PASS |
| validation_pure | 3 | 256 | 1 | 1.000 | 0.000 | 95 | 5 | True/False | PASS |
| validation_pure | 3 | 512 | 1 | 1.000 | 0.000 | 95 | 5 | True/False | PASS |
| validation_pure | 3 | 1024 | 1 | 1.000 | 0.000 | 95 | 5 | True/False | PASS |

Non-correction cells (write_retry, recorded only): 9.

## 5. Offline Identity Gate: Verdict

- correction-bearing cells: **27**
- cells passing: **27**
- aggregate correction_recall: **1.000**
- aggregate obsolete_exposure: **0.000**
- failing cells: **none**
- **OFFLINE GATE: PASS**

Per task:

| task | correction_recall | obsolete_exposure | mean_store_tokens | mean_memory_count |
| --- | --- | --- | --- | --- |
| cache_readonly | 1.000 | 0.000 | 90.0 | 5.0 |
| user_ids | 1.000 | 0.000 | 67.0 | 4.0 |
| validation_pure | 1.000 | 0.000 | 95.0 | 5.0 |

## 6. Targeted Coding Validation: Design

2 correction-bearing tasks (['user_ids', 'validation_pure']) x [1, 2, 3] seeds x [256, 512, 1024] budgets x ['adaptive', 'vanilla_rag', 'llm_summarization', 'raw_clipped'] methods = 72 runs. `sliding_window` excluded per the Phase 14 spec. Reuses `experiments/coding_benchmark.py` unchanged (same prompt scaffold, temperature 0.1, retry policy, tokenizer).

## 7. Coding Results: Success Rate by Method x Budget

| method | 256 | 512 | 1024 | overall | first-pass |
| --- | --- | --- | --- | --- | --- |
| adaptive | 1.000 | 1.000 | 0.833 | 0.944 | 0.944 |
| vanilla_rag | 0.500 | 0.500 | 0.500 | 0.500 | 0.333 |
| llm_summarization | 0.500 | 0.500 | 0.500 | 0.500 | 0.444 |
| raw_clipped | 0.500 | 0.333 | 0.500 | 0.444 | 0.389 |

## 8. Coding Results: Paired Comparisons (adaptive - other)

| vs | pairs | adaptive | other | mean diff |
| --- | --- | --- | --- | --- |
| vanilla_rag | 18 | 0.944 | 0.500 | 0.444 |
| llm_summarization | 18 | 0.944 | 0.500 | 0.444 |
| raw_clipped | 18 | 0.944 | 0.444 | 0.500 |

## 9. Failure Taxonomy Distribution

| method | budget | failure classes |
| --- | --- | --- |
| adaptive | 256 | {'SUCCESS': 6} |
| adaptive | 512 | {'SUCCESS': 6} |
| adaptive | 1024 | {'CODING_ERROR': 1, 'SUCCESS': 5} |
| vanilla_rag | 256 | {'OBSOLETE_INFORMATION_USED': 3, 'SUCCESS': 3} |
| vanilla_rag | 512 | {'OBSOLETE_INFORMATION_USED': 3, 'SUCCESS': 3} |
| vanilla_rag | 1024 | {'OBSOLETE_INFORMATION_USED': 3, 'SUCCESS': 3} |
| llm_summarization | 256 | {'CODING_ERROR': 1, 'MEMORY_MISS': 2, 'SUCCESS': 3} |
| llm_summarization | 512 | {'MEMORY_MISS': 3, 'SUCCESS': 3} |
| llm_summarization | 1024 | {'CODING_ERROR': 1, 'MEMORY_MISS': 2, 'SUCCESS': 3} |
| raw_clipped | 256 | {'RETRIEVAL_MISS': 3, 'SUCCESS': 3} |
| raw_clipped | 512 | {'RETRIEVAL_MISS': 4, 'SUCCESS': 2} |
| raw_clipped | 1024 | {'RETRIEVAL_MISS': 3, 'SUCCESS': 3} |

## 10. Store & Token Metrics

Offline cells: store_tokens range (67, 95); memory_count range (4, 6).

## 11. Context Integrity & Leakage

- leaking runs: **0** (none).
- runs classifying as CONTEXT_OVERFLOW: **0**.
- predictions total-prompt-tokens <= model window enforced by the harness on every run.

## 12. Per-Cell Failure Analysis

| cell | method | success | failure_class | corr_recall | obs_exposure | context_tokens |
| --- | --- | --- | --- | --- | --- | --- |
| user_ids:1:256 | adaptive | ok | SUCCESS | 1.000 | 0.000 | 67 |
| user_ids:1:256 | vanilla_rag | ok | SUCCESS | 0.000 | 1.000 | 85 |
| user_ids:1:256 | llm_summarization | ok | SUCCESS | 1.000 | 0.000 | 181 |
| user_ids:1:256 | raw_clipped | ok | SUCCESS | 0.000 | 0.000 | 255 |
| user_ids:1:512 | adaptive | ok | SUCCESS | 1.000 | 0.000 | 67 |
| user_ids:1:512 | vanilla_rag | ok | SUCCESS | 0.000 | 1.000 | 85 |
| user_ids:1:512 | llm_summarization | ok | SUCCESS | 1.000 | 0.000 | 265 |
| user_ids:1:512 | raw_clipped | ok | SUCCESS | 0.000 | 0.000 | 511 |
| user_ids:1:1024 | adaptive | ok | SUCCESS | 1.000 | 0.000 | 67 |
| user_ids:1:1024 | vanilla_rag | ok | SUCCESS | 0.000 | 1.000 | 85 |
| user_ids:1:1024 | llm_summarization | ok | SUCCESS | 0.000 | 0.000 | 270 |
| user_ids:1:1024 | raw_clipped | ok | SUCCESS | 0.000 | 0.000 | 1018 |
| user_ids:2:256 | adaptive | ok | SUCCESS | 1.000 | 0.000 | 67 |
| user_ids:2:256 | vanilla_rag | ok | SUCCESS | 0.000 | 1.000 | 85 |
| user_ids:2:256 | llm_summarization | ok | SUCCESS | 1.000 | 0.000 | 184 |
| user_ids:2:256 | raw_clipped | ok | SUCCESS | 0.000 | 0.000 | 252 |
| user_ids:2:512 | adaptive | ok | SUCCESS | 1.000 | 0.000 | 67 |
| user_ids:2:512 | vanilla_rag | ok | SUCCESS | 0.000 | 1.000 | 85 |
| user_ids:2:512 | llm_summarization | ok | SUCCESS | 1.000 | 0.000 | 312 |
| user_ids:2:512 | raw_clipped | FAIL | RETRIEVAL_MISS | 0.000 | 0.000 | 506 |
| user_ids:2:1024 | adaptive | ok | SUCCESS | 1.000 | 0.000 | 67 |
| user_ids:2:1024 | vanilla_rag | ok | SUCCESS | 0.000 | 1.000 | 85 |
| user_ids:2:1024 | llm_summarization | ok | SUCCESS | 0.000 | 0.000 | 331 |
| user_ids:2:1024 | raw_clipped | ok | SUCCESS | 0.000 | 0.000 | 1015 |
| user_ids:3:256 | adaptive | ok | SUCCESS | 1.000 | 0.000 | 67 |
| user_ids:3:256 | vanilla_rag | ok | SUCCESS | 0.000 | 1.000 | 85 |
| user_ids:3:256 | llm_summarization | ok | SUCCESS | 1.000 | 0.000 | 193 |
| user_ids:3:256 | raw_clipped | ok | SUCCESS | 0.000 | 0.000 | 246 |
| user_ids:3:512 | adaptive | ok | SUCCESS | 1.000 | 0.000 | 67 |
| user_ids:3:512 | vanilla_rag | ok | SUCCESS | 0.000 | 1.000 | 85 |
| user_ids:3:512 | llm_summarization | ok | SUCCESS | 1.000 | 0.000 | 325 |
| user_ids:3:512 | raw_clipped | ok | SUCCESS | 0.000 | 0.000 | 499 |
| user_ids:3:1024 | adaptive | ok | SUCCESS | 1.000 | 0.000 | 67 |
| user_ids:3:1024 | vanilla_rag | ok | SUCCESS | 0.000 | 1.000 | 85 |
| user_ids:3:1024 | llm_summarization | ok | SUCCESS | 0.000 | 0.000 | 266 |
| user_ids:3:1024 | raw_clipped | ok | SUCCESS | 0.000 | 0.000 | 1019 |
| validation_pure:1:256 | adaptive | ok | SUCCESS | 1.000 | 0.000 | 95 |
| validation_pure:1:256 | vanilla_rag | FAIL | OBSOLETE_INFORMATION_USED | 0.000 | 1.000 | 105 |
| validation_pure:1:256 | llm_summarization | FAIL | CODING_ERROR | 1.000 | 0.000 | 170 |
| validation_pure:1:256 | raw_clipped | FAIL | RETRIEVAL_MISS | 0.000 | 0.000 | 244 |
| validation_pure:1:512 | adaptive | ok | SUCCESS | 1.000 | 0.000 | 95 |
| validation_pure:1:512 | vanilla_rag | FAIL | OBSOLETE_INFORMATION_USED | 0.000 | 1.000 | 105 |
| validation_pure:1:512 | llm_summarization | FAIL | MEMORY_MISS | 0.000 | 0.000 | 279 |
| validation_pure:1:512 | raw_clipped | FAIL | RETRIEVAL_MISS | 0.000 | 0.000 | 509 |
| validation_pure:1:1024 | adaptive | ok | SUCCESS | 1.000 | 0.000 | 95 |
| validation_pure:1:1024 | vanilla_rag | FAIL | OBSOLETE_INFORMATION_USED | 0.000 | 1.000 | 105 |
| validation_pure:1:1024 | llm_summarization | FAIL | CODING_ERROR | 1.000 | 0.000 | 324 |
| validation_pure:1:1024 | raw_clipped | FAIL | RETRIEVAL_MISS | 0.000 | 0.000 | 1023 |
| validation_pure:2:256 | adaptive | ok | SUCCESS | 1.000 | 0.000 | 95 |
| validation_pure:2:256 | vanilla_rag | FAIL | OBSOLETE_INFORMATION_USED | 0.000 | 1.000 | 105 |
| validation_pure:2:256 | llm_summarization | FAIL | MEMORY_MISS | 0.000 | 0.000 | 177 |
| validation_pure:2:256 | raw_clipped | FAIL | RETRIEVAL_MISS | 0.000 | 0.000 | 250 |
| validation_pure:2:512 | adaptive | ok | SUCCESS | 1.000 | 0.000 | 95 |
| validation_pure:2:512 | vanilla_rag | FAIL | OBSOLETE_INFORMATION_USED | 0.000 | 1.000 | 105 |
| validation_pure:2:512 | llm_summarization | FAIL | MEMORY_MISS | 0.000 | 0.000 | 272 |
| validation_pure:2:512 | raw_clipped | FAIL | RETRIEVAL_MISS | 0.000 | 0.000 | 505 |
| validation_pure:2:1024 | adaptive | FAIL | CODING_ERROR | 1.000 | 0.000 | 95 |
| validation_pure:2:1024 | vanilla_rag | FAIL | OBSOLETE_INFORMATION_USED | 0.000 | 1.000 | 105 |
| validation_pure:2:1024 | llm_summarization | FAIL | MEMORY_MISS | 0.000 | 0.000 | 203 |
| validation_pure:2:1024 | raw_clipped | FAIL | RETRIEVAL_MISS | 0.000 | 0.000 | 1023 |
| validation_pure:3:256 | adaptive | ok | SUCCESS | 1.000 | 0.000 | 95 |
| validation_pure:3:256 | vanilla_rag | FAIL | OBSOLETE_INFORMATION_USED | 0.000 | 1.000 | 105 |
| validation_pure:3:256 | llm_summarization | FAIL | MEMORY_MISS | 0.000 | 0.000 | 171 |
| validation_pure:3:256 | raw_clipped | FAIL | RETRIEVAL_MISS | 0.000 | 0.000 | 249 |
| validation_pure:3:512 | adaptive | ok | SUCCESS | 1.000 | 0.000 | 95 |
| validation_pure:3:512 | vanilla_rag | FAIL | OBSOLETE_INFORMATION_USED | 0.000 | 1.000 | 105 |
| validation_pure:3:512 | llm_summarization | FAIL | MEMORY_MISS | 0.000 | 0.000 | 306 |
| validation_pure:3:512 | raw_clipped | FAIL | RETRIEVAL_MISS | 0.000 | 0.000 | 509 |
| validation_pure:3:1024 | adaptive | ok | SUCCESS | 1.000 | 0.000 | 95 |
| validation_pure:3:1024 | vanilla_rag | FAIL | OBSOLETE_INFORMATION_USED | 0.000 | 1.000 | 105 |
| validation_pure:3:1024 | llm_summarization | FAIL | MEMORY_MISS | 0.000 | 0.000 | 284 |
| validation_pure:3:1024 | raw_clipped | FAIL | RETRIEVAL_MISS | 0.000 | 0.000 | 1021 |

## 13. What Was NOT Changed

- `_is_supersession()`: unchanged - still requires a revision/negation marker plus full restatement of the stored fact.
- `memory_optimizer/retrieval.py` and `memory_optimizer/pipeline.py`: not modified in Phase 14 (retrieval-weight fix only if evidence gates demanded it).
- `config.yaml` production values: untouched (no threshold, weight, budget or embedding-model changes).
- E17 artifacts: immutable; not regenerated.

## 14. Threats to Validity & Limitations

- Task suite is synthetic-but-real-code; 6-8 facts per 600-turn history, so the store stays far under its budget - the offline gate isolates consolidation, not budget pressure.
- Codes' corrections are oracle-pre-extracted; live-extraction corrections are out of scope.
- 72 coding runs => wide success-rate CIs; absence of a difference is not evidence of equivalence.
- Ollama-served llama3.1:8b at temperature 0.1 is not deterministic; the grid is resumable and per-run cells are the unit.

## 15. Relationship to E17 (Phase 13 artifacts, immutable)

E17 verdict: gates_all_passed=True, adaptive_advances=False. E17's adaptive failure cells were exactly the ones where the obsolete fact was exposed and the correction was not recalled. E18's offline gate verifies the consolidation boundary is fixed; the coding half re-measures the downstream outcome on the two E17 offenders (user_ids, validation_pure).

## 16. Conclusion

- OFFLINE GATE: **PASS** - every correction-bearing cell keeps the current fact and holds no obsolete fact in the store.

## 17. Reproduction / Artifacts / Provenance

- JSON: `e18_correction_safety.json`.
- generated_at: 1789719456.4490452
- tokenizer: `shared_word_count`; model: `llama3.1:8b`; embedding: `nomic-embed-text`; use_embeddings: True.
- work dir: `experiments/results/e18_work/` (gitignored).

## 18. Open Questions

- How does the same correction handling behave under store budget pressure (facts-heavy histories), where eviction and the correction-protection flag interact?
- Is oracle-pre-extracted correction text a fair proxy for candidate corrections produced by a live extractor?

## 19. Next Research Question

Evaluate whether correction authoritativeness holds when the history contains *many* corrections of the same fact (fact versioning) and whether retrieval should treat superseded_prior references as a guardrail signal rather than a lexical cue.

— All numbers are generated from the E18 JSON by `generate_report()`; no hand-typed figures.