# E20 Counterfactual-History Benchmark Calibration (Phase 17)

*This is an E20 revalidation after replacement of the invalid identifier_policy counterfactual. The original E20 result remains immutable. The replacement group is routing_policy. The purpose is benchmark validity, not adaptive-memory comparison.*

## 1. Purpose

E19 pilot gate C failed: fixtures had to be demonstrated *genuinely* history dependent before any memory-system comparison could be meaningful. E20 builds counterfactual history pairs — identical workspace and prompt, histories engineered toward different engineering decisions, different hidden tests and gold patches — and asks whether providing the variant's history (direct_history oracle) changes solvability while providing none (no_history) does not. This calibrates the benchmark; it is not an adaptive-memory performance experiment.

## 2. Design

- Grid: 54 cells = 3 groups x 2 variants x 3 seeds x 3 methods.
- Historical budget: 1024 words (shared_word_count).
- Model: llama3.1:8b @ http://localhost:11434, temperature 0.1.
- Methods: `no_history` (identical prompt for A/B), `direct_history` (oracle: the same variant's non-obsolete gold facts only), `full_context` (raw 600-turn history; recorded as an overflow diagnostic when it exceeds the budget).

## 3. Hard Gate 1 — Gold Patch Validity (offline)

| group | variant | base hidden fails | gold patch applies | hidden tests pass |
| --- | --- | --- | --- | --- |
| routing_policy | A | PASS | PASS | PASS |
| routing_policy | B | PASS | PASS | PASS |
| retry_policy | A | PASS | PASS | PASS |
| retry_policy | B | PASS | PASS | PASS |
| serialization_policy | A | PASS | PASS | PASS |
| serialization_policy | B | PASS | PASS | PASS |

Base hidden-test gate: the unmodified workspace must NOT already pass each variant's hidden tests (base_hidden_test_pass == False).

**Hard Gate 1 passed: True**

## 4. Pair-Integrity Gates

| check | result |
| --- | --- |
| all_workspaces_mutually_distinct | PASS |
| retry_policy:final_fact_ids_differ | PASS |
| retry_policy:gold_patches_differ | PASS |
| retry_policy:hidden_tests_differ | PASS |
| retry_policy:histories_differ | PASS |
| retry_policy:history_hashes_reproducible | PASS |
| retry_policy:no_prompt_history_leak | PASS |
| retry_policy:prompt_identical_a_b | PASS |
| retry_policy:workspace_identical_a_b | PASS |
| routing_policy:final_fact_ids_differ | PASS |
| routing_policy:gold_patches_differ | PASS |
| routing_policy:hidden_tests_differ | PASS |
| routing_policy:histories_differ | PASS |
| routing_policy:history_hashes_reproducible | PASS |
| routing_policy:no_prompt_history_leak | PASS |
| routing_policy:prompt_identical_a_b | PASS |
| routing_policy:workspace_identical_a_b | PASS |
| serialization_policy:final_fact_ids_differ | PASS |
| serialization_policy:gold_patches_differ | PASS |
| serialization_policy:hidden_tests_differ | PASS |
| serialization_policy:histories_differ | PASS |
| serialization_policy:history_hashes_reproducible | PASS |
| serialization_policy:no_prompt_history_leak | PASS |
| serialization_policy:prompt_identical_a_b | PASS |
| serialization_policy:workspace_identical_a_b | PASS |

**all pair-integrity gates passed: True**

## 5. Per-Group Results (seeds 1,2,3)

| group | variant | method | success | context tokens | overflow | failure class |
| --- | --- | --- | --- | --- | --- | --- |
| retry_policy | A | direct_history | PASS | 54 | no | None |
| retry_policy | A | direct_history | PASS | 54 | no | None |
| retry_policy | A | direct_history | PASS | 54 | no | None |
| retry_policy | A | full_context | fail | 7254 | yes | CONTEXT_OVERFLOW |
| retry_policy | A | full_context | fail | 7254 | yes | CONTEXT_OVERFLOW |
| retry_policy | A | full_context | fail | 7254 | yes | CONTEXT_OVERFLOW |
| retry_policy | A | no_history | PASS | 0 | no | None |
| retry_policy | A | no_history | PASS | 0 | no | None |
| retry_policy | A | no_history | PASS | 0 | no | None |
| retry_policy | B | direct_history | PASS | 44 | no | None |
| retry_policy | B | direct_history | PASS | 44 | no | None |
| retry_policy | B | direct_history | fail | 44 | no | CODING_ERROR |
| retry_policy | B | full_context | fail | 7253 | yes | CONTEXT_OVERFLOW |
| retry_policy | B | full_context | fail | 7253 | yes | CONTEXT_OVERFLOW |
| retry_policy | B | full_context | fail | 7253 | yes | CONTEXT_OVERFLOW |
| retry_policy | B | no_history | fail | 0 | no | MEMORY_MISS |
| retry_policy | B | no_history | fail | 0 | no | MEMORY_MISS |
| retry_policy | B | no_history | fail | 0 | no | MEMORY_MISS |
| routing_policy | A | direct_history | PASS | 71 | no | None |
| routing_policy | A | direct_history | PASS | 71 | no | None |
| routing_policy | A | direct_history | PASS | 71 | no | None |
| routing_policy | A | full_context | fail | 7226 | yes | CONTEXT_OVERFLOW |
| routing_policy | A | full_context | fail | 7226 | yes | CONTEXT_OVERFLOW |
| routing_policy | A | full_context | fail | 7226 | yes | CONTEXT_OVERFLOW |
| routing_policy | A | no_history | fail | 0 | no | MEMORY_MISS |
| routing_policy | A | no_history | fail | 0 | no | MEMORY_MISS |
| routing_policy | A | no_history | fail | 0 | no | MEMORY_MISS |
| routing_policy | B | direct_history | PASS | 71 | no | None |
| routing_policy | B | direct_history | PASS | 71 | no | None |
| routing_policy | B | direct_history | PASS | 71 | no | None |
| routing_policy | B | full_context | fail | 7241 | yes | CONTEXT_OVERFLOW |
| routing_policy | B | full_context | fail | 7241 | yes | CONTEXT_OVERFLOW |
| routing_policy | B | full_context | fail | 7241 | yes | CONTEXT_OVERFLOW |
| routing_policy | B | no_history | fail | 0 | no | MEMORY_MISS |
| routing_policy | B | no_history | fail | 0 | no | MEMORY_MISS |
| routing_policy | B | no_history | fail | 0 | no | MEMORY_MISS |
| serialization_policy | A | direct_history | PASS | 50 | no | None |
| serialization_policy | A | direct_history | PASS | 50 | no | None |
| serialization_policy | A | direct_history | PASS | 50 | no | None |
| serialization_policy | A | full_context | fail | 7268 | yes | CONTEXT_OVERFLOW |
| serialization_policy | A | full_context | fail | 7268 | yes | CONTEXT_OVERFLOW |
| serialization_policy | A | full_context | fail | 7268 | yes | CONTEXT_OVERFLOW |
| serialization_policy | A | no_history | PASS | 0 | no | None |
| serialization_policy | A | no_history | fail | 0 | no | MEMORY_MISS |
| serialization_policy | A | no_history | fail | 0 | no | MEMORY_MISS |
| serialization_policy | B | direct_history | PASS | 42 | no | None |
| serialization_policy | B | direct_history | PASS | 42 | no | None |
| serialization_policy | B | direct_history | PASS | 42 | no | None |
| serialization_policy | B | full_context | fail | 7265 | yes | CONTEXT_OVERFLOW |
| serialization_policy | B | full_context | fail | 7265 | yes | CONTEXT_OVERFLOW |
| serialization_policy | B | full_context | fail | 7265 | yes | CONTEXT_OVERFLOW |
| serialization_policy | B | no_history | fail | 0 | no | MEMORY_MISS |
| serialization_policy | B | no_history | fail | 0 | no | MEMORY_MISS |
| serialization_policy | B | no_history | fail | 0 | no | MEMORY_MISS |

## 6. Hard Gate 2 — History Dependence (per group, /6)

Thresholds: direct_history >= 5/6, no_history <= 3/6, separation >= 2/6.

| group | direct_history | no_history | separation | passed |
| --- | --- | --- | --- | --- |
| routing_policy | 6/6 | 0/6 | 6 | PASS |
| retry_policy | 5/6 | 3/6 | 2 | PASS |
| serialization_policy | 6/6 | 1/6 | 5 | PASS |

**Hard Gate 2 passed: True**

## 7. Verdict

- history_dependence_benchmark: **VALID**
- e19_full_grid_eligible: **True**

rationale: E19 full grid eligible: counterfactual history pairs show genuine history dependence

*E20 does not modify the adaptive-memory system, its production defaults, or any prior experiment artifact; it only reports whether the E19 benchmark is fit to measure history dependence.*