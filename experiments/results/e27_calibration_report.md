# E27 Counterfactual Calibration Repair - Pilot Report (Phase 22)

## 1. Purpose

E27 calibrates a *single-policy* counterfactual measurement surface for a future head-to-head benchmark. It does not run a full grid, does not declare a winner, and does not claim any adaptive-vs-RAG result: a head-to-head grid is a separate future phase that must be explicitly commissioned.

## 2. Relation to E26 (Phase 21)

E26 was stopped at the pilot because two gates failed: ``history_dependence`` (release_adapter direct 0/4; message 2/4; invoice 3/4) and ``contradiction_state`` (adaptive corr_recall 0.94 / obs_exposure 0.0; vanilla obs_exposure 0.29 < 0.75). Its 54-cell full grid never ran and ``adaptive_beats_vanilla`` is False. E27 repairs the surface by reducing each task to exactly ONE decisive historical policy with one obsolete fact, one current correction and four distractors at a single 96-token budget. The E26 suite, runner and results are byte-identical and are never re-run.

## 3. Design

- groups: route_contract, serialization_contract, retry_contract
- each group: byte-identical workspace+prompt across variants; hidden history is the only difference
- facts per variant: 1 obsolete policy fact (turns 80-150, 50-56 tokens), 1 current correction (turns 470-530, 50-56 tokens), 4 distractors (18-24 tokens); obsolete+current sum > 96 tokens so the two sides of the conflict never both fit
- base workspace fails the hidden test; gold patch passes; obsolete-only patch fails

## 4. Config

- groups: ['route_contract', 'serialization_contract', 'retry_contract']
- variants: ['A', 'B'] seeds: [1, 2] budgets: [96]
- methods: ['no_history', 'direct_history', 'vanilla_rag', 'adaptive']
- model: llama3.1:8b embedding: nomic-embed-text
- cells: 48 / 48 expected

## 5. Offline fixture gates (Table A)

| variant | status | failing checks |
|---|---|---|
| route_contract:A | PASS |  |
| route_contract:B | PASS |  |
| serialization_contract:A | PASS |  |
| serialization_contract:B | PASS |  |
| retry_contract:A | PASS |  |
| retry_contract:B | PASS |  |

- **all 6 variants validated: PASS**

## 6. Grid completeness

- cells: 48 / 48 expected; complete=True; missing=[]

## 7. Pilot gates - history dependence (Table B)

- route_contract: direct=1.00 (ok=True) no_history=0.75 (ok=False) diff=1 (ok=False) -> FAIL
- serialization_contract: direct=1.00 (ok=True) no_history=0.75 (ok=False) diff=1 (ok=False) -> FAIL
- retry_contract: direct=0.00 (ok=False) no_history=0.00 (ok=True) diff=0 (ok=False) -> FAIL

## 8. Pilot gates - contradiction state

### Gate contradiction_state

- status: FAIL
  - vanilla_rag: cells=12; correction_recall=0.3333; obsolete_fact_exposure=0.6667
  - adaptive: cells=12; correction_recall=1.0; obsolete_fact_exposure=0.0

### Gate budget_binding

- status: PASS
  - vanilla_rag: cells=12; bound_cells=12; fraction=1.0
  - adaptive: cells=12; bound_cells=12; fraction=1.0

### Gate context_difference

- status: PASS

### Gate no_leakage

- status: PASS
  - leaked_cells=0
  - leaked=[]

- **all pilot gates: FAIL**

## 9. Per-group x method success (Table C)

| group | no_history | direct_history | vanilla_rag | adaptive |
|---|---|---|---|---|
| route_contract | 0.75 | 1.00 | 0.00 | 1.00 |
| serialization_contract | 0.75 | 1.00 | 0.50 | 1.00 |
| retry_contract | 0.00 | 0.00 | 0.00 | 0.00 |

## 10. Per-method aggregates (Table D)

| method | runs | success | first-pass | ctx tokens | budget ratio | corr recall | obs exposure | states |
|---|---|---|---|---|---|---|---|---|
| adaptive | 12 | 0.67 | 0.50 | 91 | 0.94 | 1.00 | 0.00 | CC:12 CPO:0 OO:0 N:0 |
| direct_history | 12 | 0.67 | 0.50 | 53 | 0.55 | 1.00 | 0.00 | CC:12 CPO:0 OO:0 N:0 |
| no_history | 12 | 0.50 | 0.17 | 0 | 0.00 | 0.00 | 0.00 | CC:0 CPO:0 OO:0 N:12 |
| vanilla_rag | 12 | 0.17 | 0.17 | 92 | 0.95 | 0.33 | 0.67 | CC:4 CPO:0 OO:8 N:0 |

## 11. Correction-state distribution (diagnostics)

- adaptive: {'CLEAN_CURRENT': 12, 'CURRENT_PLUS_OBSOLETE': 0, 'OBSOLETE_ONLY': 0, 'NEITHER': 0}
- direct_history: {'CLEAN_CURRENT': 12, 'CURRENT_PLUS_OBSOLETE': 0, 'OBSOLETE_ONLY': 0, 'NEITHER': 0}
- no_history: {'CLEAN_CURRENT': 0, 'CURRENT_PLUS_OBSOLETE': 0, 'OBSOLETE_ONLY': 0, 'NEITHER': 12}
- vanilla_rag: {'CLEAN_CURRENT': 4, 'CURRENT_PLUS_OBSOLETE': 0, 'OBSOLETE_ONLY': 8, 'NEITHER': 0}

## 12. Gold checks (StaticCoder, offline)

- gold patches pass the hidden tests for all 6 variants: PASS

## 13. Verdict

- calibration_valid: False
- history_dependence_passed: False
- contradiction_state_passed: False
- budget_binding_passed: True
- context_difference_passed: True
- no_leakage_passed: True
- full_head_to_head_eligible: False
- rationale: ['pilot gates did not pass']

## 14. Stop rule

E27 stops after this pilot. Calibration validity is judged on the five locked gates and a complete 48-cell grid; nothing is tuned on pilot outcomes. Even if ``full_head_to_head_eligible`` is True, E27 does not run or claim a head-to-head and declares no winner.

## 15. Threats to validity

- Facts are oracle-pre-extracted; extraction drift is out of scope and would bias both recall arms identically.
- A single 96-token budget is used; the calibration does not sweep budget sensitivity.
- Vanilla RAG observation mode and re-ranking depend on the embedding model (nomic-embed-text); a different embedder could shift vanilla exposure.
