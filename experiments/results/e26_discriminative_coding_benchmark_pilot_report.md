# E26 Discriminative Benchmark - Pilot Report (Phase 21)

## 1. Purpose

Does the production adaptive pipeline keep only the current policy facts in context when the historical contract is contradicted, and does vanilla RAG drown the current contract in superseded decisions? This pilot fixes the 48-cell grid, the five gates and the pairs for the later full grid.

## 2. Config

- groups: ['release_adapter', 'invoice_adapter', 'message_adapter']
- variants: ['A', 'B']
- seeds: [1, 2] budgets: [96]
- methods: ['no_history', 'direct_history', 'vanilla_rag', 'adaptive']
- model: llama3.1:8b embedding: nomic-embed-text
- cells: 48 / 48 expected

## 3. Offline fixture gates

- all 6 variants validated: PASS
  - release_adapter:A: PASS
  - release_adapter:B: PASS
  - invoice_adapter:A: PASS
  - invoice_adapter:B: PASS
  - message_adapter:A: PASS
  - message_adapter:B: PASS

## 4. Pilot gates

### Gate history_dependence

- status: FAIL
  - release_adapter: cells=4; direct_history_success=0.0; no_history_success=0.0; direct_ok=False; no_history_ok=True; diff_cells=0; diff_ok=False
  - invoice_adapter: cells=4; direct_history_success=0.75; no_history_success=0.0; direct_ok=True; no_history_ok=True; diff_cells=3; diff_ok=True
  - message_adapter: cells=4; direct_history_success=0.5; no_history_success=0.0; direct_ok=False; no_history_ok=True; diff_cells=2; diff_ok=True

### Gate contradiction_state

- status: FAIL
  - vanilla_rag: cells=12; correction_recall=0.375; obsolete_fact_exposure=0.2917
  - adaptive: cells=12; correction_recall=0.9375; obsolete_fact_exposure=0.0

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

## 5. Per-method pilot results

| method | runs | success | first-pass | ctx tokens | budget ratio | corr recall | obs exposure | states |
|---|---|---|---|---|---|---|---|---|
| adaptive | 12 | 0.50 | 0.33 | 83 | 0.86 | 0.94 | 0.00 | CC:45 CPO:0 OO:0 N:3 |
| direct_history | 12 | 0.42 | 0.42 | 84 | 0.88 | 1.00 | 0.00 | CC:48 CPO:0 OO:0 N:0 |
| no_history | 12 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | CC:0 CPO:0 OO:0 N:48 |
| vanilla_rag | 12 | 0.00 | 0.00 | 83 | 0.87 | 0.38 | 0.29 | CC:18 CPO:12 OO:2 N:16 |

## 6. Paired adaptive vs vanilla (pilot, 12 pairs)

- pairs: None mean_diff N/A CI95 None

## 7. Verdict (pilot)

- `adaptive_beats_vanilla` (full decision forwarded to the full grid): the full-grid CI is computed over 27 paired cells; the pilot only verifies the gates.

## 8. Threats

- The fixture gate set is offline and deterministic; any gate failure here stops the full grid by construction.
- Pilot cells run at a single budget (96); budgets 64 and 128 are held for the full grid only.
