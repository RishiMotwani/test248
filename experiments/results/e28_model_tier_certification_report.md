# E28 Model-Tier Certification - Qwen2.5:7b @ Frozen E27 Surface (Phase 23)

## 1. Purpose

E28 certifies whether the frozen E27 calibration surface behaves as the memory system expects at a *second* LLM tier (`qwen2.5:7b`), measured with the exact same 48-cell procedure, fixtures, budget, methods, embeddings and locked gate thresholds that produced E27's verdict at `llama3.1:8b`. It does NOT redesign the surface, does NOT re-run E26/E19, does NOT declare a winner and does NOT run a head-to-head grid.

## 2. Why E27 Was Frozen

E27 was stopped at two failed pilot gates and its `calibration_valid=False` verdict at `llama3.1:8b` is on record — the artifacts are byte-immutable and are never re-run. E28 therefore re-uses the E27 runner exactly (same thresholds in `e27_calibration`, delegated through a thin output-rebinding wrapper, so the E27 suite, runner, tests and result files stay untouched). The only dimension changed is the serving model (`qwen2.5:7b`).

## 3. Model-Tier Protocol

- grid: identical frozen 48-cell geometry (3 groups x 2 variants x seeds {1, 2} x budget 96 x 4 methods)
- groups: route_contract, serialization_contract, retry_contract; variants: A, B; seeds: [1, 2]; budget: 96
- methods: no_history, direct_history, vanilla_rag, adaptive
- model: `qwen2.5:7b`; embedding: `nomic-embed-text`; endpoint: `http://localhost:11434`; temperature: 0.1
- offline fixture gates are re-run before any LLM call (route_contract/serialization_contract/retry_contract west/history/gold geometry must hold)
- gate thresholds are E27's, unchanged: history dependence (direct>=3/4, no_history<=1/4, diff>=2 cells), contradiction state (adaptive corr_recall>=0.90, obs_exposure<=0.10; vanilla obs_exposure>=0.75), budget binding (>=75% of recall-method cells at >=0.8x budget), context difference (>=80% of the 12 pairs), no leakage (0 cells).
- fraction run: 48 cell(s) recorded.

## 4. History-Dependence Gate

- status: FAIL (locked: direct_history >= 3/4, no_history <= 1/4, per-group difference >= 2 cells)
  - route_contract: cells=4; direct_history_success=1.000; no_history_success=1.000; direct_ok=True; no_history_ok=False; diff_cells=0; diff_ok=False
  - serialization_contract: cells=4; direct_history_success=1.000; no_history_success=1.000; direct_ok=True; no_history_ok=False; diff_cells=0; diff_ok=False
  - retry_contract: cells=4; direct_history_success=0.250; no_history_success=0.500; direct_ok=False; no_history_ok=False; diff_cells=-1; diff_ok=False

## 5. Contradiction-State Gate

- status: FAIL (locked: adaptive corr_recall >= 0.90 and obs_exposure <= 0.10; vanilla obs_exposure >= 0.75)
  - vanilla_rag: cells=12; correction_recall=0.333; obsolete_fact_exposure=0.667
  - adaptive: cells=12; correction_recall=1.000; obsolete_fact_exposure=0.000

**Table D. Contradiction-state detail (`qwen2.5:7b`).**

| method | corr_recall | obs_exposure | 12-cell pass |
|---|---|---|---|
| vanilla_rag | 0.333 | 0.667 | FAIL |
| adaptive | 1.000 | 0.000 | PASS |

## 6. Budget-Binding Gate

- status: PASS (locked: >=75% of each recall-method cell at budget_ratio >= 0.8 x 96 = 76.8 tokens)
  - vanilla_rag: cells=12; bound_cells=12; fraction=1.000
  - adaptive: cells=12; bound_cells=12; fraction=1.000

**Table E. Budget / context-pair structure (`qwen2.5:7b`).**

| recall method | cells | bound_cells | bound fraction | threshold |
|---|---|---|---|---|
| vanilla_rag | 12 | 12 | 1.000 | >= 0.75 |
| adaptive | 12 | 12 | 1.000 | >= 0.75 |

adaptive-vs-vanilla context_sha pairs: 12 / 12 differ (threshold >= 80%).

## 7. Context-Difference Gate

- status: PASS (locked: >=80% of the 12 adaptive-vs-vanilla paired cells differ in context_sha)
  - pairs=12 differ=12 fraction=1.000

## 8. Leakage Gate

- status: PASS (locked: 0 leaked cells)
  - leaked_cells=0

## 9. Llama3.1:8b vs Qwen2.5:7b Gate Comparison

E28 compares the gate verdicts of the identical surface at the two tiers. A divergence isolates the model tier as the causal factor; a match keeps the surface-level explanation intact. No model is ranked.

**Table A. Gate-by-gate comparison (`llama3.1:8b` vs `qwen2.5:7b`).**

| gate | E27 @ llama3.1:8b | E28 @ qwen2.5:7b | outcome |
|---|---|---|---|
| history_dependence | FAIL | FAIL | both fail |
| contradiction_state | FAIL | FAIL | both fail |
| budget_binding | PASS | PASS | both pass |
| context_difference | PASS | PASS | both pass |
| no_leakage | PASS | PASS | both pass |

source calibration_valid: False; candidate calibration_valid: False.

## 10. Per-Group Results

**Table B. Per-method success and faithfulness diagnostics (`qwen2.5:7b`; E27 source success in brackets).**

| method | runs | E27 success | E28 success | E28 corr_recall | E28 obs_exposure | E28 mean budget ratio |
|---|---|---|---|---|---|---|
| no_history | 12 | 0.500 | 0.833 | 0.000 | 0.000 | 0.000 |
| direct_history | 12 | 0.667 | 0.750 | 1.000 | 0.000 | 0.548 |
| vanilla_rag | 12 | 0.167 | 0.417 | 0.333 | 0.667 | 0.953 |
| adaptive | 12 | 0.667 | 0.667 | 1.000 | 0.000 | 0.945 |

**Table C. Per-group history dependence (`qwen2.5:7b`).**

| group | direct_history success | no_history success | diff cells | pass |
|---|---|---|---|---|
| route_contract | 1.000 | 1.000 | 0 | FAIL |
| serialization_contract | 1.000 | 1.000 | 0 | FAIL |
| retry_contract | 0.250 | 0.500 | -1 | FAIL |

## 11. Retry-Contract Analysis

On the E27@`llama3.1:8b` run the retry_contract group failed the strongest (direct_history success 0/4 at the earlier tier). At `qwen2.5:7b` retry_contract direct_history success is 0.250 and no_history success is 0.500 (diff_cells=-1); the group FAILS the history-dependence gate. This is the key cell-by-cell probe for whether the earlier failure was a tier artifact of the coding model rather than a surface defect.

## 12. Calibration Verdict

**Calibration verdict (E28 @ qwen2.5:7b, frozen E27 protocol).**

- calibration_valid: **False**
- model: `qwen2.5:7b`; source calibration: e27_calibration@`llama3.1:8b`
- gates_passed: {'history_dependence': False, 'contradiction_state': False, 'budget_binding': True, 'context_difference': True, 'no_leakage': True}
- head_to_head_eligible: **False**
- rationale: pilot gates did not pass
- rationale: executed at qwen2.5:7b under the frozen E27 protocol

## 13. What E28 Establishes

The frozen E27 calibration surface is **not valid** at `qwen2.5:7b`: the locked pilot gates did not pass under the identical protocol. The E27 failure therefore does NOT isolate to the llama3.1:8b tier alone; the surface remains the leading explanation. No redesign is performed here - this is a measurement, not a fix.

## 14. What E28 Does Not Establish

- No adaptive-vs-vanilla result and no 'winner'; `adaptive_advances=false` (Phase 18 / E19) remains in force.
- No claim about any model tier other than `llama3.1:8b` and `qwen2.5:7b`.
- No claim about throughput, latency or serving cost.
- No claim that the surface is valid at any tier on evidence from one grid; gate thresholds are pre-registered and locked.

## 15. Eligibility for a Future Head-to-Head

`head_to_head_eligible = False` at `qwen2.5:7b`: the frozen surface does not behave as expected at either tier. No head-to-head may be planned; stop and bring the two-tier gate evidence to the architect for the surface-level redesign decision.
