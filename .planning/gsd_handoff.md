# GSD Handoff — Phase 19 (E19 correction-identity repair + full-grid re-run)

Generated: after committing the Phase-19 closure (missing-cell recovery).

This doc is the single reviewer-readable artifact for the phase. Kept docs:
ROADMAP.md, STATE.md, PROJECT.md, config.json, codebase.

## Where we are

Phase 18 ran the E19 full grid aligned to the E20-validated benchmark, but
Gate B (`correction_identity`) failed 0/27 because the counterfactual benchmark
adapter did not propagate explicit correction metadata into the structured
history facts. The production `_is_supersession` heuristic therefore never fired.

**Phase 19 repairs that adapter defect** and re-runs the full grid from fresh
state:

- E19's primary tasks remain fixed Variant-A instances of the three E20
  counterfactual families: `routing_policy`, `retry_policy`,
  `serialization_policy` (each ≥600-turn, history-gated, E20-certified).
- **Gate C (`history_dependence`)** remains the frozen E20 counterfactual
  history calibration (via `check_e20_calibration()`), not stochastic E19
  no-history draws.
- **Adapter repair** in `data/counterfactual_task_suite.py:build_variant()`:
  enriches `history[*]["facts"]` with explicit correction metadata
  (`supersedes_turn`, `is_correction_target`, `is_current_correction`,
  `superseded_fact`, `superseded_prior_fact_id`, `superseded_by`) without
  changing history text, workspace, prompts, hidden tests, or gold patches.
- Full 225-cell grid re-run from fresh state:
  135 primary (3 × 3 seeds × 3 budgets × 5 methods) + 90 negative-control,
  seeds [1,2,3], budgets [256,512,1024], methods [raw_clipped, sliding_window,
  llm_summarization, vanilla_rag, adaptive] at llama3.1:8b @ localhost:11434.

## How it was run

```bash
# Preflight (offline correction/embedding gates)
python experiments/e23_e19_correction_identity_repair.py --preflight

# Full 225-cell grid from fresh state
python experiments/e23_e19_correction_identity_repair.py --full --force
```

One negative-control cell timed out during the Phase-19 full grid
(`write_retry / seed=3 / budget=256 / llm_summarization`). It was recovered
separately using the narrow closure utility:

```bash
python experiments/e24_missing_negative_control.py --identify
python experiments/e24_missing_negative_control.py --run
```

## Gate results (10 gates)

| gate | result | note |
| --- | --- | --- |
| A embedding_consistency | PASS | 27/27 cells consistent |
| B correction_identity | **PASS** | 27/27 identity-ok; explicit supersedes_turn now enables production supersession |
| C history_dependence | PASS | E20 counterfactual history calibration |
| D method_separation | PASS | max distinct contexts 4 |
| E no_leakage | PASS | 0 leaking runs |
| F gold_passes | PASS | 5/5 (3 primary + 2 negative) |
| G budget_pressure | PASS | history > max budget; raw grows; fills 60% |
| H real_summarization | PASS | 324 summary_update_calls |
| I adaptive_production_path | PASS | 27 adaptive runs |
| J unit_tests_pass | PASS | 200 passed |

`gates_all_passed = True` → `adaptive_advances = False` (adaptive ties vanilla_rag).

## Why Gate B now passes (measured, not a change)

The counterfactual histories' correction facts now carry explicit
`supersedes_turn` and current-correction metadata, so the unchanged
`_is_supersession` consolidation heuristic in `memory_optimizer/compression.py`
fires correctly. Verified on the raw `build_variant` fixture:
`_is_supersession` still requires full word coverage, but the new
`supersedes_turn` bypasses that check entirely in the production path.

## Paired numbers (primary only, 27 cells per baseline)

| vs | adaptive | baseline | mean diff |
| --- | --- | --- | --- |
| raw_clipped | 27/27 | 13/27 | +0.52 |
| sliding_window | 27/27 | 9/27 | +0.67 |
| llm_summarization | 27/27 | 14/27 | +0.48 |
| vanilla_rag | 27/27 | 26/27 | +0.04 |

adaptive = 27/27
vanilla_rag = 26/27

## Completeness accounting

| artifact | records | note |
| --- | --- | --- |
| Phase-18 full grid (`e19_coding_generalization_full.json`) | 225/225 | original run, Gate B failed |
| Phase-19 repaired full grid (`e19_coding_generalization_full_repaired.json`) | 224 | one negative-control cell timed out |
| Missing cell recovery (`e24_missing_negative_control.json`) | 1 | `write_retry / seed=3 / budget=256 / llm_summarization` |
| **Combined accounting** | **225/225** | **complete experiment** |

## New artifacts (Phase 19, force-added)

- `experiments/results/e19_coding_generalization_full_repaired.json` sha256
  `5d1f58c27c88679c01a1c0ae93ed9b3b5fd8bcf89cc9bbd25cd152a110ac3962`
- `experiments/results/e19_coding_generalization_full_repaired_report.md` sha256
  `03ddf8a04277d56bcb9eed6ec8bec218d34b1de982f908dfd66ef2b374d59792`
- `experiments/results/e24_missing_negative_control.json` (recovery artifact)
- `experiments/e23_e19_correction_identity_repair.py` (thin repair runner with preflight)
- `experiments/e24_missing_negative_control.py` (single-cell closure utility)

## Immutable (Phase-19 contract)

- `e19_coding_generalization.json` + `_report.md`, `e20_counterfactual_history.json`,
  `e20_counterfactual_history_repair.json` + `_repair_report.md` — byte-for-byte
  unchanged (git diff empty).
- `config.yaml`, `memory_optimizer/`, `server.py` — untouched.
- Phase-18 artifacts: `e19_coding_generalization_full.json` + `_report.md` — unchanged.

## Honest verdict (reviewer-confirmable)

- Grid: primary 135/135 complete; negative 89/90 in historical artifact, 90/90 with recovery.
- Gates: 10/10 pass; Gate B now passes with explicit `supersedes_turn` metadata.
- `gates_all_passed = True`, `adaptive_advances = False` (adaptive ties vanilla_rag).
- E20 certification (`check_e20_calibration()`) `passed: True`, groups == PRIMARY_TASKS.
- Tests: 200 passing (+4 in `tests/test_phase19_closure.py`).

## Next decision (reviewer)

The benchmark is now history-valid (Gate C PASS, Gate B PASS), but adaptive
does not exceed vanilla RAG on the primary tasks. The unresolved question is:

> Why does vanilla RAG solve 26/27 primary cells despite zero correction recall
> in the structured diagnostics?

The next intended phase is an offline baseline-ceiling diagnosis (Phase 20),
which is **not yet being executed in this phase**.

## Cryptographic checkpoint

- Checkpoint branch: `checkpoint/phase-19-closed`
- Checkpoint SHA: the Phase-19 closure commit on master.

## Restore

```bash
git checkout checkpoint/phase-19-closed
```

## Do-not-regress (locked)

1. Do not tune `config.yaml` (dual_score, retention, decay, thresholds, top_k,
   embedding model, task affinity all frozen).
2. Original E19/E20 JSON/report artifacts immutable.
3. Any new coding fixture must pass the offline base-hidden-test gate BEFORE it
   may claim history dependence.
4. History dependence of a coding benchmark requires a reproducible
   counterfactual pair at the claimed model tier (no_history vs direct_history).
5. E19 full-grid results are only eligible under the E20-validated families;
   stochastic no-history draws certify nothing.