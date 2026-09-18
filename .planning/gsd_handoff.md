# GSD Handoff — Phase 18 (E19 aligned to the E20-validated benchmark + full grid)

Generated: after committing the Phase-18 E19 full-grid run.

This doc is the single reviewer-readable artifact for the phase. Kept docs:
ROADMAP.md, STATE.md, PROJECT.md, config.json, codebase.

## Where we are

Phase 17 revalidated the E20 counterfactual benchmark (routing_policy replaced
the fixture that was solvable from the workspace alone; all three groups
discriminated → `history_dependence_benchmark = VALID`, `e19_full_grid_eligible
= TRUE`). **Phase 18 is the reviewer-approved run of the E19 full grid**, with
the experiment itself re-anchored onto the validated benchmark:

- E19's primary tasks are fixed **Variant-A** instances of the three E20
  counterfactual families: `routing_policy`, `retry_policy`,
  `serialization_policy` (each ≥600-turn, history-gated, E20-certified).
- **Gate C (`history_dependence`) is the frozen E20 counterfactual history
  calibration** (via `check_e20_calibration()`, reading
  `e20_counterfactual_history_repair.json`), not stochastic E19 no-history
  draws. E19's no_history/direct_history/full_context diagnostics are
  descriptive only.
- Full 225-cell grid: 135 primary (3 × 3 seeds × 3 budgets × 5 methods) + 90
  negative-control (cache_readonly, write_retry), seeds [1,2,3], budgets
  [256,512,1024], methods [raw_clipped, sliding_window, llm_summarization,
  vanilla_rag, adaptive] at llama3.1:8b @ localhost:11434.

## How it was run

```bash
python experiments/e22_e19_full_grid.py --full --force   # 225 cells
python experiments/e22_e19_full_grid.py --full --resume  # if interrupted
```

`e22` is a thin runner (swaps `e19.OUT_JSON`/`OUT_REPORT` to the `_full*`
paths, calls `e19.main`, restores in `finally`). One cell transiently failed
with an Ollama 500 during the first pass and was completed by a `--resume`
pass (final grid = 225/225 records).

## Gate results (10 gates)

| gate | result | note |
| --- | --- | --- |
| A embedding_consistency | PASS | 27/27 cells consistent |
| B correction_identity | **FAIL** | 0/27 identity-ok; obsolete fact survives in all 27 correction-bearing cells |
| C history_dependence | PASS | E20 counterfactual history calibration |
| D method_separation | PASS | max distinct contexts 4 |
| E no_leakage | PASS | 0 leaking runs |
| F gold_passes | PASS | 5/5 (3 primary + 2 negative) |
| G budget_pressure | PASS | history > max budget; raw grows; fills 60% |
| H real_summarization | PASS | 324 summary_update_calls |
| I adaptive_production_path | PASS | 27 adaptive runs |
| J unit_tests_pass | PASS | 185 passed |

`gates_all_passed = False` (Gate B) → `adaptive_advances = False` by the
locked, predeclared verdict logic. Nothing was weakened or tuned to change
this.

## Why Gate B fails (measured, not a change)

The counterfactual histories' correction facts do **not** restate the prior
fact with full word coverage, so the unchanged `_is_supersession`
consolidation heuristic (in `memory_optimizer/compression.py` — requires all
of the old fact's words to recur) never fires. Verified directly on the raw
`build_variant` fixture with zero E19 modifications:
`_is_supersession(obs_text, cur_text) == False` for e.g. routing_policy
(`rp.a.inter.001` → `rp.a.corr.001`). Per the phase contract this is
recorded, reported, and left un-tuned — no claims are made either way.

## Paired numbers (primary only, 27 cells per baseline)

| vs | adaptive | baseline | mean diff |
| --- | --- | --- | --- |
| raw_clipped | 27/27 | 13/27 | +0.52 |
| sliding_window | 27/27 | 9/27 | +0.67 |
| llm_summarization | 27/27 | 14/27 | +0.48 |
| vanilla_rag | 27/27 | 27/27 | 0.00 |

## New artifacts (Phase 18, force-added)

- `experiments/results/e19_coding_generalization_full.json` sha256
  `116159cc285e0ac8866e8d72f94104176f5c89289d8228939a87dbe68b402381`
- `experiments/results/e19_coding_generalization_full_report.md` sha256
  `9c9fda34614e9264b42086b8bf8684952ee7543cfc2779f137bcd823c164c904`

## Immutable (Phase-18 contract)

- `e19_coding_generalization.json` + `_report.md`, `e20_counterfactual_history.json`,
  `e20_counterfactual_history_repair.json` + `_repair_report.md` — byte-for-byte
  unchanged (git diff empty).
- `config.yaml`, `memory_optimizer/`, `server.py` — untouched.

## Honest verdict (reviewer-confirmable)

- Grid: 225/225 records; exactly 5 task families; no
  user_ids/validation_pure/transaction_atomicity as primary records.
- Gates: 9/10 pass; Gate B (`correction_identity`) FAILS 0/27.
- `gates_all_passed = False`, `adaptive_beats_all_baselines = False`,
  `adaptive_advances = False` (locked logic; no claims).
- E20 certification (`check_e20_calibration()`) `passed: True`, groups ==
  PRIMARY_TASKS.
- Tests: 185 passing (+8 in `tests/test_e19_full_grid_alignment.py`).

## Next decision (reviewer)

whether to treat Gate B's uniform obsolete-fact retention in the counterfactual
histories as (a) a fixture wording issue (correction phrasing that would allow
the locked supersession heuristic to fire), (b) a measurement of
`_is_supersession` strictness, or (c) leave as-is. This phase takes no position.

## Cryptographic checkpoint

- Checkpoint branch: `checkpoint/phase-18-e19-full-grid-complete`
- Checkpoint SHA: the Phase-18 commit on master.

## Restore

```bash
git checkout checkpoint/phase-18-e19-full-grid-complete
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