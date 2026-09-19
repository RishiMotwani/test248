# Checkpoint — Phase 22 complete (E27 counterfactual calibration, pilot stopped)

Status: **CLOSED** — phase completed; step value negative but recorded honestly.

## Scope

Phase 22 (E27): repair the E26 calibration surface (Phase 21, D43) by reducing
every task to exactly ONE decisive obsolete policy fact vs ONE current
correction with four distractors per variant at a single 96-token budget, so
the two sides of the conflict never both fit. Certify the suite offline, run
the 48-cell pilot with five predeclared gates, and judge calibration validity
(`calibration_valid = grid_complete AND fixture_gates AND pilot_gates`).
Any pilot gate failure invalidates the calibration — `full_head_to_head_eligible
= calibration_valid`. E27 never runs a head-to-head and declares no winner.

## What ran

1. `data/e27_calibration_suite.py` + `data/e27_calibration_tasks/*`
   (3 groups x 2 variants: route_contract, serialization_contract,
   retry_contract; 1 obsolete policy fact at turns 80-150 (>300 turns, 50-56
   tokens), 1 current correction at turns 470-530 with `supersedes_turn` (50-56
   tokens), 4 distractors (18-24 tokens); obsolete+current ratio 103-110 > 96;
   workspace+prompt byte-identical per pair). Written via
   `python -m data.e27_calibration_suite --write`.
2. Offline fixture gate (StaticCoder, 0 LLM) on all 6 variants: base workspace
   FAILS hidden, `gold.patch` PASSES, `obsolete.patch` FAILS — all PASS
   (`python -m experiments.e27_calibration --validate`).
3. `experiments/e27_calibration.py` — runner (run_one, StaticCoder gold checks,
   5 locked gates, grid completeness, per-group/method tables, verdict,
   15-section report; `--validate`/`--pilot`/`--resume`/`--report-only`).
4. 48-cell pilot at llama3.1:8b (budget 96, seeds 1-2) — 48/48 cells complete.

## Pilot gates (locked thresholds, never retuned)

| gate | result |
| --- | --- |
| history_dependence (direct>=0.75, no_history<=0.25, diff>=2 per group) | FAIL (all 3 groups) |
| contradiction_state (adaptive corr_recall>=0.90 & obs<=0.10; vanilla obs>=0.75) | FAIL |
| budget_binding (>=0.75 cells >=0.8x96, both recall methods) | PASS (1.0/1.0) |
| context_difference (>=80% of 12 paired cells differ) | PASS (1.0) |
| no_leakage (0 detections) | PASS (0) |

Details: route_contract and serialization_contract direct_history 1.0 vs
no_history 0.75 (diff 1 < 2; no_history exactly at the 0.25 cap); retry_contract
direct_history 0.0/4 (single decisive fact insufficient at this tier);
vanilla_rag obsolete_fact_exposure 0.6667 (< 0.75) with correction_recall
0.3333 (current correction surfaced on 4/12 cells — serialization_variant_A 2,
retry_variant_A 2); adaptive correction_recall 1.0 / obsolete_fact_exposure
0.0. Per-method pilot success: adaptive 8/12 (0.67), direct_history 8/12
(0.67), no_history 6/12 (0.50), vanilla_rag 2/12 (0.17).

## Verdict

`calibration_valid = False`, `full_head_to_head_eligible = False`. No winner,
no head-to-head, no adaptive-vs-RAG claim. No retuning, no gate weakening, no
production change.

## Artifacts

- New/force-added: `data/e27_calibration_suite.py`, `data/e27_calibration_tasks/*`,
  `experiments/e27_calibration.py`, `tests/test_e27_calibration_suite.py`,
  `tests/test_e27_calibration.py`, `experiments/results/e27_calibration.json`,
  `experiments/results/e27_calibration_report.md`.
- Immutable (byte-identical, git diff empty): `memory_optimizer/**`,
  `baselines/**`, `config.yaml`, `server.py`, E26 suite/runner/results, and all
  E19/E20/E24/E25 results/reports.
- Tests: 354 passing (74 new E27).

## Reasons the phase stopped here

The calibration surface cannot certify a head-to-head at llama3.1:8b: (1) the
decisive one-shot policy is not enough for direct_history on retry_contract
and no_history is at the cap (3/4) on the other two families, so history
dependence is not separated; (2) vanilla's obsolete-only contradiction signal
is inconsistent (0.67 < 0.75 obs_exposure). Both are structural findings at
this model tier and design, not tunable within the locked contract.

## Restore

```bash
git checkout checkpoint/phase-22-e27-calibration-complete
```