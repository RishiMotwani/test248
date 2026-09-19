# Checkpoint — Phase 21 complete (E26 discriminative benchmark, pilot stopped)

Status: **CLOSED** — phase completed; step value negative but recorded honestly.

## Scope

Phase 21 (E26): build a D42-aligned discriminative adaptive-vs-vanilla RAG
coding benchmark, certify it offline, run the 48-cell pilot with five
predeclared gates, and gate the 54-cell full grid on an all-gates-pass pilot.
Any pilot gate failure is a hard stop with `adaptive_beats_vanilla = False`.

## What ran

1. `data/discriminative_coding_suite.py` + `data/discriminative_coding_tasks/*`
   (3 groups x 2 variants; 4 current corrections at turns 420-540 with
   `supersedes_turn` metadata, 4 obsolete facts at turns 70-180 (>300 turns),
   4 distractors; 18-28-token facts; workspace+prompt byte-identical per pair).
2. Offline fixture gate (StaticCoder, 0 LLM) on all 6 variants: base workspace
   FAILS hidden, `gold.patch` PASSES, `obsolete.patch` FAILS — all PASS.
3. `experiments/e26_discriminative_coding_benchmark.py` — runner (run_one,
   gold_check, 5 gates, grid completeness, paired bootstrap CI, verdict,
   report generators; full mode hard-gated on a passing pilot artifact).
4. 48-cell pilot at llama3.1:8b (budget 96, seeds 1-2) — 48/48 cells.

## Pilot gates (locked thresholds, never retuned)

| gate | result |
| --- | --- |
| history_dependence (dh>=3/4, nh<=1/4, diff>=2 per group) | FAIL |
| contradiction_state (adaptive corr_recall>=0.75 & obs<=0.25; vanilla obs>=0.75) | FAIL |
| budget_binding (>=0.75 cells >=0.8x96, both recall methods) | PASS (1.0/1.0) |
| context_difference (>=80% of 12 paired cells differ) | PASS |
| no_leakage (0 detections) | PASS (0) |

Details: release_adapter direct_history 0.0/4, message_adapter 0.5/4,
invoice_adapter 0.75/4 (no_history 0.0 everywhere); vanilla_rag
obsolete_fact_exposure 0.2917 (< 0.75), correction_recall 0.375; adaptive
correction_recall 0.9375 / obsolete_fact_exposure 0.0. Per-method pilot
success: adaptive 6/12, direct_history 5/12, vanilla_rag 0/12, no_history 0/12.

## Verdict

- `adaptive_beats_vanilla = False` (pilot_gates_passed = False).
- 54-cell full grid NOT run (hard stop). No retuning, no gate weakening.

## Artifacts (force-added, `experiments/results/`)

- `e26_discriminative_coding_benchmark_pilot.json`
- `e26_discriminative_coding_benchmark_pilot_report.md`

## Immutables (byte-identical, git diff empty)

E19/E20/E24/E25 JSON + reports; `config.yaml`; `memory_optimizer/**`;
`baselines/**`; `server.py`.

## Tests

Full suite: `python -m pytest tests/ -q` -> 279 passing (62 new E26 offline
tests; no Ollama/network).