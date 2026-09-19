# GSD Handoff — Phase 22 (E27 counterfactual calibration repair)

Generated: after committing the Phase-22 closure (single-policy calibration pilot ran, gates failed, hard stop — no winner, no head-to-head).

This doc is the single reviewer-readable artifact for the phase. Kept docs:
ROADMAP.md, STATE.md, PROJECT.md, config.json, codebase.

## Where we are

Phase 19 (commit `0354c5e`) closed the E19 experiment with a locked verdict:
`adaptive_advances = false` (adaptive 27/27, vanilla_rag 26/27, lapsing on a
135-cell primary grid). Phase 20 (E25) audited that artifact offline and set
four constraints (D42): obsolete-only evidence must be materially unsafe,
budgets must bind both recall methods below ~84–116 tokens, discrimination must
be possible above the 96.3% vanilla ceiling, and E19 must not be read as
adaptive superiority.

**Phase 21 built E26** (contradiction-unsafe suite, budgets 64/96/128). Its
48-cell pilot failed two gates — history_dependence (two of three families
unsolvable even with the full current-policy oracle history at llama3.1:8b)
and contradiction_state (vanilla recency dropped the superseded facts, obs
exposure 0.29 < 0.75) — so the 54-cell full grid never ran and
`adaptive_beats_vanilla = False`. The E26 suite, runner, results and report are
byte-identical and are never re-run.

**Phase 22 builds E27** — a *single-policy* calibration surface that repairs
both failures: every task is reduced to exactly ONE decisive obsolete policy
fact vs ONE current correction with four distractors, so the conflict point is
small and the two sides never both fit the 96-token budget. E27 only calibrates
the surface; it does not run a head-to-head grid and declares no winner. The
48-cell pilot ran with five locked gates. **Two gates failed → calibration
surface is INVALID for a head-to-head at this model tier; E27 stops.**

## How it was run

```bash
python -m data.e27_calibration_suite --write        # write fixtures
python -m experiments.e27_calibration --validate    # offline gates (0 LLM)
python -m experiments.e27_calibration --pilot       # 48 cells + 5 locked gates
python -m pytest tests/ -q                          # 354 passing (74 new E27)
```

Outputs written (new, force-added): `data/e27_calibration_suite.py` +
`data/e27_calibration_tasks/*`; `experiments/e27_calibration.py`;
`tests/test_e27_calibration_suite.py` +
`tests/test_e27_calibration.py`;
`experiments/results/e27_calibration.json` + `_calibration_report.md`.

## Locked earlier results (never modified, carried for the reviewer)

- **Phase 19 (E23+E24):** with the counterfactual adapter repair,
  `correction_identity` = 27/27 PASS, `embedding_consistency` = 27/27, primary
  grid = 135/135, and **`adaptive_advances = false`** — adaptive = 27/27,
  vanilla_rag = 26/27, llm_summarization 14/27, raw_clipped 13/27,
  sliding_window 9/27. One negative-control cell recovered separately
  (`write_retry / seed=3 / budget=256 / llm_summarization`); combined
  accounting 225/225. Artifact `e19_coding_generalization_full_repaired.json`
  byte-identical (E25 re-verified).
- **Phase 20 (E25):** budget geometry audit — budgets never bind either recall
  arm below ~84–116 tokens; constraints D42: obsolete-only evidence must be
  unsafe, budgets must bind both methods, discrimination above the 96.3%
  vanilla ceiling, never read E19 as adaptive superiority. Declares no winner.
- **Phase 21 (E26):** 48-cell pilot at llama3.1:8b (budget 96, seeds 1–2) —
  budget_binding/context_difference/no_leakage PASS; history_dependence FAIL
  (release_adapter direct 0.0/4, message_adapter 0.5/4, invoice_adapter
  0.75/4 vs no_history 0.0) and contradiction_state FAIL (adaptive corr_recall
  0.9375 / obs 0.0; vanilla obs 0.2917) → `adaptive_beats_vanilla = False`;
  54-cell full grid never run.

## E27 suite (offline fixture gates — all 6 variants PASS)

- Groups `route_contract` / `serialization_contract` / `retry_contract`,
  variants A and B per group; A/B differ ONLY by history contract (workspace
  and prompt byte-identical within a pair; hidden tests and gold patches
  differ).
- Each variant: exactly 1 obsolete policy fact (turns 80–150, >300 turns old,
  50–56 shared-word tokens), 1 current correction (turns 470–530, explicit
  `supersedes_turn`, 50–56 tokens), 4 distractors (18–24 tokens);
  obsolete+current token ratio 103–110 (>96) so the two sides of the conflict
  can never both fit the budget.
- Offline gate per variant (StaticCoder, 0 LLM): base workspace FAILS hidden,
  `gold.patch` PASSES, `obsolete.patch` FAILS (obsolete-only evidence is
  unsafe). Forbidden contract terms (`lane-a`/`lane-b`,
  `future_key`/`preserved`/`dropped`, `retry`/`retries`/`exactly once`) absent
  from workspace/prompt; histories seed-1 deterministic; no leakage.

## E27 pilot (48 cells, llama3.1:8b @ localhost:11434, budget 96, seeds 1–2)

Predeclared gates, spec-locked (never retuned):

| gate | threshold | result |
| --- | --- | --- |
| history_dependence | per group direct≥0.75, no_history≤0.25, diff≥2 | **FAIL (all 3 groups)** |
| contradiction_state | adaptive corr_recall≥0.90 ∧ obs≤0.10; vanilla obs≥0.75 | **FAIL** |
| budget_binding | ≥75% of each recall method's cells ≥0.8×96 tokens | PASS (1.0 / 1.0) |
| context_difference | ≥80% of 12 paired adaptive-vs-vanilla cells differ | PASS (1.0) |
| no_leakage | 0 leakage detections | PASS (0) |

Why the two failures are findings, not measurement noise:

- **history_dependence:** route_contract and serialization_contract reach
  direct 1.0 / no_history 0.75 — inside the gate on direct, but no_history is
  at the ≥0.25 cap exactly, so the diff collapses to 1 (< 2). retry_contract
  is the deeper failure: direct_history is 0.0/4, so even a single decisive
  obsolete-vs-current contradiction is not enough for llama3.1:8b to derive
  the record-then-ignore-a-retry behavior. Retry-style tasks cannot be
  calibrated as one-shot policy facts at this model tier.
- **contradiction_state:** the adaptive side is clean (correction_recall 1.0 /
  obsolete_fact_exposure 0.0), and vanilla improved vs E26 (obs_exposure
  0.2917 → 0.6667) but still misses the 0.75 bar: vanilla retrieved the
  current correction on 4/12 cells (serialization_variant_A 2, retry_variant_A
  2), so the obsolete-only contradiction signal is not consistent across the
  grid.

Per-method pilot success (Table D): adaptive 0.67, direct_history 0.67,
no_history 0.50, vanilla_rag 0.17. Group × method success (Table C):
route 0.75/1.00/0.00/1.00, serialization 0.75/1.00/0.50/1.00, retry
0.00/0.00/0.00/0.00.

## Verdict (predeclared, phase 22)

- `calibration_valid = grid_complete AND fixture_gates_passed AND
  pilot_gates_passed`; `full_head_to_head_eligible = calibration_valid`.
- **Two pilot gates did NOT pass → `calibration_valid = False`,
  `full_head_to_head_eligible = False`. E27 declares no winner and claims no
  adaptive-vs-RAG result.** The verdict is a locked, negative result — no
  retuning, no gate weakening, no production change.

## Remaining questions (for any next phase)

1. **Solvability floor:** retry_contract is unsolvable even from the single
   decisive gold fact (direct_history 0/4). Should a future calibration
   (a) gate on the model tier being able to solve the direct-history cell
   BEFORE running the grid, (b) drop retry-style behaviors from single-fact
   calibration, or (c) use multi-fact decisive policies so no_history falls
   without an oracle?
2. **no_history floor:** route/serialization no_history is 3/4 — just over the
   cap. Is a stricter no_history gate (≤0/4) honest for this tier, or is the
   cap the right contract and the families under-specified?
3. **Contradiction consistency:** vanilla obs_exposure 0.67 is closer to the
   bar but variant-A serialization/retry cells still surface the current
   correction. Does the comparator need an explicit recency-neutral whole-
   history mode, or is a different baseline required?
4. **Model tier:** every E27 failure is consistent with llama3.1:8b being too
   weak to separate one-shot policy contradictions. A future head-to-head
   would need a stronger or cheaper tier re-certified on the same offline
   gates before any run.

## Immutable (Phase-22 contract)

- All E26 artifacts (suite, runner,
  `e26_discriminative_coding_benchmark_pilot.json` + `_pilot_report.md`) and all
  E19/E20/E24/E25 results and reports are byte-for-byte unchanged
  (git diff empty).
- `config.yaml`, `memory_optimizer/`, `baselines/`, `server.py` — untouched.

## Honest verdict (reviewer-confirmable)

- The E27 suite is offline-certified (all 6 variants: base fails / gold passes /
  obsolete.patch fails) and the pilot is complete (48/48 cells).
- The pilot stopped the phase: budget_binding (both methods 1.0 bound),
  context_difference (12/12) and no_leakage (0) passed; history_dependence and
  contradiction_state failed for structural reasons (one-shot direct_history
  unsolvability on retry; no_history at the cap on the other two groups; vanilla
  obs_exposure 0.67 < 0.75). No adaptive advantage appeared anyway.
- `calibration_valid = False`, `full_head_to_head_eligible = False`; no winner,
  no head-to-head, no ranking change; production + E26 + earlier artifacts
  byte-identical.

## Next decision (reviewer)

Decide whether a future benchmark line should (a) require a new model tier
re-certified on the same offline gates, (b) re-specify families so
no_history falls harder while direct_history stays solvable (e.g. multi-fact
decisive policies), or (c) stop the benchmark line. E27 itself does none of
these — its calibration was a negative gate.

## Cryptographic checkpoint

- Checkpoint branch: `checkpoint/phase-22-e27-calibration-complete`
- Checkpoint SHA: the Phase-22 closure commit on master.

## Restore

```bash
git checkout checkpoint/phase-22-e27-calibration-complete
```

## Do-not-regress (locked)

1. Do not tune `config.yaml` (dual_score, retention, decay, thresholds, top_k,
   embedding model, task affinity all frozen).
2. Original E19/E20/E24/E25/E26 JSON/report artifacts immutable.
3. Any new coding fixture must pass the offline base-hidden-test gate BEFORE it
   may claim history dependence.
4. History dependence of a coding benchmark requires a reproducible
   counterfactual pair at the claimed model tier (no_history vs direct_history).
5. E19 full-grid results are only eligible under the E20-validated families;
   stochastic no-history draws certify nothing.
6. Benchmark separation requires BOTH contradiction sensitivity AND
   method-specific budget pressure (D42) — one without the other cannot
   discriminate adaptive from vanilla_rag above the ceiling.
7. E26 hard-stop rule stays locked: no 54-cell full grid / head-to-head unless
   a pilot records all its gates passed; no retuning of pilot gates to pass.
8. E27 declares no winner and never claims an adaptive-vs-RAG result;
   `calibration_valid` was False → no head-to-head may be built on this
   surface at llama3.1:8b.