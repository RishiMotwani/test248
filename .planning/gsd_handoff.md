# GSD Handoff — Phase 21 (E26 discriminative adaptive-vs-vanilla benchmark)

Generated: after committing the Phase-21 closure (pilot ran, gates failed, hard stop).

This doc is the single reviewer-readable artifact for the phase. Kept docs:
ROADMAP.md, STATE.md, PROJECT.md, config.json, codebase.

## Where we are

Phase 19 (commit `0354c5e`) closed the E19 experiment with a locked verdict:
after the counterfactual adapter repair (explicit `supersedes_turn` +
current-correction metadata), `correction_identity` = 27/27 PASS,
`embedding_consistency` = 27/27, primary grid = 135/135, and
**`adaptive_advances = false`** (adaptive = 27/27, vanilla_rag = 26/27,
llm_summarization 14/27, raw_clipped 13/27, sliding_window 9/27). One
negative-control cell was recovered separately (`write_retry / seed=3 / budget=256 / llm_summarization`), giving combined 225/225 accounting.

Phase 20 (E25) audited that locked artifact offline and set four constraints
for any next benchmark (D42): obsolete-only evidence must be materially unsafe
for the hidden test, budgets must actually bind both recall methods below the
observed ~84–116-token range, discrimination must be possible above the 96.3%
vanilla ceiling, and E19 must not be read as adaptive superiority.

**Phase 21 builds E26** — the first benchmark built to those constraints. A
D42-aligned discriminative suite (3 groups × 2 variants; contradiction-unsafe
hidden tests; budgets 64/96/128) was certified offline on all 6 variants, and
the 48-cell pilot ran at llama3.1:8b with the predeclared five gates. **The
pilot failed two gates → hard stop. The 54-cell full grid was NOT run, and
`adaptive_beats_vanilla = False`.**

## How it was run

```bash
python -m data.discriminative_coding_suite --force        # write fixtures
python -m experiments.e26_discriminative_coding_benchmark --validate # offline gates
python -m experiments.e26_discriminative_coding_benchmark --pilot      # 48 cells + gates
python -m pytest tests/ -q                                # 279 passing (62 new E26)
```

Outputs written (new, force-added): `data/discriminative_coding_suite.py` +
`data/discriminative_coding_tasks/*`; `experiments/e26_discriminative_coding_benchmark.py`;
`tests/test_discriminative_coding_suite.py` + `tests/test_e26_discriminative_coding_benchmark.py`;
`experiments/results/e26_discriminative_coding_benchmark_pilot.json` +
`_pilot_report.md`.

## Locked Phase-19 result (re-verified by E25, never modified)

| method | primary success | correction state |
| --- | --- | --- |
| adaptive | 27/27 | 27x CLEAN_CURRENT |
| vanilla_rag | 26/27 | 27x OBSOLETE_ONLY (26/27 succeed) |
| llm_summarization | 14/27 | 16 OBSOLETE_ONLY / 6 NEITHER / 5 CLEAN_CURRENT |
| raw_clipped | 13/27 | 27x NEITHER |
| sliding_window | 9/27 | 27x NEITHER |

`gates_all_passed = true`; `verdict.adaptive_advances = false`. Single
vanilla_rag failure: `serialization_policy / seed=2 / budget=1024 /
OBSOLETE_INFORMATION_USED` (102 tokens, context SHA `50c96fd1d0bf2f96`).

## E26 suite (offline fixture gates — all 6 variants PASS)

- Groups `release_adapter` / `invoice_adapter` / `message_adapter`, variants A
  and B per group; A/B differ ONLY by history contract (workspace and prompt
  byte-identical within a pair; hidden tests and gold patches differ).
- Facts: 4 current corrections (turns 420–540, `supersedes_turn`,
  `is_correction_target`, `is_current_correction`, `superseded_fact`,
  `superseded_prior_fact_id`), 4 obsolete policy facts (turns 70–180, >300
  turns old, `superseded_by`), 4 distractors; 18–28 shared-word tokens each;
  probe strings are substrings of their fact texts.
- Offline gate per variant (StaticCoder, 0 LLM): base workspace FAILS hidden,
  `gold.patch` PASSES, `obsolete.patch` FAILS. Forbidden contract terms
  (`pending_releases`, `journal`, `MISCELLANEOUS`, `confirmation`, ...) absent
  from workspace/prompt; `history.txt` seed-1 deterministic; no leakage.

## E26 pilot (48 cells, llama3.1:8b @ localhost:11434, budget 96, seeds 1–2)

Predeclared gates, spec-locked (never retuned):

| gate | threshold | result |
| --- | --- | --- |
| history_dependence | per group dh≥3/4, nh≤1/4, diff≥2 | **FAIL** |
| contradiction_state | adaptive corr_recall≥0.75 ∧ obs≤0.25; vanilla obs≥0.75 | **FAIL** |
| budget_binding | ≥75% of each recall method's cells ≥0.8×96 tokens | PASS (1.0 / 1.0) |
| context_difference | ≥80% of 12 paired adaptive-vs-vanilla cells differ | PASS |
| no_leakage | 0 leakage detections | PASS (0) |

Why the two failures are findings, not measurement noise:

- **history_dependence:** even the full current-policy oracle history
  (`direct_history`) fails to clear the hidden tests on two of three families —
  `release_adapter` 0.0/4, `message_adapter` 0.5/4 (only `invoice_adapter`
  0.75/4). `no_history` is 0.0 everywhere. The workspace + gold contract is
  internally inconsistent with what llama3.1:8b will produce on those two
  families, so the benchmark cannot separate history-aware methods at this
  model tier.
- **contradiction_state:** the adaptive side is clean (correction_recall 0.9375,
  obsolete_fact_exposure 0.0), but vanilla_rag reached obsolete_fact_exposure
  only 0.2917 (< 0.75 required) with correction_recall 0.375: at 96 tokens the
  recency-weighted vanilla retriever dropped the superseded facts instead of
  drowning the current contract in them — the intended obsolete-only signal
  never formed. Per-method pilot success: adaptive 6/12 (0.50), direct_history
  5/12 (0.42), vanilla_rag 0/12, no_history 0/12.

## Verdict (predeclared, phase 21)

- `adaptive_beats_vanilla = pilot_gates_passed AND full_grid_complete AND
  fixture_gates_passed AND bootstrap_ci_lower_bound > 0` (CI = paired
  `bootstrap_ci_mean_diff` over 27 group/seed/budget-paired cells).
- **Pilot gates did NOT pass → hard stop. The 54-cell full grid was not run.
  `adaptive_beats_vanilla = False`.** The verdict is a locked, negative result —
  no retuning, no gate weakening, no production change.

## Remaining questions (for any next phase)

1. **Ceiling effect:** at llama3.1:8b, two of three E26 families are unsolvable
   even with the full current-policy oracle history. Which task families are
   within the model tier's solve capability yet still punish obsolete-only
   evidence? (E16-style calibration of family difficulty must precede any
   claim.)
2. **Contradiction formation:** vanilla_rag at 96 tokens did not surface the
   superseded facts (obs_exposure 0.29). Should the contract require
   *recency-penalized* baseline retrieval to actually inject the obsolete
   facts, or is a different baseline the right comparator?
3. **Budget geometry:** with both recall methods bound at 96 (1.0 fraction),
   D42's binding constraint is satisfied — but only for the two recall arms;
   the failure is upstream (task solvability + contradiction signal), not the
   budget range.

## Immutable (Phase-21 contract)

- All E19/E20/E24/E25 results and reports (incl.
  `e19_coding_generalization_full_repaired.json`,
  `e20_counterfactual_history_repair.json`, `e24_missing_negative_control.json`,
  `e25_e19_baseline_ceiling_audit.json`) are byte-for-byte unchanged
  (git diff empty).
- `config.yaml`, `memory_optimizer/`, `baselines/`, `server.py` — untouched.

## Honest verdict (reviewer-confirmable)

- The E26 suite is offline-certified (all 6 variants: base fails / gold passes /
  obsolete.patch fails) and the pilot is complete (48/48 cells).
- The pilot stopped the phase: three gates passed, two failed for structural
  reasons (task solvability at the model tier; vanilla recency dropping the
  obsolete facts). No adaptive advantage appeared anyway (6/12).
- `adaptive_beats_vanilla = False`; no winner; no ranking change; PPF artifacts
  byte-identical.

## Next decision (reviewer)

Decide whether to recalibrate E26 families to the llama3.1:8b constraint set
(families that the model can solve from the oracle history but that still make
obsolete-only evidence unsafe), run a different baseline, or stop the
benchmark line.

## Cryptographic checkpoint

- Checkpoint branch: `checkpoint/phase-21-e26-discriminative-benchmark-complete`
- Checkpoint SHA: the Phase-21 closure commit on master.

## Restore

```bash
git checkout checkpoint/phase-21-e26-discriminative-benchmark-complete
```

## Do-not-regress (locked)

1. Do not tune `config.yaml` (dual_score, retention, decay, thresholds, top_k,
   embedding model, task affinity all frozen).
2. Original E19/E20/E24/E25 JSON/report artifacts immutable.
3. Any new coding fixture must pass the offline base-hidden-test gate BEFORE it
   may claim history dependence.
4. History dependence of a coding benchmark requires a reproducible
   counterfactual pair at the claimed model tier (no_history vs direct_history).
5. E19 full-grid results are only eligible under the E20-validated families;
   stochastic no-history draws certify nothing.
6. Benchmark separation requires BOTH contradiction sensitivity AND
   method-specific budget pressure (D42) — one without the other cannot
   discriminate adaptive from vanilla_rag above the ceiling.
7. E26 hard-stop rule stays locked: no 54-cell full grid unless a pilot records
   all five gates passed; no retuning of the pilot gates to pass.