# GSD Handoff — Phase 15 of 15 (E19 coding-generalization pilot, HONEST)

Generated: after committing the Phase-15/E19 honest-pilot grid.

Note: `gsd_handoff.md` was rewritten (not appended) this phase per §37 to keep
the doc a single reviewer-readable artifact. `gsd_metrics.json`/`gsd_review_files.md`
are not present in `.planning/` at handoff time; the docs kept are ROADMAP.md,
STATE.md, PROJECT.md, config.json, codebase.

## Where we are

Phase 15 (E19) complete — **pilot-only, honest verdict**. The D36/D37/D39
adaptive-memory system generalizes the *history dependence* gates across a
broader set of genuinely history-dependent long-horizon coding tasks, but the
broader-set *advantage over baselines* is NOT established at the llama3.1:8b
tier, so the full 225-cell grid was NOT run and no production change was made.

## Locked contract (from the E19 prompt)

- **Config:** `config.yaml` UNTOUCHED (production defaults immutable).
- **No tuning:** embedding fix upstream (E18) is a correctness repair only; no
  dual_score / retention / decay / retrieval-weight / threshold changes in E19.
- **One adaptive method:** `adaptive` = production `dual_score` +
  `protect_corrections`. Do not add `adaptive_v2`/`adaptive_tuned`/`adaptive_plus`.
- **Primary set:** `user_ids`, `validation_pure`, `transaction_atomicity`
  (history-gated). `config_contract` is fixture-ONLY (calibration probe
  0/3–1/3 oracle direct_history across four fixture designs at llama3.1:8b —
  task nuance beyond this tier; excluded from the primary set and from gates).
- **Negative controls:** `cache_readonly_nh`, `write_retry_neg`. Diagnostic and
  diagnostic-only; never counted toward the adaptive-advances claim.
- **Pilot rule (locked, honest):** verdict = adaptive_advances ONLY IF every
  gate passes AND adaptive shows a positive paired advantage over each baseline
  with a paired 95% CI lower bound > 0. Do not call it from a point estimate.

## Late/surprise findings in E19 (measured, not inferred)

1. **Task-suite calibration (multiple restate/delete designs this phase):** the
   first fixture drafts leaked history (tasks solvable by reading the file tree
   alone). Final suite requires per-primary `gold.patch` verified against
   `hidden/test_hidden.py` tests with `no_history` oracle-probes, and a
   workspace calibration step before any grid cell runs.
2. **Gate C (history-dependence) failed honestly on a 3-draw diagnostic**
   (`DIAG_DRAWS=3`, llama3.1:8b): only `validation_pure` is cleanly gated
   (no_history 0/3); `user_ids` 2/3 and `transaction_atomicity` 1/3 no-history
   successes — both partially workspace-solvable at this tier. `rg_error` was
   dropped/replaced with `transaction_atomicity` after probing (see report
   section 12).
3. **`adaptivity_advances = False`:** the adaptive-vs-baseline paired 95% CI
   lower bounds include 0 for every arm on the pilot grid (and the paired
   per-task advantage is flat across all five methods on `user_ids` 19/20).
   Verdict rule honest (gate B + gate C history-dependence work; the broader-set
   advantage is NOT established at this model tier). Full grid NOT run.
4. Self-consistency repair during the pilot run (file-edit block application)
   is required for the LLM path; diagnostics draw 3 times (DIAG_DRAWS=3) to
   avoid single-draw variance masking a genuinely gated task.

## Honest verdict (pilot; reviewer-confirmable)

- `adaptive_advances`: **False** (point estimate alone never advances).
- `gates_all_passed`: **False** — gate C (history dependence) failed.
- `config.yaml` UNTOUCHED; E17/E18 artifacts immutable; suite fixtures
  calibratable; report is pilot-only and reviewer-confirmable.
- Verdict and full 24-section report:
  `experiments/results/e19_coding_generalization.json` +
  `experiments/results/e19_coding_generalization_report.md` (force-added).

## Cryptographic checkpoint

- Checkpoint branch: `checkpoint/phase-14-e18-complete` → `07259e0`
- Checkpoint tag: `checkpoint-phase-14-e18-07259e0` = same SHA
- HEAD = `07259e0` (branch `master`), matches checkpoint (branch created AFTER
  E18; tag immutable)

## Restore

```bash
git checkout checkpoint/phase-14-e18-complete
```

or

```bash
git reset --hard checkpoint/phase-14-e18-complete
```

## Do-not-regress (locked)

1. Do not tune `config.yaml` (dual_score, retention, decay, thresholds, top_k,
   embedding model, task affinity all frozen).
2. Do not add selective-retention/summarization improvements to pretend the
   broader-set advantage is established.
3. Do not rerun the full 225-cell grid until reviewer confirms the honest
   pilot + the per-task history-dependence optimization.
4. E17/E18 JSON + report immutable; E19 JSON/report are pilot-only.

## Next phase (if reviewer approves the pilot)

Rerun the full 225-cell grid ONLY after: (a) reviewer confirms the pilot
verdict is honest, and (b) a fix to the history-dependence ceiling at this
tier is proposed and gated. No full grid in Phase 15.
