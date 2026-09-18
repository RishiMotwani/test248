# GSD Handoff — Phase 17 (E20 counterfactual fixture repair + revalidation, VALID)

Generated: after committing the Phase-17 / E20 repair revalidation grid.

This doc is the single reviewer-readable artifact for the phase. Kept docs:
ROADMAP.md, STATE.md, PROJECT.md, config.json, codebase.

## Where we are

Phase 16 declared E20 INVALID because the `identifier_policy` counterfactual
was solvable from the workspace alone at llama3.1:8b (no_history 6/6,
separation 0) — the fixture family, not the measurement, was wrong. **Phase 17
replaced `identifier_policy` with `routing_policy`** (opaque operation codes →
lane assignment; the op→lane mapping exists ONLY in variant hidden tests, gold
patches, and 600-turn histories — it is underivable from the workspace, which
contains no mapping table), added a strict **offline base-hidden-test gate**
(unmodified workspace must FAIL every variant's hidden tests; gold patch must
apply and pass), and re-ran the full 54-cell E20 calibration as a repair run.

**All three counterfactual groups now pass Hard Gate 2:**

| group | direct_history | no_history | separation | gate |
| --- | --- | --- | --- | --- |
| routing_policy | 6/6 | 0/6 | 6 | PASS |
| retry_policy | 5/6 | 3/6 | 5 | PASS |
| serialization_policy | 6/6 | 1/6 | 6 | PASS |

`history_dependence_benchmark = VALID`, `e19_full_grid_eligible = TRUE`.

## Locked contract (from the Phase-17 prompt)

- **Config:** `config.yaml` UNTOUCHED; `memory_optimizer/` and `server.py`
  behavior UNTOUCHED. Repair = fixture + offline-gate + re-run, not tuning.
- **Original E20 artifacts immutable:** `e20_counterfactual_history.json` +
  `_report.md` byte-for-byte unchanged (sha256 re-verified:
  `1aeb771c…` / `46edec2a…`). New artifacts only:
  `e20_counterfactual_history_repair.json` (`b0e1894a…`) +
  `_repair_report.md` (`0a3ba3aa…`).
- **Methods (3):** `no_history` (identical prompt for A/B), `direct_history`
  (oracle: same variant's non-obsolete gold facts), `full_context` (raw 600-turn
  history → always overflows 1024-word budget → CONTEXT_OVERFLOW diagnostic).
- **Grid:** 54 cells (3 × 2 × 3 × 3) @ 1024 words, llama3.1:8b @ localhost:11434,
  temperature 0.1; run via `python -m experiments.e21_counterfactual_history_repair --run --force`.
- **Hard Gate 1 (offline):** all 6 variants: base workspace FAILS hidden tests,
  gold patch applies cleanly and passes → STOP if any fails. No LLM used.
- **Hard Gate 2 (per group, /6):** dh ≥ 5, nh ≤ 3, sep ≥ 2; ALL three groups
  must pass → verdict VALID/INVALID. No averaging, no threshold override.
- **Final group list:** exactly `["routing_policy","retry_policy","serialization_policy"]`.
- **E19 full grid NOT auto-run** — eligibility is now TRUE but running it is the
  reviewer's decision.

## Findings (measured, not inferred)

1. **routing_policy is genuinely history dependent:** direct_history 6/6 vs
   no_history 0/6 (separation 6). With no history the model cannot guess the
   op→lane allocation (ops are opaque labels; no mapping in the workspace,
   docstring, constants, or filenames) — 0/6 solves it. With the variant's
   history it solves 6/6. This is exactly the fix Phase 16 required.
2. **retry_policy revalidated PASS:** dh 5/6, nh 3/6, sep 5 (Phase 16: sep 3).
3. **serialization_policy revalidated PASS:** dh 6/6, nh 1/6, sep 6 (Phase 16:
   sep 5).
4. **Offline base-hidden-test gate gives every gold check a `base_hidden_test_pass`
   field** — False for all 6 variants, with gold patch applies + hidden tests
   pass. Gate 1 therefore proves the fixture is not solvable without history
   before any model runs.
5. **Pair-integrity gates now compare real hashes** (`workspace_sha`/`prompt_sha`
   per variant, A==B) instead of hardcoded True; routing workspace contains all
   4 ops and both lanes but none of the 8 op-to-lane literal pairs.
6. **Originals preserved:** pre- and post-run sha256 of the Phase-16 JSON and
   report are identical; `git diff` on those two files is empty.
7. **No adaptive-memory change:** only `data/counterfactual_task_suite.py`,
   `experiments/e20_counterfactual_history.py`, new
   `experiments/e21_counterfactual_history_repair.py`, `tests/…`, fixtures,
   and `.planning` docs changed.

## Honest verdict (reviewer-confirmable)

- `history_dependence_benchmark`: **VALID**
- `e19_full_grid_eligible`: **TRUE** (running it = reviewer's decision)
- `config.yaml` UNTOUCHED; E17/E18/E19 + original E20 artifacts immutable.
- Repair JSON + 141-section report: `experiments/results/e20_counterfactual_history_repair.json` +
  `_repair_report.md` (force-added).
- Tests: 177 passing (+3 in `tests/test_counterfactual_history.py`).

## Next decision (reviewer)

The E20 benchmark is now fully calibrated; all groups discriminate. Whether to
run the E19 full 225-cell grid (unconstrained adaptive vs baselines across the
4-task fixture set, now understood as genuinely history dependent) is the
reviewer's call. It is not run automatically and nothing in this phase
overrides that.

## Cryptographic checkpoint

- Checkpoint branch: `checkpoint/phase-17-e20-repair-complete` → `ba7238c`
- Checkpoint tag: `checkpoint-phase-17-e20-repair-ba7238c` = same SHA

## Restore

```bash
git checkout checkpoint/phase-17-e20-repair-complete
```

or

```bash
git reset --hard checkpoint/phase-17-e20-repair-complete
```

## Do-not-regress (locked)

1. Do not tune `config.yaml` (dual_score, retention, decay, thresholds, top_k,
   embedding model, task affinity all frozen).
2. Original E20 JSON/report and all E17/E18/E19 artifacts immutable.
3. Any new coding fixture must pass the offline base-hidden-test gate BEFORE it
   may claim history dependence: unmodified workspace fails, gold patch passes.
4. Requires a reproducible counterfactual pair at the claimed model tier to
   declare a coding benchmark history dependent (no_history vs direct_history).
5. E19 full grid must not be auto-run by a phase; only a reviewer decision may
   trigger it.