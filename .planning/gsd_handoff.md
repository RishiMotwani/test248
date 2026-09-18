# GSD Handoff — Phase 16 (E20 counterfactual-history benchmark calibration, HONEST INVALID)

Generated: after committing the Phase-16 / E20 calibration grid.

This doc is the single reviewer-readable artifact for the phase. Kept docs:
ROADMAP.md, STATE.md, PROJECT.md, config.json, codebase.

## Where we are

Phase 16 (E20) complete — **honest INVALID verdict at llama3.1:8b**. The E20
counterfactual-history calibration answers E19's open gate-C question directly:
two of three counterfactual groups ARE genuinely history dependent
(retry_policy dh 6/6 vs nh 3/6; serialization_policy dh 5/6 vs nh 0/6), but
`identifier_policy` is fully solvable from the workspace alone (dh 6/6 AND
nh 6/6, separation 0). Because the predeclared Hard Gate 2 requires ALL three
groups to pass, `history_dependence_benchmark = INVALID` and
`e19_full_grid_eligible = FALSE`. **The E19 full grid must NOT be run until
identifier_policy-family fixtures (including E19's `user_ids`) are redesigned.**

## Locked contract (from the E20 prompt)

- **Config:** `config.yaml` UNTOUCHED; `memory_optimizer/` and `server.py`
  behavior UNTOUCHED. E20 is benchmark calibration, not an adaptive-memory run.
- **No tuning:** no dual_score / retention / decay / retrieval-weight / threshold
  / embedding-model changes.
- **Methods (3):** `no_history` (identical prompt for A/B), `direct_history`
  (oracle: the same variant's non-obsolete gold facts only), `full_context`
  (raw 600-turn history; always overflows the 1024-word budget → recorded as
  CONTEXT_OVERFLOW diagnostic).
- **Grid:** 3 groups × 2 variants × 3 seeds × 3 methods = 54 cells @ 1024 words,
  llama3.1:8b @ localhost:11434, temperature 0.1.
- **Hard Gate 1:** all 6 gold patches must apply cleanly and pass hidden tests
  (offline) BEFORE any model run → STOP if failed.
- **Hard Gate 2 (per group, /6):** direct_history ≥ 5, no_history ≤ 3,
  separation ≥ 2; ALL three groups must pass → else INVALID.
- **Pair-integrity gates:** workspace+prompt byte-identical per pair; history,
  final fact ids, hidden tests and gold patches differ per variant; no
  leakage.

## Findings (measured, not inferred)

1. **Hard Gate 1 PASS:** all 6 variants' `gold.patch` files apply cleanly and
   pass their hidden tests (offline `StaticCoder` path). Fixture quality is good.
2. **Pair-integrity PASS** — workspace/prompt identical within each pair;
   histories, final fact IDs, hidden tests and gold patches all differ;
   zero prompt/history leakage across all 54 records.
3. **Hard Gate 2 FAILS on `identifier_policy`** — no_history 6/6, direct_history
   6/6, separation 0. The model guesses `str(value)`/`int(value)` for the
   canonical-user-id contract without any history at llama3.1:8b. This is the
   SAME failure class as E19 gate C's `user_ids` (2/3 no-history successes),
   providing direct causal evidence for the E19 finding.
4. **retry_policy PASS** — dh 6/6 vs nh 3/6 (sep 3). The "no auto-retry" arm is
   default-guess solvable; the "retry exactly once" arm is not (nh 0/3 for B).
5. **serialization_policy PASS** — dh 5/6 vs nh 0/6 (sep 5). Cleanest pair: no
   history solves neither preserve-unknown-keys nor drop-unknown-keys.
6. **full_context overflow confirmed:** 18/18 diagnostic cells overflow the
   1024-word budget (raw ~7.2k words) — consistent with E17/E19; recorded, not
   used for any gate.

## Honest verdict (reviewer-confirmable)

- `history_dependence_benchmark`: **INVALID**
- `e19_full_grid_eligible`: **FALSE**
- `config.yaml` UNTOUCHED; E17/E18/E19 artifacts immutable (E19 report had a
  presentation-only heading/column fix: sections 14/15/18 now derive from
  cfg mode/budgets — data unchanged).
- Verdict + 7-section report: `experiments/results/e20_counterfactual_history.json` +
  `experiments/results/e20_counterfactual_history_report.md` (force-added).
- Tests: 174 passing (+31 in `tests/test_counterfactual_history.py`).

## What must change for E19 full-grid eligibility (next phase)

Redesign `identifier_policy`-family fixtures so the required behavior is NOT
derivable from the workspace / function signature alone (e.g., make both
variants require an external contract fact — the NUMBER or TYPE of id can no
longer be inferred from the empty-stub repo), then re-run E20's Hard Gate 2
on that group. Only when all three groups pass is the E19 full grid eligible.

## Cryptographic checkpoint

- Checkpoint branch: `checkpoint/phase-16-e20-complete` → `d331243`
- Checkpoint tag: `checkpoint-phase-16-e20-d331243` = same SHA

## Restore

```bash
git checkout checkpoint/phase-16-e20-complete
```

or

```bash
git reset --hard checkpoint/phase-16-e20-complete
```

## Do-not-regress (locked)

1. Do not tune `config.yaml` (dual_score, retention, decay, thresholds, top_k,
   embedding model, task affinity all frozen).
2. Do not run the E19 full 225-cell grid — E20 declares it NOT eligible until
   identifier_policy-family fixtures are redesigned and re-gated.
3. E17/E18/E19 JSON + reports immutable (E19 report presentation-only fix
   applied).
4. Do not claim history dependence for any coding task without a counterfactual
   no_history-vs-direct_history gate.

## Next phase

Redesign the identifier_policy counterfactual (require history to choose the
id contract), re-run E20 Hard Gate 2, and only then revisit E19.