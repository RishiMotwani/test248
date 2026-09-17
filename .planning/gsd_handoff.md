# GSD Handoff — Phase 11 of 11 COMPLETE

Generated: 2026-09-17 after committing `research: separate memory activation from retention (E15)`

Note: `gsd_metrics.json`/`gsd_review_files.md` are not present in `.planning/` at
handoff time; the docs kept are ROADMAP.md, STATE.md, REQUIREMENTS.md, config.json,
codebase/. Any reviewer tooling that expected them should treat them as absent (not
deleted) — nothing this phase removed them.

## Current commit

- **Commit:** `4889f2b` (pushed to `origin/master`, RishiMotwani/test248)
- Previous: `7a026b9` (Phase 10 — E14 retention diagnosis)
- Working tree: clean after commit.
- `experiments/results/e15_retention_policy.json` is ~85 MB (81 stress cells with
  per-fact activation+retention transcripts). Pushed with a GitHub GH001 large-file
  warning (best-effort, matches the repo's "commit raw evidence" convention).

## What changed this phase (D32, E15)

1. **Config default advanced** to `retention: {mode: dual_score, eviction_priority:
   retention_priority}` in `config.yaml`. Legacy mode (`hard_threshold` /
   `current_importance`) is preserved and remains the pipeline's fallback when a
   settings dict omits the `retention` key (backward compatible).
2. **decay.py**: `calculate_retention_priority()` (stable, no temporal decay) +
   `step_decay_and_prune(memories, turn, retention_mode)` — hard_threshold (legacy
   prune), soft_decay/dual_score (never threshold-prune, flag `retained_below_threshold`),
   ValueError on unknown mode (validated before the update loop).
3. **budget.py**: `token_budget_evict(..., priority_key)` — dual_score uses
   `retention_priority`; tie-break unchanged (oldest source_turn first).
4. **pipeline.py**: reads `retention` cfg, threads mode + eviction priority into
   decay/eviction; store-pressure instrumentation intact.
5. **experiments/e15_retention_policy.py**: primary grid (4 policies × 4 budgets ×
   5 seeds) + mandatory stress grid (scale 27, ~8272 natural store tokens, 3 store
   budgets × 3 active budgets × 3 seeds, 1200 turns) + report + auto-classifier.
   `--quick`, `--no-stress`, `--stress-only`, `--report-only` modes.
6. **Tests**: 80 passing total; +26 in `tests/test_e15_retention_policy.py`.
   E14 test helper now pins legacy `hard_threshold` explicitly (its lifecycle tests
   target the legacy semantics; the new default is dual_score).

## Evidence (E15; details in experiments/results/e15_retention_policy_report.md)

- hard_threshold (legacy): ctx_recall 0.576/0.705/0.800/0.917, store 0.672/0.774/
  0.852/0.917 at 64/128/256/512; decay removals 31.4/20.4/10.6/0.2; corr_recall
  0.50/0.70/1.0/1.0.
- soft_decay: ctx 0.862/0.883/0.888/0.917, store 0.917 flat, decay removals 0,
  obsolete 0, corr_recall 1.0 at ALL budgets; revived facts 30.0/20.4/9.6/0.2.
- dual_score: identical to soft_decay in primary (no store evictions); under the
  tightest stress store (1024) ctx 0.368/0.434/0.503 vs soft 0.322/0.405/0.484,
  corr_recall 0.94 vs 0.28, fewer evictions (430 vs 518), archival store_recall
  −1 pt tradeoff.
- Stress store 2048/4096: dual store 0.805/0.897 vs soft 0.775/0.877; hard loses up
  to ~409 facts to decay-pruning before eviction binds.
- obsolete_retention = 0 under EVERY policy/budget. Supersession always correct.

## Decision (auto-classified, reviewer-confirmable)

1. SEPARATING ACTIVATION FROM SURVIVAL IS SUPPORTED → **SUPPORTED** (store +0.113,
   ctx +0.138; active context unchanged; safety preserved; store growth bounded).
2. RETENTION PRIORITY FOR STORE EVICTION IS SUPPORTED → **SUPPORTED** (tightest
   store: ctx +0.031, corr +0.667; archival store_recall −0.009 tradeoff).
3. SOFT RETENTION ... CREATES A STALENESS TRADEOFF → **NOT SUPPORTED** (obsolete 0,
   correction recall improved).

Candidate advanced to default per rule: `dual_score` + `retention_priority`.

## Verification status

- `python -m pytest tests/` → **80 passed** (venv python at
  `/home/goku/prototype/prototype3/venv/bin/python`; system `/usr/bin/python` has no pytest).
- E15 quick smoke, full primary, and full stress grid all completed and written to
  `experiments/results/e15_retention_policy.json` + `report.md`.
- Only pre-run bugs were in the NEW experiment/test code (age-bucket tail overflow
  in stress, dead `nested` block, revival-test empty-store handling, classifier
  tolerances) — all fixed; production-path code validated by the 80-passing suite.

## Next unresolved question (do NOT start a new phase)

- The archival-vs-task tradeoff at the tightest store budget: retention-priority
  eviction drops never-retrieved facts first (store_recall −1 pt at store=1024).
  Is that acceptable for long-horizon reuse beyond the 1200-turn grid? This is a
  design question, not a bug; it is documented under UNRESOLVED in D32.

External reviewers: read experiments/results/e15_retention_policy_report.md and the
D32 brain.md entry for the complete OBSERVED/INFERRED/UNRESOLVED breakdown.