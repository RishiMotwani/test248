# GSD Handoff — Phase 12 of 12 COMPLETE

Generated: 2026-09-17 after committing `research: test retention selectivity under hard memory pressure (E16)`

Note: `gsd_metrics.json`/`gsd_review_files.md` are not present in `.planning/` at
handoff time; the docs kept are ROADMAP.md, STATE.md, REQUIREMENTS.md, config.json,
codebase/. Any reviewer tooling that expected them should treat them as absent (not
deleted) — consistent with the Phase 11 handoff.

## Current commit

- **Commit:** `(this commit)` (pushed to `origin/master`, RishiMotwani/test248)
- Previous: `9ec8f03` (Phase 11 docs handoff); actual Phase 11 research commit `4889f2b`
- Working tree: clean after commit.
- `experiments/results/e16_retention_selectivity.json` (300 grid + 15 no-pressure +
  10 window cells + e15_audit + verdict) and `..._report.md` are gitignored →
  force-added per the repo's "commit raw evidence" convention.

## What changed this phase (D33, E16)

1. **Identity-safe evaluation** (`data/coding_workload.py`): deterministic `fact_id =
   category:qtype:source_turn:SHA256(text)[:16]` (SHA-256, NOT Python `hash()`);
   queries carry `target_fact_ids`/`forbidden_fact_ids`; corrections assert new id
   present / old id absent. New metrics `fact_identity_{context,store,correction}_recall`
   + `fact_identity_obsolete_retention` — immune to token collisions.
2. **Access separation** (`retrieval.py`, `compression.py`): `retrieval_access_count`
   increments only via retriever `_bump`; `ingest_reinforcement_count` only via
   compressor merge/supersession. Legacy `access_count` unchanged (backward compat).
   Feedback loops diagnosable per cell (`diagnostics.spearman_vs_future_use`).
3. **`memory_optimizer/task_state.py`**: `TaskStateTracker(window=32)` FIFO of
   fact-carrying turn embeddings; `task_affinity = mean(top-4 cosine)`, clamped [0,1],
   observed at current-turn ingest (causal, future-blind). Guardrails assert oracle
   future labels never reach policy/decay code.
4. **Eviction policies** (all `retention.mode=dual_score`, retrieval FROZEN):
   `dual_score` (retention_priority), `base_score_only`, `task_affinity` (lexicographic
   `(task_affinity, retention_priority)`), `random` (seeded lower bound),
   `oracle_future_use` (offline future-peeking ceiling).
5. **`pipeline.py`**: `protect_corrections=True` in E16 (a current correction is never
   evicted while its superseded predecessor exists); `protected_capacity_conflict`
   records when the budget is too tight to honour it.
6. **experiments/e16_retention_selectivity.py**: 300-cell grid (5 policies ×
   STORE[256,512,1024,2048] × active[64,128,256] × seeds[42..46], 1200 turns,
   scale 27 = 662 gt facts, genuine pressure) + no-pressure fingerprints (store_budget=0;
   identical across policies — True) + window ablation 16/32/64 + 18-section auto
   report + E15 raw-JSON audit. Flags: `--quick`, `--report-only`, `--force`,
   `--no-window`, `--no-nopressure`.
7. **Tests**: 117 passing total; +37 in `tests/test_e16_retention_selectivity.py`.

## Evidence (E16; details in experiments/results/e16_retention_selectivity_report.md)

- identity_store_recall (mean over active/seeds; stores 256/512/1024/2048):
  dual_score 0.057/0.103/0.129/0.253; task_affinity 0.049/0.093/0.126/0.252;
  base_score_only 0.045/0.088/0.120/0.241; random 0.048/0.095/0.126/0.250;
  oracle_future_use 0.057/0.103/0.129/0.253 (≈ dual at grid store range).
- precision_of_retention = 1.000 for ALL policies; correction recall 1.0;
  fact-level obsolete retention 0; SUPERSEDED_INCORRECTLY = 0 across all 300 cells.
- E15 reconciliation (81 stress cells, from raw JSON): token-based
  obsolete_retention > 0 in 27 cells, correction_recall < 1 in 44 (11 both, 21 clean) —
  token-collision artifact (`memcache` 3, `redis` 8 facts; `100`/`250` only legacy).
  Real residual loss = NEW corrected fact evicted (dual 0/0/0, hard_threshold 18/14/13,
  soft_decay 18/13/9 at 1024/2048/4096) — now guarded by correction protection.
- Window sensitivity (task_affinity, store 512, active 128): w16 rr 0.0636/prec 1.0;
  w64 rr 0.0667/prec 1.0 — insensitive.
- dual_score identity correction recall dips to 0.867 at store 1024 (vs 1.0 at
  256/512) — verified real marginal-eviction behavior near natural store size
  (~1043 tokens), NOT a metric or code bug.

## Decision (auto-classified, reviewer-confirmable)

8-criteria gate, task_affinity vs dual_score:
1. `task_affinity` future-use retention recall (low store 256/512): 0.0709 vs
   dual 0.0802 → **FAIL** (c1)
2. Same at tightest low store 256: 0.060 vs 0.070; precision both 1.0 → **FAIL** (c6:
   0/5 seed wins)
3. vs random (0.0709 vs 0.0706) → **FAIL** (c7); oracle store gap task +0.009 vs
   dual −0.0003 → **FAIL** (c8)
4. PASS: precision parity (c2), correction recall 1.0 (c3), obsolete 0 (c4), active
   context unchanged (c5) — but hard-fail criteria dominate.

**Verdict: TASK_AFFINITY_IS_SUPPORTED → NOT SUPPORTED (0/8). RETENTION STAYS
`dual_score`. Config UNCHANGED.** No silent flip; `task_affinity` remains opt-in for
workloads with concentrated future use. Retrieval stays frozen (sim 0.85 + imp 0.15 +
cat_bonus 0.02, top_k 5, sim_threshold 0.35).

## Verification status

- `python -m pytest tests/` → **117 passed** (venv python at
  `/home/goku/prototype/prototype3/venv/bin/python`; system `/usr/bin/python` has no pytest).
- Full grid (300 cells) + no-pressure + window completed in one foreground run;
  JSON + report written to `experiments/results/`; report regenerated with 0 "N/A"
  via `--report-only`.
- `/tmp` was 100% full — all logs/artifacts under the repo (`experiments/results/`).

## Next unresolved question (do NOT start a new phase)

- Whether ANY future-blind survival signal can beat retrieval-feedback survival in
  this workload remains open — `task_affinity` is not it. At low store budgets
  oracle ≈ causal policies, so the grid's store range cannot discriminate policies;
  a workload with more concentrated future use (fewer distinct target facts, denser
  repeats) may separate them. Documented under UNRESOLVED in D33. This is a design
  question, not a bug.

External reviewers: read experiments/results/e16_retention_selectivity_report.md and
the D33 brain.md entry for the complete OBSERVED/INFERRED/UNRESOLVED breakdown.