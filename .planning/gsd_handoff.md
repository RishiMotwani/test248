# GSD Handoff — Phase 14 of 14 COMPLETE

Generated: 2026-09-18 after committing `research: make correction supersession retrieval-safe`

Note: `gsd_metrics.json`/`gsd_review_files.md` are not present in `.planning/` at
handoff time; the docs kept are ROADMAP.md, STATE.md, REQUIREMENTS.md, config.json,
codebase/. Reviewer tooling should treat them as absent (not deleted), consistent
with prior handoffs.

## Current commit

- **Commit:** `(this commit)` (pushed to `origin/master`, RishiMotwani/test248)
- Previous: `4a9e032` (Phase 13 docs); research commit `0025589` (E17)
- Working tree: clean after commit.
- `experiments/results/e18_correction_safety.json` (36-cell offline identity gate +
  72-run targeted coding grid + gates + verdict) and `_report.md` (19 sections)
  are gitignored → force-added per the repo's "commit raw evidence" convention.
  Work directory `experiments/results/e18_work/` stays gitignored.
- E17 artifacts (`e17_coding_capability.json` + `_report.md`) are IMMUTABLE — not
  regenerated this phase.

## What changed this phase (D35, E18)

1. **Root-cause fix** (`memory_optimizer/compression.py`): E17's failing cells
   stored the obsolete fact AND the correction as separate memories because
   `dedupe_incremental` ran `_is_supersession` only *inside* `if sim >= 0.90` — a
   correction restating the old fact with new wording (cosine ~0.7) was stored as
   an additional memory and the obsolete fact was never replaced. Fix: factored
   `_replace_with_supersession(ex, mem, mem_emb)` and invoke it for
   `supersedes_turn` targets AND before any similarity computation in the generic
   same-category loop. `_is_supersession()` unchanged (still requires a
   revision/negation marker + full restatement — no weakening); duplicate-merge
   kept its `ex["confidence"] = max(...)` gain line so existing correction-safety
   tests pass. Docstring updated to state resolution ordering.
2. **Regression tests** (134 → 138):
   - `tests/test_compression_supersession.py` +2: explicit supersession precedes
     the embedding threshold (cosine 0.7 < 0.90 still supersedes); correction with
     new semantic wording replaces the old fact via the lexical path.
   - `tests/test_retrieval_ranking.py` +1 (e2e): after dedupe, the compressed
     correction is the *only* authoritative memory retrievable — the obsolete fact
     is gone by identity.
   - `tests/test_e17_coding_capability.py` +1: asserts the rewritten cache_readonly
     hidden test strictly enforces the coordinator (contains `CacheCoordinator`,
     `monkeypatch.setattr`, `calls ==`, gold uses CacheCoordinator).
3. **Benchmark fixture hardened** (`data/coding_tasks/cache_readonly/hidden/test_hidden.py`):
   rewritten as a *delegating spy* — `monkeypatch.setattr(CacheCoordinator, "set",
   wrapper)` records the call AND calls the original set, asserts
   `calls == [("beta","2")]` and `store.get("beta")=="2"`. A naive `store.put`
   fails; the gold patch still passes. Resolves known-issue #15 from Phase 13.
4. **E18 driver** (`experiments/e18_correction_safety.py`): (a) 36-cell offline
   identity gate — 4 tasks × 3 seeds × 3 budgets, asserts on store *identity*
   (current fact sole authority, obsolete fact absent by id) with lexical
   embeddings; (b) 72-run targeted coding grid — `user_ids` + `validation_pure`
   × 3 seeds × 3 budgets × 4 methods (adaptive / vanilla_rag / llm_summarization
   / raw_clipped), llama3.1:8b, reusing `experiments/coding_benchmark.py`
   unchanged; (c) verdict rule + 19-section report. All artifacts under
   `experiments/results/` (no /tmp).

## Evidence (E18; details in experiments/results/e18_correction_safety_report.md)

- **Offline identity gate: 36 cells run, 27 correction-bearing — ALL PASS.**
  `correction_recall = 1.0`, `obsolete_exposure = 0.0` in every correction-bearing
  cell. The current fact is the only authority in the store; the obsolete fact is
  absent by identity (not merely deprioritized).
- **Targeted coding grid (72 runs, llama3.1:8b):**
  - adaptive: **17/18 (94.4%)**, correction_recall 1.0, obsolete_exposure 0.0
  - vanilla_rag: 9/18 (50%), correction_recall 0.0, obsolete_exposure 1.0 — all
    failures OBSOLETE_INFORMATION_USED (obsolete fact injected, correction missed)
  - llm_summarization: 9/18 (50%), correction_recall 0.17–0.67, MEMORY_MISS dominant
  - raw_clipped: 8/18 (44.4%), correction_recall 0.0, RETRIEVAL_MISS dominant
  - Paired adaptive diffs: +0.4444 vs vanilla_rag, +0.5 vs raw_clipped.
- **Adaptive's single failure** (validation_pure, seed 2, budget 1024): CODING_ERROR — the
  correction WAS in the injected context; the model applied a wrong edit. Model
  failure, not memory failure.
- **Same-model summaries are NOT a reliable correction carrier** (llm_summarization
  correction_recall 0.17–0.67): a correction can be carried textually, but its
  propagation across summary updates is lossy.

## Decision (auto-classified, reviewer-confirmable)

- EXPLICIT CORRECTION SUPERSESSION IS RETRIEVAL-SAFE AFTER THE D35 FIX →
  **SUPPORTED**. Offline identity gate all-PASS (correction_recall 1.0,
  obsolete_exposure 0.0, 27/27 correction-bearing cells) + coding grid
  (correction_recall 1.0, obsolete_exposure 0.0, adaptive 17/18).
- The E17 bottleneck was retrieval *consolidation*, not retriever ranking: once the
  obsolete memory is actually gone, retrieval of the current fact is automatic.
- Production defaults unchanged. `config.yaml` untouched. The change is ordering of
  the supersession check only — no weight, policy, or model change.

## Verification status

- `./venv/bin/python -m pytest tests/` → **138 passed** (system `/usr/bin/python`
  has no pytest).
- E18 gate + 72-run grid + report completed with artifacts under the repo
  (`experiments/results/`), per the /tmp-full convention.

## Next unresolved questions (do NOT start a new phase)

- **Summary-based everyday correction propagation is unreliable** (llm_summarization
  correction_recall 0.17–0.67, MEMORY_MISS dominant). Where an agent relies on a
  running text summary rather than an authoritative store, corrections are
  frequently lost across summary updates. A follow-up decision could gate
  summary-only retention on correction-safety guarantees.
- **The assignment-required adaptive-vs-baseline comparison is done at the penalty
  of design breadth**: Phase 14 reorders when supersession runs; it does NOT add
  selective-retention or summarization improvements, which remain open (E16
  task_affinity rejected; E17 fixed-budget answer still stands).
- **Model capability is the binding constraint on the remaining failures** even
  with a correct memory: adaptive's single E18 failure had the right fact in
  context and the model still produced a wrong edit.

External reviewers: read experiments/results/e18_correction_safety_report.md and
the D35 brain.md entry for the complete OBSERVED/INFERRED/UNRESOLVED breakdown.