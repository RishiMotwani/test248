# GSD Handoff — Phase 13 of 13 COMPLETE

Generated: 2026-09-18 after committing `research: evaluate adaptive memory on long-horizon coding tasks (E17)`

Note: `gsd_metrics.json`/`gsd_review_files.md` are not present in `.planning/` at
handoff time; the docs kept are ROADMAP.md, STATE.md, REQUIREMENTS.md, config.json,
codebase/. Reviewer tooling should treat them as absent (not deleted), consistent
with prior handoffs.

## Current commit

- **Commit:** `(this commit)` (pushed to `origin/master`, RishiMotwani/test248)
- Previous: `6b25a80` (Phase 12 E16 raw results + report docs); research commit `c971553`
- Working tree: clean after commit.
- `experiments/results/e17_coding_capability.json` (180-run full grid + 12
  diagnostics + gold checks + gates + paired CIs) and `_report.md` (24 sections)
  are gitignored → force-added per the repo's "commit raw evidence" convention.
  Work directory `experiments/results/e17_work/` stays gitignored.

## What changed this phase (D34, E17)

1. **Task suite** (`data/coding_task_suite.py` + `data/coding_tasks/*`): 4 real
   packages (`cache_readonly`, `user_ids`, `write_retry`, `validation_pure`); each
   has a ~600-turn transcript that buries critical facts (incl. >300-turn-old facts,
   corrections, negative constraints) among distractors, a workspace the model
   edits, and a deterministic hidden test the model never sees. Gold patches
   verified (base fails / gold passes). Ingestion is oracle-pre-extracted (E17
   measures retention + retrieval + context allocation, not extraction).
2. **Shared harness** (`experiments/coding_benchmark.py`): one code path for every
   arm — method `prepare(history, budget) → retrieve(prepared, task, budget)` → the
   harness builds the prompt, calls the model, applies the edit, runs hidden tests.
   Same model, temperature 0.1, retry (initial + repair), shared word-count
   tokenizer (`memory_optimizer/tokenizer.py`). Failure taxonomy is deterministic;
   recall diagnostics exclude obsolete facts from denominators.
3. **Harness fix (required, not an experiment result):** llama3.1:8b emits
   malformed multi-hunk unified diffs (missing `@@`) that `git apply` applied
   *partially and silently* — even the oracle failed. Primary edit format is now
   complete-file blocks (`### FILE:`/`### END FILE`, tolerant of missing end marker
   and code fences) applied verbatim; a diff fallback is retained. The applied
   change is captured as a git diff for provenance/determinism.
4. **Methods** (the history mechanism is the only difference): `raw_clipped`,
   `sliding_window`, `llm_summarization` (running summary from the same model every
   50 turns), `vanilla_rag` (static cosine store, no decay/importance/eviction),
   `adaptive` (production `dual_score` + `protect_corrections=True`, e.g.
   `injection_token_limit=budget`, `memory_store_token_budget=max(4096, budget*4)`).
   Diagnostics: `no_history`, `full_context` (unconstrained upper bound),
   `direct_history` (oracle gold facts).
5. **Driver** (`experiments/e17_coding_capability.py`): pilot (3×2×2×5=60) then
   full grid (4×3×3×5=180), 9 gates, paired bootstrap CIs
   (`statistics.bootstrap_ci_mean_diff`), verdict rule, 24-section report. **Runs
   are resumable**: JSON is rewritten incrementally after every run and completed
   `task:seed:budget:method` keys are skipped on restart; `--report-only`
   reclassifies without reruns; `--limit`/`--force`/`--no-embed`/`--skip-model-check`
   flags.
6. **Tests**: +17 in `tests/test_e17_coding_capability.py` (offline; fake coder +
   lexical embeddings): task-suite determinism, gold-patch harness correctness on
   all 4 tasks, per-method budget enforcement, no-history vs oracle diagnostics,
   leakage logic-vs-imports, file-edit parse/apply + path-traversal guard, failure
   precedence, aggregation/verdict shape, 24-section report, artifact paths under repo.

## Evidence (E17 full grid; details in experiments/results/e17_coding_capability_report.md)

- **All 9 gates pass** (pilot and full): budget pressure (full raw history ~7.2k
  tokens > 1k budget; `full_context` always overflows the 8k window once the
  workspace is attached), workspace independence, methods differ (4+ distinct
  contexts), real summarization (144 model update calls), adaptive uses production
  pipeline, zero leakage, history dependence (user_ids + validation_pure:
  no-history fails, oracle succeeds), test determinism, edit determinism.
- **Overall success** (final, ≤2 attempts): adaptive 0.75, llm_summarization 0.75,
  raw_clipped/sliding_window/vanilla_rag 0.722. No separation between arms.
- **Paired adaptive vs X** (36 pairs each): raw +0.028 (95% CI 0.000–0.083),
  sliding +0.028 (CI 0.000–0.083), llm +0.000 (CI 0–0), vanilla +0.028 (CI
  0.000–0.083). Every CI lower bound = 0.
- **First-pass success**: adaptive 0.53 (worst) vs 0.58–0.69 others.
- **Failing-cell mechanism**: `correction_recall=0` + `obsolete_fact_exposure=1.0`
  — adaptive and vanilla_rag inject the same obsolete fact and miss the correction
  (identical `context_sha`), i.e. retrieval *contamination*, not store size.
- **History dependence is partial**: cache_readonly and write_retry are solvable
  from the workspace alone (no-history succeeds); the archive of this project's
  own suite limitation: cache_readonly's hidden test only asserts `put_count==1`,
  which a naive `store.put` also satisfies (CacheCoordinator constraint not strictly
  enforced).

## Decision (auto-classified, reviewer-confirmable)

- ADAPTIVE MEMORY IMPROVES LONG-HORIZON CODING AT FIXED BUDGET → **NOT SUPPORTED**.
  `adaptive_advances=False`: no paired 95% CI excludes zero; all arms cluster at
  0.72–0.75. The grid is valid (all gates pass), so this negative is a finding.
- No production change. `config.yaml` untouched; no `memory_optimizer` behavior
  modified by E17 (harness + baselines + data only). Retrieval stays frozen
  (sim 0.85 + imp 0.15 + cat_bonus 0.02, top_k 5, sim_threshold 0.35).

## Verification status

- `python -m pytest tests/` → **134 passed** (venv python at
  `/home/goku/prototype/prototype3/venv/bin/python`; system `/usr/bin/python` has no pytest).
- Full grid (180 runs) + diagnostics completed in one foreground run (user-aborted
  run resumed cleanly via the incremental JSON — this is the documented resume
  path). JSON + 24-section report written to `experiments/results/`.
- `/tmp` was 100% full — all logs/artifacts under the repo (`experiments/results/`).

## Next unresolved questions (do NOT start a new phase)

- **Retrieval contamination dominates failing cells.** In every E17 failure the
  injected context contained the obsolete fact and missed the correction
  (`correction_recall=0`, `obsolete_fact_exposure=1.0`) for BOTH adaptive and
  vanilla_rag — identical contexts. The corrected fact exists in the store but is
  outranked by its obsolete predecessor. Priorities/fusion of correction-aware
  retrieval is a possible follow-up decision.
- **History dependence is only 2/4 tasks** in this suite — cache_readonly and
  write_retry are solvable from the workspace alone. A denser/cleaner suite (and a
  strictly enforced constraint test like CacheCoordinator) would increase power.
- The archive keeps asking the SAME upstream question E16 left open: whether ANY
  future-blind survival signal can beat retrieval-feedback survival. E17 shows that
  even the value of correct retrieval is not visible at fixed budgets on
  these tasks — model-noise and patch-format were controlled, but coding-model
  capability is now the binding constraint.

External reviewers: read experiments/results/e17_coding_capability_report.md and
the D34 brain.md entry for the complete OBSERVED/INFERRED/UNRESOLVED breakdown.