# GSD Handoff — Phase 20 (E25 E19 baseline-ceiling + budget-geometry audit)

Generated: after committing the Phase-20 closure (offline audit).

This doc is the single reviewer-readable artifact for the phase. Kept docs:
ROADMAP.md, STATE.md, PROJECT.md, config.json, codebase.

## Where we are

Phase 19 (commit `0354c5e`) closed the E19 experiment: after the counterfactual
adapter repair (explicit `supersedes_turn` + current-correction metadata),
`correction_identity` = 27/27 PASS, `embedding_consistency` = 27/27, primary
grid = 135/135, and **`adaptive_advances = false`** (adaptive 27/27, vanilla_rag
26/27, llm_summarization 14/27, raw_clipped 13/27, sliding_window 9/27). One
negative-control cell was recovered separately (`write_retry / seed=3 /
budget=256 / llm_summarization`), giving combined 225/225 accounting.

**Phase 20 runs E25** — a deterministic, offline audit of that locked artifact.
It re-derives the budget geometry (how each method used the 256/512/1024
historical-token budgets), the correction states behind the success counts, the
single vanilla_rag failure, the adaptive-vs-vanilla context-hash relationship,
and four diagnostic flags that constrain any next benchmark. E25 makes **zero
LLM/embedding calls**, imports no production memory code, declares **no winner**,
changes **no ranking**, and leaves every E19/E20/E24 artifact byte-identical.

## How it was run

```bash
python experiments/e25_e19_baseline_ceiling_audit.py
python -m pytest -q tests/            # 218 passing (18 new E25 tests)
```

Outputs written (new, force-added):
`experiments/results/e25_e19_baseline_ceiling_audit.json` +
`_report.md`; plus `experiments/e25_e19_baseline_ceiling_audit.py` and
`tests/test_e25_baseline_ceiling_audit.py`.

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

## E25 budget geometry (means; binding = tokens >= 0.90·budget)

| method | 256 | 512 | 1024 | binding |
| --- | --- | --- | --- | --- |
| raw_clipped | 244 tok (0.95) | 509 (0.99) | 1018 (0.99) | 100% at every budget |
| sliding_window | 116 (0.45) | 116 (0.23) | 116 (0.11) | 0% |
| llm_summarization | 184 (0.72) | 318 (0.62) | 412 (0.40) | 0% |
| vanilla_rag | 109 (0.43) | 109 (0.21) | 109 (0.11) | 0% |
| adaptive | 92 (0.36) | 92 (0.18) | 92 (0.09) | 0% |

- Adaptive and vanilla_rag are **context-stable across 256/512/1024 on all 9
  tracks** each (SHA-identical), i.e. budget had no effect on their context.
- Raw_clipped and llm_summarization are budget-sensitive (context changes with
  budget). Adaptive/vanilla contexts **never share a context_sha** with each
  other at any budget/track.
- Observed context threshold range: adaptive 84–99 tokens; vanilla_rag 102–116.

## E25 diagnostic flags (all four True)

| flag | finding | next-benchmark constraint |
| --- | --- | --- |
| A | vanilla OBSOLETE_ONLY succeeds 26/27 (0.963 >= 0.80) | obsolete-only evidence must become materially unsafe for the hidden test |
| B | adaptive/vanilla mean utilization @256 = 0.359/0.427 (< 0.50), 9/9 stable tracks each | budget range must actually bind both recall methods (below ~84–116 tokens) |
| C | vanilla primary success 0.963 >= 0.90 (near ceiling) | more discrimination above the vanilla ceiling |
| D | `adaptive_advances` = false (locked) | do not treat E19 as evidence of adaptive superiority |

E25 reports the observed threshold range only; it does **not** choose final
numerical budgets for a future benchmark.

## Three remaining questions (for any next phase)

1. **Contradiction sensitivity:** at which budget/task does obsolete-only
   evidence (vanilla_rag's state on all 27 primary cells) actually change the
   hidden-test outcome, so correction-aware vs correction-unaware retrieval can
   be separated?
2. **Binding threshold:** exactly where does the budget start to bind adaptive
   and vanilla_rag (locate the ceiling below the current 256 low end, given the
   observed 84–116-token range)?
3. **Task design:** which task family (new hidden-test contract or sharper
   counterfactual pair) makes the final-behavior decision underivable from the
   workspace while still penalizing the obsolete alternative at the tested model
   tier (llama3.1:8b)?

## New artifacts (Phase 20, force-added)

- `experiments/e25_e19_baseline_ceiling_audit.py` (offline audit script)
- `tests/test_e25_baseline_ceiling_audit.py` (18 offline tests)
- `experiments/results/e25_e19_baseline_ceiling_audit.json` + `_report.md`

## Immutable (Phase-20 contract)

- All E19/E20/E24 results and reports (incl. `e19_coding_generalization_full_repaired.json`,
  `e20_counterfactual_history_repair.json`, `e24_missing_negative_control.json`)
  are byte-for-byte unchanged (git diff empty).
- `config.yaml`, `memory_optimizer/`, `server.py` — untouched.

## Honest verdict (reviewer-confirmable)

- Primary grid re-verified as exactly 135/135 unique, complete cells.
- The audit is diagnostic: 0 LLM calls, 0 embedding calls, no production
  pipeline import, no winner declared, no ranking changed.
- `adaptive_advances` remains `false`; E19 is **not** evidence of adaptive
  superiority over vanilla_rag at the tested budgets.

## Next decision (reviewer)

Phase 20 deliberately does not start a Phase 21. Any next benchmark must
satisfy the four constraints above before it can claim to separate adaptive
from vanilla_rag — decide whether to run such a benchmark, extend E25's
diagnostics, or stop the benchmark line.

## Cryptographic checkpoint

- Checkpoint branch: `checkpoint/phase-20-e25-audit-complete`
- Checkpoint SHA: the Phase-20 closure commit on master.

## Restore

```bash
git checkout checkpoint/phase-20-e25-audit-complete
```

## Do-not-regress (locked)

1. Do not tune `config.yaml` (dual_score, retention, decay, thresholds, top_k,
   embedding model, task affinity all frozen).
2. Original E19/E20/E24 JSON/report artifacts immutable.
3. Any new coding fixture must pass the offline base-hidden-test gate BEFORE it
   may claim history dependence.
4. History dependence of a coding benchmark requires a reproducible
   counterfactual pair at the claimed model tier (no_history vs direct_history).
5. E19 full-grid results are only eligible under the E20-validated families;
   stochastic no-history draws certify nothing.
6. Benchmark separation requires BOTH contradiction sensitivity AND
   method-specific budget pressure (D42) — one without the other cannot
   discriminate adaptive from vanilla_rag above the ceiling.