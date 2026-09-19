# GSD Handoff — Phase 23 (E28 model-tier certification)

Generated: after committing the Phase-23 closure (frozen E27 surface re-run at
qwen2.5:7b; gates failed identically; measurement reported — no redesign, no
winner, no head-to-head).

This doc is the single reviewer-readable artifact for the phase. Kept docs:
ROADMAP.md, STATE.md, PROJECT.md, config.json, codebase.

## Where we are

Phase 19 (commit `0354c5e`) closed E19 with `adaptive_advances = false`.
Phase 20 (E25) set four constraints (D42). Phase 21 built E26; its pilot failed
two gates → `adaptive_beats_vanilla = False` and the 54-cell full grid never
ran. **Phase 22 built E27** — a *single-policy* calibration surface (1 obsolete
policy fact vs 1 current correction vs 4 distractors per variant, obsolete+
current > 96 tokens so the two sides never both fit). Its 48-cell pilot at
llama3.1:8b failed history_dependence (all groups) and contradiction_state
(vanilla obs_exposure 0.667 < 0.75), so `calibration_valid = False` and E27
declared no head-to-head. The open question (D44) was whether that failure was
a property of the *surface* or of the *model tier*.

**Phase 23 (E28) answers exactly that question.** The frozen E27 surface was
re-run byte-identically at a second tier, `qwen2.5:7b`, through a thin
output-rebinding wrapper that delegates the whole grid + gate computation to
`e27_calibration.main` unchanged. The gate verdict is **identical to E27**:
budget_binding (1.0/1.0), context_difference (12/12) and no_leakage (0) PASS;
history_dependence and contradiction_state FAIL. `calibration_valid = False,
head_to_head_eligible = False` at qwen2.5:7b too. The E27 failure therefore
does **NOT** isolate to the llama3.1:8b tier — the surface itself remains the
leading explanation. E28 is a measurement, not a fix: no redesign was
performed and none is claimed.

## How it was run

```bash
python -m experiments.e28_model_tier_certification --dry-run     # no-LLM context construction
python -m experiments.e28_model_tier_certification --run --resume  # 48 cells at qwen2.5:7b
python -m experiments.e28_model_tier_certification --report-only   # re-render report from JSON
python -m pytest tests/ -q                                       # 372 passing (18 new E28)
```

Outputs written (new, force-added): `experiments/e28_model_tier_certification.py`;
`tests/test_e28_model_tier_certification.py`;
`experiments/results/e28_model_tier_certification.json` + `_report.md`.

## Locked earlier results (never modified, carried for the reviewer)

- **Phase 19 (E23+E24):** `correction_identity` 27/27, `embedding_consistency`
  27/27, primary 135/135, **`adaptive_advances = false`** (adaptive 27/27,
  vanilla_rag 26/27, llm 14/27, raw 13/27, sliding 9/27).
- **Phase 20 (E25):** budget geometry — budgets never bind either recall arm
  below ~84–116 tokens; D42 constraints (obsolete-only evidence unsafe, budgets
  bind both methods, discrimination above the 96.3% vanilla ceiling, never read
  E19 as adaptive superiority).
- **Phase 21 (E26):** 48-cell pilot at llama3.1:8b failed history_dependence
  and contradiction_state → `adaptive_beats_vanilla = False`; full grid not run.
- **Phase 22 (E27):** 48-cell pilot at llama3.1:8b — budget_binding/context_
  difference/no_leakage PASS; history_dependence FAIL all groups; contradiction
  state FAIL (vanilla obs 0.667 < 0.75) → `calibration_valid = False,
  full_head_to_head_eligible = False`; no winner, no head-to-head.

## E28 (frozen surface @ qwen2.5:7b) — the two-tier comparison

| gate | E27 @ llama3.1:8b | E28 @ qwen2.5:7b |
| --- | --- | --- |
| history_dependence | FAIL (all 3 groups) | FAIL (all 3 groups) |
| contradiction_state | FAIL (vanilla obs 0.667) | FAIL (vanilla obs 0.667) |
| budget_binding | PASS (1.0 / 1.0) | PASS (1.0 / 1.0) |
| context_difference | PASS (12/12) | PASS (12/12) |
| no_leakage | PASS (0) | PASS (0) |

Per-method pilot success at qwen2.5:7b: no_history 0.833, direct_history 0.75,
adaptive 0.667, vanilla_rag 0.417.

History-dependence detail at qwen2.5:7b:
- route_contract: direct 1.0 vs no_history 1.0 → diff 0 (the stronger tier
  solves the group with NO history — a new failure mode vs E27's no_history at
  the 0.75 cap);
- serialization_contract: direct 1.0 vs no_history 1.0 → diff 0 (same);
- retry_contract: direct 0.25 vs no_history 0.50 → diff -1 (direct below the
  0.75 bar, no_history above the 0.25 cap).

Contradiction-state detail at qwen2.5:7b: adaptive correction_recall 1.0 /
obsolete_fact_exposure 0.0 (clean side); vanilla_rag correction_recall 0.333 /
obsolete_fact_exposure **0.667** — the same value as E27@llama3.1:8b — since
vanilla still retrieves the current correction on the serialization/retry
variant-A cells. The obsolete-contradiction signal is inconsistent across the
grid at both tiers.

## Verdict (predeclared, phase 23)

- `head_to_head_eligible = calibration_valid`; E28 runs no head-to-head and
  ranks no model.
- **`calibration_valid = False`, `head_to_head_eligible = False` at
  qwen2.5:7b, with a gate pattern identical to E27@llama3.1:8b.** The E27
  failure is therefore **not** explained by the model tier; the calibrated
  surface itself remains the leading explanation.
- No redesign was performed: single-policy decisive facts are not enough for
  history separation on route/serialization (both tiers solve them without
  history) or retry (direct history fails to carry the behavior at both tiers),
  and the vanilla comparator keeps surfacing the current correction on the
  variant-A cells.

## Remaining questions (for any next phase)

1. **Redesign surface, not tier:** both tiers fail identically, so a stronger
   third tier is unlikely to change the verdict. Does the next phase move to
   multi-fact decisive policies so `no_history` falls harder (e.g. non-derivable
   multi-step contracts) while `direct_history` stays solvable, plus a
   per-tier solvability pre-gate (direct_history must clear a floor BEFORE the
   grid runs)?
2. **Vanilla comparator:** vanilla obs_exposure is pinned at 0.667 at both
   tiers because the recency-weighted static store keeps retrieving the current
   correction on the same 4 cells. Does the comparator need an explicit
   recency-neutral whole-history mode, or a different baseline?
3. **Stop rule:** given two tiers now both fail the identical calibration, is
   continuing the coding-benchmark line (vs stopping it) still warranted?

## Immutable (Phase-23 contract)

- All E27/E26 artifacts (suites, runners, results, reports) and all
  E19/E20/E24/E25 results and reports are byte-for-byte unchanged (git diff
  empty).
- `config.yaml`, `memory_optimizer/`, `baselines/`, `server.py` — untouched.

## Honest verdict (reviewer-confirmable)

- E28 reused the frozen E27 runner verbatim (only `--model` changed), re-ran
  the offline fixture gates (PASS on all 6 variants), completed 48/48 cells and
  reported the two-tier gate comparison. The E27 suite, runner, tests and
  results were never modified (byte-identical after E28).
- The failure is reproducible at two tiers with the identical gate pattern and
  the *identical* vanilla obs_exposure (0.667), which points at the surface
  (single-fact geometry + vanilla recency behaviour), not at llama3.1:8b.
- `calibration_valid = False`, `head_to_head_eligible = False`; no winner, no
  head-to-head, no model ranking, no redesign; production + E27/E26 + earlier
  artifacts byte-identical.

## Next decision (reviewer)

Decide whether to (a) redesign the calibration surface (multi-fact decisive
policies + per-tier solvability gate + recency-neutral comparator), (b) stop
the coding-benchmark line, or (c) some other direction. E28 itself does none of
these — it was a two-tier measurement that isolated the surface.

## Cryptographic checkpoint

- Checkpoint branch: `checkpoint/phase-23-e28-model-tier-certification-complete`
- Checkpoint SHA: the Phase-23 closure commit on master.

## Restore

```bash
git checkout checkpoint/phase-23-e28-model-tier-certification-complete
```

## Do-not-regress (locked)

1. Do not tune `config.yaml` (dual_score, retention, decay, thresholds, top_k,
   embedding model, task affinity all frozen).
2. Original E19/E20/E24/E25/E26/E27 JSON/report artifacts immutable.
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
8. E27/E28 declare no winner and never claim an adaptive-vs-RAG result;
   `head_to_head_eligible` was False at BOTH llama3.1:8b and qwen2.5:7b, so no
   head-to-head may be built on the E27 surface at either tier without a
   surface redesign.