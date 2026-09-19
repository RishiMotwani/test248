# Checkpoint — Phase 23 complete (E28 model-tier certification, measurement reported)

Status: **CLOSED** — phase completed; two-tier gate comparison recorded honestly; no redesign (measurement only).

## Scope

Phase 23 (E28): answer D44's open question — is the E27 calibration failure a
property of the benchmark surface or of the llama3.1:8b model tier? — by
re-running the byte-identical frozen 48-cell E27 procedure at exactly one other
tier, `qwen2.5:7b`, with the same fixtures, budget 96, methods, nomic-embed-text
embeddings and E27's locked gate thresholds. E28 never redesigns the surface,
never re-runs E26/E19, never runs a head-to-head grid and never ranks models.

## What ran

1. `experiments/e28_model_tier_certification.py` (new): thin output-rebinding
   wrapper (E22/E23 pattern). Rebinds `e27.OUT_JSON/OUT_REPORT/WORK_ROOT` in a
   try/finally, delegates the whole grid + gate computation to
   `e27_calibration.main(["--pilot","--model","qwen2.5:7b", ...])`, then wraps
   the E27-shaped payload into the E28 result shape. `wrap_payload` is
   idempotent for both E27- and E28-shaped verdicts; `dry_run` builds the four
   contexts with no LLM. Flags `--run`/`--dry-run`/`--report-only`/`--resume`/
   `--force`/`--skip-model-check`/`--endpoint`/`--embedding-model`.
2. `tests/test_e28_model_tier_certification.py` (new, offline): 18 tests —
   grid surface identical to E27, output plumbing, delegation with
   `--model qwen2.5:7b` + rebinding restored in `finally`, no LLM coder in
   dry-run, E28 top-level shape + verdict forwarding, model-comparison gate
   table, 15 sections + Tables A-E, no ranking/wins language, no redefinition of
   locked thresholds, no `requests`/`server` imports. Hermetic (no
   Ollama/network, no writes to immutables).
3. No-LLM dry-run confirmed all four contexts construct at qwen2.5:7b:
   no_history 0 tokens, direct_history 51, vanilla_rag 93, adaptive 88.
4. 48-cell run at qwen2.5:7b (budget 96, seeds 1-2) — 48/48 cells complete;
   offline fixture gates PASS on all 6 variants. Report re-rendered from JSON.

## Two-tier gate comparison (locked thresholds reused verbatim)

| gate | E27 @ llama3.1:8b | E28 @ qwen2.5:7b |
| --- | --- | --- |
| history_dependence | FAIL (all 3 groups) | FAIL (all 3 groups) |
| contradiction_state | FAIL (vanilla obs 0.667) | FAIL (vanilla obs 0.667) |
| budget_binding | PASS (1.0/1.0) | PASS (1.0/1.0) |
| context_difference | PASS (12/12) | PASS (12/12) |
| no_leakage | PASS (0) | PASS (0) |

History-dependence details at qwen2.5:7b: route_contract and
serialization_contract direct_history 1.0 **and** no_history 1.0 -> diff 0 (the
stronger tier solves both groups with NO history at all — a new failure mode
relative to E27, where no_history sat at the 0.25 cap); retry_contract
direct_history 0.25 (< 0.75) vs no_history 0.50 (> 0.25), diff -1.
Contradiction-state details at qwen2.5:7b: adaptive correction_recall 1.0 /
obsolete_fact_exposure 0.0 (clean side); vanilla_rag correction_recall 0.3333 /
obsolete_fact_exposure 0.6667 (< 0.75) — the identical exposure to E27 (vanilla
still retrieves the current correction on the serialization/retry variant-A
cells). Per-method pilot success at qwen2.5:7b: no_history 10/12 (0.833),
direct_history 9/12 (0.75), adaptive 8/12 (0.667), vanilla_rag 5/12 (0.417).

## Verdict

`calibration_valid = False`, `head_to_head_eligible = False` at qwen2.5:7b,
with a gate pattern identical to E27@llama3.1:8b. The E27 failure therefore
does **NOT** isolate to the llama3.1:8b tier — the calibration surface itself
remains the leading explanation. E28 is a measurement, not a fix: no redesign
was performed and none is claimed. No winner, no head-to-head, no model
ranking, no adaptive-vs-RAG claim.

## Artifacts

- New/force-added: `experiments/e28_model_tier_certification.py`,
  `tests/test_e28_model_tier_certification.py`,
  `experiments/results/e28_model_tier_certification.json`,
  `experiments/results/e28_model_tier_certification_report.md`.
- Immutable (byte-identical, git diff empty): `memory_optimizer/**`,
  `baselines/**`, `config.yaml`, `server.py`, every E27 artifact
  (suite/runner/tests/results/report), and all E26/E19/E20/E24/E25
  results/reports. E27's JSON and report were not rewritten by the E28 run
  (the wrapper redirected all writes to its own `e28_work`/result paths).
- Tests: 372 passing (18 new E28) — same 1 pre-existing
  `PytestUnknownMarkWarning` for `@pytest.mark.integration` in
  `tests/test_phase19_closure.py:187`.

## Reasons the phase closed here

The identical frozen surface fails identically at two different model tiers
(identical gate pattern AND identical vanilla obs_exposure 0.667), so the
explanation for E27's failure is the surface, not the llama3.1:8b tier: (1)
the single decisive policy fact is insufficient to separate history dependence
at either tier — at qwen2.5:7b two groups are solved with no history at all
(diff 0) while retry_contract drops to direct 0.25, so the grid cannot measure
`direct_history - no_history` meaningfully; (2) vanilla's obsolete-only
contradiction signal is inconsistent at both tiers (0.667 < 0.75) because the
recency-weighted static store keeps surfacing the current correction on the
same variant-A cells. Both are structural findings of the surface, not
tunable within the locked contract.

## Restore

```bash
git checkout checkpoint/phase-23-e28-model-tier-certification-complete
```