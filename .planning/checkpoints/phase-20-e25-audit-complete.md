# Phase-20 E25 Audit Complete Immutable Checkpoint

Starting HEAD: 54088fa8c0dffd0d6958bd0f2bebf6c582f8cf66 (Phase-19 docs reconciliation on top of 0354c5e)
Closure commit: 6f6ccc61f2ea203d46081a417e94f2cd12412f04
Checkpoint branch: checkpoint/phase-20-e25-audit-complete
Checkpoint tag: checkpoint-phase-20-e25-audit-complete-6f6ccc6

Phase: 20
Experiment: E25 — offline audit of the E19 baseline ceiling and budget geometry
Status: COMPLETED

Objective:
Deterministic, artifact-only audit of the locked E19 primary grid that
separates WHY adaptive_advances = false (budget pressure not binding the
recall methods vs. contamination/saturation) and sets input constraints for
any next benchmark. No winner, no ranking change.

Offline discipline:
0 LLM calls, 0 embedding calls, no production memory code imported, no
network. Source material: e19_coding_generalization_full_repaired.json +
data/counterfactual_task_suite.py fixtures.

Primary grid validation:
135/135 unique, complete cells, non-null context_sha on every record (PASS)
Grid key: (task_id, seed, budget, method) for all combinations.

Budget geometry (mean context tokens / mean utilization):
raw_clipped         244/0.953    509/0.994    1018/0.994   binding 100% all budgets
sliding_window      116/0.453    116/0.227    116/0.113    binding 0%
llm_summarization   184/0.719    318/0.622    412/0.403    binding 0%
vanilla_rag         109/0.427    109/0.213    109/0.107    binding 0%
adaptive            92/0.359     92/0.180     92/0.090     binding 0%
(binding = tokens >= 0.90 * budget, diagnostic only)

Budget elasticity / stability (SHA-based, never token-count):
adaptive: 9/9 tracks context-stable across 256/512/1024
vanilla_rag: 9/9 tracks context-stable
sliding_window: 9/9 tracks context-stable
raw_clipped: 0/9 (budget-sensitive)
llm_summarization: 0/9 (budget-sensitive)
adaptive vs vanilla_rag context_sha equal: 0 tracks at every budget.

Correction-state classification (exact mapping; unknown pair = hard stop):
adaptive: 27x CLEAN_CURRENT
vanilla_rag: 27x OBSOLETE_ONLY
raw_clipped: 27x NEITHER
sliding_window: 27x NEITHER
llm_summarization: 5 CLEAN_CURRENT / 16 OBSOLETE_ONLY / 6 NEITHER
Correction-pair metadata: 1 pair per primary task, verified from variant A
fixtures and cross-checked against E19 offline gate cells.

Success by correction state (primary):
adaptive CLEAN_CURRENT 27/27 (1.000)
vanilla_rag OBSOLETE_ONLY 26/27 (0.963)
raw_clipped NEITHER 13/27 (0.481)
sliding_window NEITHER 9/27 (0.333)
llm_summarization OBSOLETE_ONLY 7/16 (0.438), NEITHER 4/6 (0.667),
                  CLEAN_CURRENT 3/5 (0.600)

Vanilla_rag failure analysis:
1 failed cell of 27 -> serialization_policy / seed=2 / budget=1024 /
OBSOLETE_INFORMATION_USED, 102 tokens, context_sha 50c96fd1d0bf2f96,
OBSOLETE_ONLY correction state.

Diagnostic flags (all four True):
flag_a vanilla OBSOLETE_ONLY success 0.963 >= 0.80 -> TRUE
flag_b adaptive/vanilla utilization @256 0.359/0.427 < 0.50 and 9/9 stable
      tracks each -> TRUE
flag_c vanilla primary success 0.963 >= 0.90 -> TRUE
flag_d adaptive_advances (locked) false -> TRUE

Next-benchmark requirement constraints (derived, no budgets chosen):
1. make obsolete-only historical evidence materially unsafe for the hidden test
2. use a budget range that actually binds both adaptive and vanilla retrieval
3. more discrimination against strong retrieval than E19 currently provides
4. do not treat E19 as evidence of adaptive superiority; adaptive_advances
   remains false
Observed threshold range (reported, not selected): adaptive 84-99 tokens,
vanilla_rag 102-116 tokens.

Conclusion:
Offline + declarative. No winner, no ranking change, adaptive_advances
unchanged (false), E19 artifacts untouched.

Tests:
py_compile: PASS (e25 script + e25 tests)
E25 targeted tests: PASS (18)
full suite `python -m pytest -q tests/`: PASS (218)

Immutable historical artifacts:
PRESERVED — all E19/E20/E24 JSON/report artifacts byte-identical (git diff empty)

New artifacts:
experiments/e25_e19_baseline_ceiling_audit.py
tests/test_e25_baseline_ceiling_audit.py
experiments/results/e25_e19_baseline_ceiling_audit.json sha256
22e94ec63d02906f35dc06b2ca52e82026e8800d1b108a352580638aea374759
experiments/results/e25_e19_baseline_ceiling_audit_report.md sha256
e2e696b197875f3b6af91a60cbecec163fe42afc98eea4688e296b4aafeaa0de

Production defaults:
UNCHANGED (config.yaml, memory_optimizer/, server.py untouched)

Commit: research: audit E19 baseline ceiling and budget geometry (6f6ccc6)
Push: OK

Final worktree: clean except pre-existing results/tmp_* gitlink dirtiness
(33 submodule refs under results/, untouched and not committed).