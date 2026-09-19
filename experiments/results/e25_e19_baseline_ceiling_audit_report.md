# E25: E19 baseline ceiling and budget-geometry audit

## 1. Purpose

E25 is an offline, deterministic audit of the locked E19 primary grid. It re-verified 135/135 primary cells, the per-method success rates behind the E19 verdict, and the budget geometry. adaptive and vanilla_rag both operated far below the 256 budget (adaptive mean utilization < 0.5, vanilla_rag likewise) with context stable across every track; vanilla_rag's obsolete-only correction state still succeeded >= 0.80 of the time, including its single OBSOLETE_INFORMATION_USED failure; and the E19 verdict adaptive_advances remains false. These four diagnostics set input requirements for any next benchmark, but this audit declares no winner and changes no ranking.

## 2. Method

Offline, deterministic, artifact-only audit of `experiments/results/e19_coding_generalization_full_repaired.json` plus the in-repo counterfactual fixtures (`data/counterfactual_task_suite.py`). No LLM, no embedding calls, no production memory code. The E19 artifact is read-only; this audit writes only `e25_e19_baseline_ceiling_audit.json` and this report.

## 3. Source artifact

- Source: `/home/goku/prototype/prototype3/experiments/results/e19_coding_generalization_full_repaired.json`
- Primary cells audited: `135 cells`
- Task metadata: `data/counterfactual_task_suite.py`

## 4. Scope

- Tasks: routing_policy, retry_policy, serialization_policy; seeds [1, 2, 3]; budgets [256, 512, 1024]; methods raw_clipped, sliding_window, llm_summarization, vanilla_rag, adaptive.

## 5. Assertions

Before any analysis, the primary grid is validated as exactly 135 unique, complete cells with non-null `context_sha` on every record; any mismatch stops the audit and writes nothing.

## 6. Budget geometry (Table A)

### A. Per-method budget use (means)

| Method | Budget | Mean tokens | Mean utilization | Mean headroom | Fraction binding |
|---|---|---|---|---|---|
| raw_clipped | 256 | 244 | 95.3% | 4.7% | 100.0% |
| raw_clipped | 512 | 509 | 99.4% | 0.6% | 100.0% |
| raw_clipped | 1024 | 1018 | 99.4% | 0.6% | 100.0% |
| sliding_window | 256 | 116 | 45.3% | 54.7% | 0.0% |
| sliding_window | 512 | 116 | 22.7% | 77.3% | 0.0% |
| sliding_window | 1024 | 116 | 11.3% | 88.7% | 0.0% |
| llm_summarization | 256 | 184 | 71.9% | 28.1% | 0.0% |
| llm_summarization | 512 | 318 | 62.2% | 37.8% | 0.0% |
| llm_summarization | 1024 | 412 | 40.3% | 59.7% | 0.0% |
| vanilla_rag | 256 | 109 | 42.7% | 57.3% | 0.0% |
| vanilla_rag | 512 | 109 | 21.3% | 78.6% | 0.0% |
| vanilla_rag | 1024 | 109 | 10.7% | 89.3% | 0.0% |
| adaptive | 256 | 92 | 35.9% | 64.1% | 0.0% |
| adaptive | 512 | 92 | 18.0% | 82.0% | 0.0% |
| adaptive | 1024 | 92 | 9.0% | 91.0% | 0.0% |

Binding = `historical_context_tokens >= 0.90 * historical_budget` (diagnostic only).

## 7. Budget elasticity

Per task:seed:method track, contexts are compared across 256/512/1024 by `context_sha`. A track is context-stable iff it shows exactly one distinct SHA across the three budgets. Token-count equality alone is never a stability test.

Per-track elasticity (tokens by budget, token range, distinct context count, context/token stability) is reported in the JSON under `budget_geometry` and `context_hash_analysis`; the aggregate cross-budget stability appears in Table C.

## 8. Budget binding by method (Table C, right columns)

Estimated from per-track context stability:

| Method | Stable across 256/512/1024 | Budget-sensitive contexts |
|---|---|---|
| raw_clipped | 0/9 | 9/9 |
| sliding_window | 9/9 | 0/9 |
| llm_summarization | 0/9 | 9/9 |
| vanilla_rag | 9/9 | 0/9 |
| adaptive | 9/9 | 0/9 |

## 9. Correction states

Each primary cell is classified from `(correction_recall, obsolete_fact_exposure)`: CLEAN_CURRENT = (1.0, 0.0), CURRENT_PLUS_OBSOLETE = (1.0, 1.0), OBSOLETE_ONLY = (0.0, 1.0), NEITHER = (0.0, 0.0). Any other pair stops the audit.

## 10. Correction-pair verification

Every primary task has exactly one counterfactual correction pair (verified from variant A fixtures and the E19 offline gate cells): {'routing_policy': [('rp.a.inter.001', 'rp.a.corr.001')], 'retry_policy': [('rp.a.inter.001', 'rp.a.corr.001')], 'serialization_policy': [('sp.a.inter.001', 'sp.a.corr.001')]}.

## 11. Success by correction state (Table B)

| Method | Correction state | Cells | Successes | Success rate |
|---|---|---|---|---|
| raw_clipped | NEITHER | 27 | 13 | 48.1% |
| sliding_window | NEITHER | 27 | 9 | 33.3% |
| llm_summarization | CLEAN_CURRENT | 5 | 3 | 60.0% |
| llm_summarization | OBSOLETE_ONLY | 16 | 7 | 43.8% |
| llm_summarization | NEITHER | 6 | 4 | 66.7% |
| vanilla_rag | OBSOLETE_ONLY | 27 | 26 | 96.3% |
| adaptive | CLEAN_CURRENT | 27 | 27 | 100.0% |

## 12. Task x budget analysis (Table D)

| Task | Method | 256 | 512 | 1024 |
|---|---|---|---|---|
| routing_policy | raw_clipped | 0/3 | 0/3 | 0/3 |
| routing_policy | sliding_window | 0/3 | 0/3 | 0/3 |
| routing_policy | llm_summarization | 0/3 | 0/3 | 2/3 |
| routing_policy | vanilla_rag | 3/3 | 3/3 | 3/3 |
| routing_policy | adaptive | 3/3 | 3/3 | 3/3 |
| retry_policy | raw_clipped | 2/3 | 2/3 | 3/3 |
| retry_policy | sliding_window | 2/3 | 1/3 | 1/3 |
| retry_policy | llm_summarization | 1/3 | 2/3 | 3/3 |
| retry_policy | vanilla_rag | 3/3 | 3/3 | 3/3 |
| retry_policy | adaptive | 3/3 | 3/3 | 3/3 |
| serialization_policy | raw_clipped | 1/3 | 3/3 | 2/3 |
| serialization_policy | sliding_window | 1/3 | 3/3 | 1/3 |
| serialization_policy | llm_summarization | 2/3 | 2/3 | 2/3 |
| serialization_policy | vanilla_rag | 3/3 | 3/3 | 2/3 |
| serialization_policy | adaptive | 3/3 | 3/3 | 3/3 |

## 13. Vanilla_rag failure analysis (Table E)

Primary vanilla_rag: 26/27 (96.3%). Failed cells: 1.

| Cell | Failure class | Context tokens | Utilization | Correction state |
|---|---|---|---|---|
| serialization_policy:2:1024 | OBSOLETE_INFORMATION_USED | 102 | 10.0% | OBSOLETE_ONLY |

## 14. Context-hash relationship (adaptive vs vanilla_rag)

| Budget | Same context_sha (tracks) | Different (tracks) |
|---|---|---|
| 256 | 0 | 9 |
| 512 | 0 | 9 |
| 1024 | 0 | 9 |

adaptive and vanilla_rag therefore produce different serialized historical contexts on every primary track at every budget while being byte-for-byte stable across budgets themselves.

## 15. Diagnostic flags

| Flag | Value | Support |
|---|---|---|
| flag_a_vanilla_obsolete_only_success_rate | True | see JSON |
| flag_b_budget_non_binding | True | see JSON |
| flag_c_vanilla_near_ceiling | True | see JSON |
| flag_d_adaptive_advances_false | True | see JSON |

## 16. E19 gate alignment

The locked E19 `budget_pressure` gate passed because *some* run filled >= 60% of budget at a budget level and raw_clipped grew with budget, not because adaptive or vanilla_rag were budget-bound. With adaptive mean utilization at 256 of 35.9% and vanilla_rag of 42.7%, both recall methods ran far below the tightest tested budget. E25's binding statement is diagnostic only and does not invalidate the E19 gate: the two gates measure different things.

## 17. Next-benchmark requirements (constraints only)

- the next benchmark must make obsolete-only historical evidence materially unsafe for the hidden test
- the next benchmark must use a budget range that actually binds both adaptive and vanilla retrieval
- the next benchmark needs more discrimination against strong retrieval than E19 currently provides
- do not treat E19 as evidence of adaptive superiority; adaptive_advances remains false

Observed context threshold range from current data: adaptive 84.0--99.0 tokens; vanilla_rag 102.0--116.0 tokens. E25 does not select final numerical budgets.

## 18. Conclusion

| Metric | Value |
|---|---|
| raw_clipped | 13/27 (48.1%) |
| sliding_window | 9/27 (33.3%) |
| llm_summarization | 14/27 (51.8%) |
| vanilla_rag | 26/27 (96.3%) |
| adaptive | 27/27 (100.0%) |
| adaptive_advances (E19 verdict) | False (locked, unchanged) |

E25 is offline and declarative: 0 LLM calls, 0 embedding calls, no production pipeline import, no winner declared, no ranking changed, E19 artifacts untouched. The four diagnostics set input constraints for any next benchmark; they are not an E19 verdict change and not a claim that all baselines are saturated.
