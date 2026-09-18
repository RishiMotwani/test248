# Phase-18/E19-Full-Grid Immutable Checkpoint

Checkpoint SHA: 6181f01acfe5393777922a5cd8fbb4faa9dcfc4d
Checkpoint short SHA: 6181f01
Current branch: master
Checkpoint branch: checkpoint/phase-18-e19-full-grid-complete
Checkpoint tag: checkpoint-phase-18-e19-full-grid-6181f01

Phase: 18
Experiment: E19 coding generalization, aligned to the E20-validated benchmark
Status: COMPLETED

Alignment (E19 <-> E20):
PRIMARY_TASKS = [routing_policy, retry_policy, serialization_policy]
E19_COUNTERFACTUAL_VARIANTS = all Variant A (the E20-validated family, fixed)
E19 primary records keyed by the unsuffixed family task_id
(meta: counterfactual=True, variant="A", group==task_id)
NEGATIVE_CONTROL_TASKS = [cache_readonly, write_retry]
Gate C = E20 counterfactual history calibration (check_e20_calibration),
not stochastic E19 no-history draws.

Grid:
225 records (135 primary + 90 negative-control)
3 primary x 3 seeds (1,2,3) x 3 budgets (256,512,1024) x 5 methods
2 negative x 3 seeds x 3 budgets x 5 methods
llama3.1:8b @ localhost:11434, nomic-embed-text, temperature 0.1,
max_output_tokens 700, shared_word_count tokenizer
run: python experiments/e22_e19_full_grid.py --full --force
(resume: python experiments/e22_e19_full_grid.py --full --resume)

Gates (10):
embedding_consistency PASS (27/27 cells consistent)
correction_identity FAIL (obsolete fact remains in the adaptive store in all
  27 correction-bearing cells; counterfactual correction texts do not restate
  the prior fact with full word coverage, so the locked _is_supersession
  heuristic does not fire)
history_dependence PASS (E20 counterfactual history calibration)
method_separation PASS (max distinct contexts 4)
no_leakage PASS (0 leaking runs)
gold_passes PASS (5/5: routing/retry/serialization/cache_readonly/write_retry)
budget_pressure PASS (history exceeds max budget; raw grows; fills 60pct)
real_summarization PASS (324 summary_update_calls)
adaptive_production_path PASS (27 adaptive runs)
unit_tests_pass PASS (185 passed)
all_passed = False -> adaptive_advances = False (locked verdict logic; no
gate weakening, no adaptive tuning)

Verdict:
gates_all_passed = False
adaptive_beats_all_baselines = False
adaptive_advances = False
(paired 27-cells/baseline: adaptive 27/27; raw_clipped 13/27,
sliding_window 9/27, llm_summarization 14/27, vanilla_rag 27/27)

New artifacts (Phase 18, force-added):
e19_coding_generalization_full.json sha256
116159cc285e0ac8866e8d72f94104176f5c89289d8228939a87dbe68b402381
e19_coding_generalization_full_report.md sha256
9c9fda34614e9264b42086b8bf8684952ee7543cfc2779f137bcd823c164c904

Historical artifacts:
IMMUTABLE - e19_coding_generalization.json / _report.md,
e20_counterfactual_history.json / _repair.json / both _report.md (git diff empty)

Production defaults:
UNCHANGED (config.yaml, memory_optimizer/, server.py untouched)

Restore:
git checkout checkpoint/phase-18-e19-full-grid-complete

or:

git reset --hard 6181f01acfe5393777922a5cd8fbb4faa9dcfc4d