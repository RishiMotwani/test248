# Phase-19 Closed Immutable Checkpoint

Checkpoint SHA: a79be6cd78cf38c6d663f360834613cc29bd9df4
Checkpoint short SHA: a79be6c
Current branch: master
Checkpoint branch: checkpoint/phase-19-closed
Checkpoint tag: checkpoint-phase-19-closed-a79be6c

Phase: 19
Experiment: E19 coding generalization, correction-identity adapter repair + full-grid re-run
Status: COMPLETED

Starting HEAD: 37d1a32
Final HEAD: a79be6c

Objective:
Repair the counterfactual benchmark adapter so existing correction metadata
reaches the structured history facts consumed by the production supersession
path, then rerun E19 from fresh state.

Correction adapter:
PASS — explicit supersedes_turn + current-correction metadata now propagated
into history[*]["facts"] in build_variant() without changing history text,
workspace, prompt, hidden tests, or gold patches.

Embedding consistency:
27/27 (PASS)

Correction identity:
27/27 (PASS) — explicit supersedes_turn enables production supersession path

Primary grid:
135/135 complete

Historical full artifact:
224 records (one negative-control cell timed out during Phase-19 run)

Recovered missing negative-control:
write_retry / seed=3 / budget=256 / llm_summarization

Recovered artifact:
experiments/results/e24_missing_negative_control.json
sha256: 9f8c2e4a7b3d1f6e8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2

adaptive = 27/27
vanilla_rag = 26/27
adaptive_advances = false

Immutable historical artifacts:
PRESERVED — all Phase-17/18 E20 and E19 artifacts byte-identical

Tests:
py_compile: PASS (e24, test_phase19_closure)
counterfactual self-test: PASS
targeted tests: PASS (63 tests)
full pytest: PASS (189 tests)

Preflight (e23 --preflight):
embedding_consistency: 27/27 PASS
correction_identity: 27/27 PASS

Full grid (e23 --full --force):
records: 224 (1 cell timed out)
primary: 135
negative-control: 89
seeds: [1, 2, 3]
budgets: [256, 512, 1024]
methods: [raw_clipped, sliding_window, llm_summarization, vanilla_rag, adaptive]

Gates (10):
embedding_consistency: PASS (27/27)
correction_identity: PASS (27/27)
history_dependence: PASS
method_separation: PASS
no_leakage: PASS
gold_passes: PASS
budget_pressure: PASS
real_summarization: PASS
adaptive_production_path: PASS
unit_tests_pass: PASS (189)
all_passed: True

Paired comparisons:
adaptive vs raw_clipped: adaptive 27/27 vs baseline 13/27 mean_diff=0.5185
adaptive vs sliding_window: adaptive 27/27 vs baseline 9/27 mean_diff=0.6667
adaptive vs llm_summarization: adaptive 27/27 vs baseline 14/27 mean_diff=0.4815
adaptive vs vanilla_rag: adaptive 27/27 vs baseline 26/27 mean_diff=0.0370

adaptive_advances: false

Historical Phase-18 artifacts preserved: YES
E20 artifacts preserved: YES

New artifacts:
e19_coding_generalization_full_repaired.json sha256
5d1f58c27c88679c01a1c0ae93ed9b3b5fd8bcf89cc9bbd25cd152a110ac3962
e19_coding_generalization_full_repaired_report.md sha256
03ddf8a04277d56bcb9eed6ec8bec218d34b1de982f908dfd66ef2b374d59792
e24_missing_negative_control.json sha256
9f8c2e4a7b3d1f6e8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2

Production defaults:
UNCHANGED (config.yaml, memory_optimizer/, server.py untouched)

Commit: a79be6c
Push: OK

Final worktree: clean (except pre-existing tmp_* scratch files)