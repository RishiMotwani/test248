# Phase-16/E20 Immutable Checkpoint

Checkpoint SHA: (this commit — set at commit time)
Checkpoint short SHA: (this commit)
Current branch: master
Checkpoint branch: checkpoint/phase-16-e20-complete
Checkpoint tag: checkpoint-phase-16-e20-<short-sha>

Phase: 16
Experiment: E20
Status: COMPLETED

Grid:
54 records
3 groups x 2 variants x 3 seeds x 3 methods
budget 1024 words, llama3.1:8b @ localhost:11434

Hard Gate 1 (gold patch validity):
PASS (6/6 variants)

Pair-integrity:
PASS

Hard Gate 2 (history dependence, /6):
identifier_policy: dh 6/6, nh 6/6, separation 0, FAIL
retry_policy: dh 6/6, nh 3/6, separation 3, PASS
serialization_policy: dh 5/6, nh 0/6, separation 5, PASS

Verdict:
history_dependence_benchmark = INVALID
e19_full_grid_eligible = FALSE

Production defaults:
UNCHANGED (dual_score, query-first retrieval, correction protection,
config.yaml untouched)

E17/E18/E19 artifacts:
IMMUTABLE (E19 report received a presentation-only heading/column fix)

Restore:
git checkout checkpoint/phase-16-e20-complete

or:

git reset --hard <full-sha-of-this-commit>