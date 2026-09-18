# Phase-14/E18 Immutable Checkpoint

Checkpoint SHA: 07259e06f3b24a01ec67d2eb19209b51cff790ca
Checkpoint short SHA: 07259e0
Current branch: master
Checkpoint branch: checkpoint/phase-14-e18-complete
Checkpoint tag: checkpoint-phase-14-e18-07259e0

Phase: 14
Experiment: E18
Status: COMPLETED

E18 offline:
36 cells
27 correction-bearing
correction_recall = 1.0
obsolete_exposure = 0.0

E18 coding:
72 runs
adaptive = 17/18
vanilla_rag = 9/18
llm_summarization = 9/18
raw_clipped = 8/18

Production defaults:
dual_score
query-first retrieval
correction protection enabled
config.yaml unchanged

E17 artifacts:
IMMUTABLE

Restore:
git checkout checkpoint/phase-14-e18-complete

or:

git reset --hard 07259e06f3b24a01ec67d2eb19209b51cff790ca