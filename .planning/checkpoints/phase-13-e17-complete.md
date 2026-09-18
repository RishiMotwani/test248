# Phase-13/E17 Immutable Checkpoint

Checkpoint SHA: 4a9e0328a8cd2e291b06e9922740ac5054bb92ca
Checkpoint short SHA: 4a9e032
Current branch: master
Checkpoint branch: checkpoint/phase-13-e17-complete
Checkpoint tag: checkpoint-phase-13-e17-4a9e032

Phase: 13
Experiment: E17
Status: COMPLETED

Working tree before Phase 14:
PREEXISTING_WORKTREE = CLEAN
(`git status --porcelain` empty at checkpoint creation; nothing captured or discarded.)

E17 primary artifacts:
- experiments/results/e17_coding_capability.json
- experiments/results/e17_coding_capability_report.md

E17 production adaptive policy:
- retention: dual_score
- correction protection: enabled
- Phase-8 query-first retrieval retained
- no Phase-14 changes applied yet

E17 must remain immutable.
Do not rewrite or regenerate historical E17 result artifacts during Phase 14.

Restore:
git checkout checkpoint/phase-13-e17-complete

or:

git reset --hard 4a9e0328a8cd2e291b06e9922740ac5054bb92ca

---

## Final checkpoint verification (output captured at gate time)

- `git rev-parse HEAD` → `4a9e0328a8cd2e291b06e9922740ac5054bb92ca`
- `git log -1 --oneline` → `4a9e032 docs: fill Phase 13 commit hash in state`
- `git branch --show-current` → `master`
- `git remote -v` → `origin https://github.com/RishiMotwani/test248.git`
- `git rev-parse checkpoint/phase-13-e17-complete` → `4a9e0328a8cd2e291b06e9922740ac5054bb92ca`
- `git rev-parse checkpoint-phase-13-e17-4a9e032^{commit}` → `4a9e0328a8cd2e291b06e9922740ac5054bb92ca`
- `git rev-list -n 1 checkpoint-phase-13-e17-4a9e032` → `4a9e0328a8cd2e291b06e9922740ac5054bb92ca`

Note: the annotation is created with `git tag -a` (annotated tag). `git rev-parse`
on an annotated tag returns the tag object id (here `bec6091...`); the tag peels
(`^{commit}`) to the exact checkpoint SHA above. The checkpoint branch resolves
directly to the checkpoint SHA.

E17 result snapshot at gate time (from the immutable JSON): 180 records, 12
diagnostic runs, 4 gold checks, `gates_all_passed=True`, `adaptive_advances=False`,
pilot_only=False, per-method success 0.722–0.75.

Preexisting-changes inventory: none.