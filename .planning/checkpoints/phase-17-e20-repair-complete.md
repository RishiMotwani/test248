# Phase-17/E20 Repair Immutable Checkpoint

Checkpoint SHA: <SHA>
Checkpoint short SHA: <SHORT_SHA>
Current branch: master
Checkpoint branch: checkpoint/phase-17-e20-repair-complete
Checkpoint tag: checkpoint-phase-17-e20-repair-<SHORT_SHA>

Phase: 17
Experiment: E20 (repair revalidation via e21 wrapper)
Status: COMPLETED

Grid:
54 records
3 groups x 2 variants x 3 seeds x 3 methods
budget 1024 words, llama3.1:8b @ localhost:11434
run: python -m experiments.e21_counterfactual_history_repair --run --force

Fixture replacement:
identifier_policy -> routing_policy (opaque op->lane mapping; assignment exists
only in variant hidden tests, gold patches, and 600-turn histories;
workspace/prompt byte-identical across A/B)
final groups: [routing_policy, retry_policy, serialization_policy]

Offline Hard Gate 1 (base workspace must fail, gold must pass):
PASS (6/6 variants: base_hidden_test_pass=False, patch applies, hidden pass)

Pair-integrity:
PASS (real per-variant workspace/prompt hash A==B; routing hashes reproducible)

Hard Gate 2 (history dependence, /6):
routing_policy: dh 6/6, nh 0/6, separation 6, PASS
retry_policy: dh 5/6, nh 3/6, separation 5, PASS
serialization_policy: dh 6/6, nh 1/6, separation 6, PASS

Verdict:
history_dependence_benchmark = VALID
e19_full_grid_eligible = TRUE (running the E19 full grid = reviewer's decision)

Original E20 artifacts (Phase 16):
IMMUTABLE - e20_counterfactual_history.json sha256
1aeb771cd40c4ed9e837e1171cd25554fadcd55d06d1ab4070c98b2dbef69584
e20_counterfactual_history_report.md sha256
46edec2a82c5202cb1b790843cc7e271e962ac62191b3429891c3075d044ecb4

Repair artifacts (Phase 17, force-added):
e20_counterfactual_history_repair.json sha256
b0e1894a079913c706a4fa63b42e634f9cf47108ef158983acfed7a8d03c912f
e20_counterfactual_history_repair_report.md sha256
0a3ba3aa72faf795a63b9a7c9968453361838341ce878e40f9e3aec712723320

Production defaults:
UNCHANGED (dual_score, query-first retrieval, correction protection,
config.yaml untouched)

E17/E18/E19 artifacts:
IMMUTABLE

Restore:
git checkout checkpoint/phase-17-e20-repair-complete

or:

git reset --hard <SHA>