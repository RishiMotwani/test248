# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-16)

**Core value:** Correction handling must work; write-time salience must be real; context-pressure experiments validate recall-per-token under genuine budget stress
**Current phase:** All phases complete (1–6); P2 context-pressure grid running and complete

## Completed Phases

- **Phase 1** — Foundation (dead code cleanup + pytest infra) · commit `0880f1b`
- **Phase 2** — Correction supersession fix · commit `be75897`
- **Phase 3** — Write-time salience + offline sync · commit `70866c3`
- **Phase 4** — Evaluation fixes + manifest versioning · commit `134d5b5`
- **Phase 5** — Dashboard + quick gate + regression tests · commits `f011d2c`, `91fa6b4`
- **Phase 6** — Revalidation + documentation + final audit · (this session) complete

## Key Metrics (Post-fix, 3-seed validation — `manifest_h3795b2da.json`)

- correction_recall: 1.0 (pre-fix 0.0)
- wrongly_retained_after_correction.fraction: 0.2 (pre-fix 1.0)
- negation_recall: 1.0, trap_recall: 1.0
- E1 adaptive: 44.0 tok/turn vs baseline 323.0 (pre-fix ~19): +7.9% vs vanilla (< 15% gate)
- E2 proposed positive recall: 0.760 (pre-fix 0.72)
- E10 context-pressure grid: 48 cells × 4 budgets × 4 turns × 3 seeds, word-count tokens, lexical embedding model
  - Probe recall (point-of-need, strict entity match, final-store): adaptive 8.4% @ 340 store-tok, summarization 10.5%→84.8% (budget-proportional), memgpt/vanilla 100% @ ~4875 toks
  - Efficiency (recall per store token): adaptive ~2.5e-4, summarization ~2.1e-4 linear with budget, unbounded baselines ~2.0e-4
  - Budget-stressed cells (raw tokens > budget): 500 turns at 512 budget stressed; 4096 budget never stressed at 50 turns
  - store recall (E2 durable): 0.76 adaptive, 1.0 baseline (blended FP ceiling)
- E4 hard-case: dist 10 adaptive 0.75/baseline 1.0; dist 25 adaptive 0.0/baseline 1.0 (transient coffee facts intentionally forgotten + degenerate unstressed baseline)

## E12 Coding-Context Usefulness Benchmark (D28) — NEGATIVE for adaptive

- `data/coding_workload.py` + `experiments/e12_coding_benchmark.py`; 84 diverse facts, 400 turns, budgets {128,256,512}, 3 seeds, nomic-embed-text (production retrieval path). Budget genuinely binding (natural store 1024 tok = 2–8× budget).
- 3-seed mean query-time injected context_recall:
  - adaptive: 0.155 / 0.167 / 0.167 (flat across budgets despite store_recall 0.19→0.35→0.53)
  - vanilla_rag: 0.905 @ ~60 injected tok; memgpt_style 0.905 @ ~121 tok; summarization_only 0.179→0.333→0.583; sliding_window 0.048
- Root causes: (1) bounded store + decay retains only ~37% of facts at top budget; (2) importance-dominant retrieval `0.6·imp+0.4·sim` — for a query, importance spans 0.47 vs similarity span 0.069, so injection is query-insensitive. Ablation @512: production 0.167, pure-similarity 0.421, pure-importance 0.115.
- Retrieval weights NOT tuned (directive: do not optimise adaptive to beat the benchmark). D3 "calibrate" flagged as remediation.
- Fairness fixes (not tuning): facts spread across whole session (was first-half + decay-prune artifact); distinct predicate families replace near-duplicate templates (dedupe retains ~86–88%, was 332→189).
- Tests: 33 passing (`tests/test_coding_benchmark.py` 9).

## Notes

- Codebase map at `.planning/codebase/` — all 7 docs committed
- `brain.md` governs all memory_optimizer/server.py changes (D24 correction supersession, D25 write-time salience, D26 eval/versioning/dashboard/audit, D27 context-pressure probe recall, D28 coding-context usefulness benchmark)
- Git: fresh repo at RishiMotwani/test248 (public), branch master
- Session relocated to prototype3
- Two worktrees removed: `prototype3_0880f1b_wt`, `prototype3_pre_reval_wt`
- /tmp cleaned; artifacts on /home filesystem

## Instrumentation gaps (honest list, for reviewers)

- E3 latency unmeasured offline (null in manifest)
- MM multi-model variance never measured
- No budget-stress experiment at raw-history > 4096 (only ~480+ turns stress)
- E4 distance-25 recall 0.0 not yet root-caused
- Negation recall has no quick-gate threshold
- No post-fix live-verified run (only stale pre-fix live manifest)

## Known issues (for reviewers)

1. Post-fix token-vs-correctness tradeoff: +121% vs pre-fix adaptive, gate basis ambiguous (REVAL-04)
2. E4 needle recall drops to 0.0 at 25-turn distance
3. Stale live manifest (pre-fix) can be mistaken for current results
4. Embed-path dependence changes per-seed misuse fraction (0.0 vs 0.2)
5. Default 50-turn config does not stress 4096 budget; barebone fits entirely
6. D27: point-of-need probe recall isolates retained content from budget effects

---

State initialized: 2026-09-16
Last updated: 2026-09-17 after Phase 6 P2 grid completion + P8 latency benchmark