# Phase 12 Checkpoint

Created: 2026-09-17 (start of Phase 13 / E17 work)

## Checkpoint identity

- **Checkpoint SHA:** `6b25a80c3fd13fffd9929f797e3b5ece72c26bdc`
- **Checkpoint branch:** `checkpoint/phase-12-retention-selectivity` (IMMUTABLE — never moved)
- **Current master SHA at checkpoint:** `6b25a80` (`docs: add E16 raw results and report`)
- **Remote:** `https://github.com/RishiMotwani/test248.git` (origin/master)

## Frozen production configuration

- **Retention policy:** `retention.mode = dual_score`, `retention.eviction_priority = retention_priority`
- **Retrieval weights:** `sim_weight=0.85`, `imp_weight=0.15`, `cat_bonus=0.02` (query-first retrieval)
- **Retrieval params:** `top_k=5`, `sim_threshold=0.35`
- **Budget semantics:** `memory_store_token_budget` (long-term store, default 4096) separated from `max_context_tokens` (active context) and `injection_token_limit` (historical-context budget exposed to the model); `max_context_tokens` default 1800, `injection_token_limit` default 600
- **Embedding model:** `nomic-embed-text` (via local Ollama on localhost:11434)
- **Coding model:** `llama3.1:8b` (local Ollama) — confirmed available at checkpoint
- **Decay:** current_importance category lambdas; hard_threshold legacy prune replaced by soft retention for dual_score

## Recent experiment results (Phase 11/12)

- **E15 (Phase 11):** separating activation from survival SUPPORTED; advanced default to `dual_score` + `retention_priority` under stress store 1024 ctx 0.368–0.503 vs soft 0.322–0.484
- **E16 (Phase 12):** `task_affinity_advances=False` (0/8 criteria; c1/c6/c7/c8 FAIL); all policies ≈ at identity_store_recall; oracle ≈ dual at low store; config unchanged after E16
- **E14 (Phase 10):** decay→prune was dominant store-loss mechanism pre-dual_score

## Known weaknesses

- No real-coding-capability evaluation yet: E10/E14 measure fact/token recall, not patch-based coding success
- E10 used "barebone" as raw-history clipping curve, not a standalone baseline method (E17 fixes via `baselines/raw_clipped.py`)
- Prior coding benchmark (E12) measured injected-context recall, not hidden-test pass
- `/tmp` was 100% full during E16 — logs/artifacts must be written under the repo
- No prior hidden-test-based deterministic coding harness exists in repo

## Exact restore command

```bash
git reset --hard 6b25a80c3fd13fffd9929f797e3b5ece72c26bdc
# or, preferably:
git checkout checkpoint/phase-12-retention-selectivity
```