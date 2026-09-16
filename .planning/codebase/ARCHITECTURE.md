---
last_mapped_commit: 58c99f010198b7e814e289d17df8cb7133f8ebdb
last_mapped_at: 2026-09-16
---
<!-- refreshed: 2026-09-16 -->

# Architecture

**Analysis Date:** 2026-09-16

## System Overview

```text
┌───────────────────────────────────────────────────────────────────┐
│                      Presentation Layer                            │
│   ui/index.html (vanilla-JS SPA, 1460 lines)                      │
│   server.py (FastAPI app, uvicorn 127.0.0.1:9002)                 │
├───────────────────────────────────────────────────────────────────┤
│                     Orchestration Layer                            │
│   Live:  server.py state dict + _ingest_turn / _prepare_chat_turn │
│   Paper: experiments/paper.py run_seed / replay_adaptive          │
│   Base:  baselines/baseline_runner.py BaselineRunner              │
├───────────────────────────┬───────────────────────────────────────┤
│                           │                                       │
│        Adaptive arm       │        Baseline arms                  │
│  memory_optimizer/pipeline.py   baselines/{sliding_window,        │
│  (AdaptiveMemoryPipeline)       memgpt_style, summarization_only, │
│                           │       vanilla_rag}.py (BaseBaseline)  │
├───────────────────────────┴───────────────────────────────────────┤
│                     Policy Components (library)                   │
│   scoring.py  decay.py  compression.py  budget.py                 │
│   retrieval.py  extraction.py  embeddings.py                      │
├───────────────────────────────────────────────────────────────────┤
│                    External: local Ollama LLM                      │
│   /api/generate (llm_model, default llama3.1:8b)                  │
│   /api/embed (embedding_model, default nomic-embed-text)          │
│   /api/tags (model list) — HTTP requests, no SDK                  │
└───────────────────────────────────────────────────────────────────┘
```

Two runtime faces share one memory-policy core:

1. **Live dashboard** (`server.py` + `ui/index.html`) — interactive head-to-head demo
   (adaptive vs sliding-window baseline) against a real local Ollama model, plus a
   research-job console that spawns the offline experiment CLIs as subprocesses.
2. **Offline paper suite** (`run_experiments.py` → `experiments/paper.py`) —
   deterministic seeded replays (E1–E6) over synthetic conversations, comparing the
   adaptive method against four baselines on the **same pre-extracted fact stream**
   (writer held fixed, brain.md D9).

`brain.md` is the governing design document: every change to `memory_optimizer/` or
`server.py` must trace back to a numbered decision (D1…D23) recorded there.

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `server.py` | FastAPI app: chat/demo/research/settings endpoints, in-memory `state`, token measurement, SSE streaming, subprocess research job runner | `server.py` |
| `AdaptiveMemoryPipeline` | The unified ingest orchestrator — same byte-for-byte component in live server and offline paper replay (D20) | `memory_optimizer/pipeline.py` |
| `ImportanceScorer` | Multi-factor base score (relevance/utility/recency/frequency) | `memory_optimizer/scoring.py` |
| `CategoryDecayEngine` | Usage-fed exponential decay per category; prunes below threshold | `memory_optimizer/decay.py` |
| `MemoryCompressor` | Semantic dedupe (embedding-cosine or lexical overlap); summarizer is a deliberate stub | `memory_optimizer/compression.py` |
| `token_budget_evict` / `fit_to_budget` | Single source of truth for budget semantics (D1/D2/D7) | `memory_optimizer/budget.py` |
| `MemoryRetriever` | Dense embedding cosine retrieval with importance-weighted re-rank; lexical fallback; reinforcement on injection | `memory_optimizer/retrieval.py` |
| `FactExtractor` | LLM atomic-fact extraction with category normalization + keyword fallback | `memory_optimizer/extraction.py` |
| `embed_ollama` | Ollama embedding access with process-wide `(endpoint, model, text) → vector` cache | `memory_optimizer/embeddings.py` |
| `BaseBaseline` + `BaselineRunner` | Uniform policy interface + batch runner for the four baseline arms | `baselines/baseline_runner.py` |
| `experiments/paper.py` | Offline paper engine: stream generation, seeded replays, E1–E6 evaluation, aggregation, manifests | `experiments/paper.py` |
| `experiments/live.py` | Live-computed E1–E6 metrics as pure functions over server state (used by `/api/experiments/run`) | `experiments/live.py` |

## Pattern Overview

**Overall:** Layered research pipeline with a strategy pattern for memory policies
(adaptive vs baseline) over a shared writer. The pipeline (ingest → score → measure
tokens → dedupe → decay → budget-evict → retrieve) is the fixed skeleton; only the
policy behind retention differs between the arms.

**Key Characteristics:**

- The **writer is held fixed** across all methods (brain.md D9): every baseline consumes
  the exact same pre-extracted fact stream as the adaptive system
- **Single source of truth**: `AdaptiveMemoryPipeline` is the only ingest path in both
  live and offline code; `memory_optimizer/budget.py` is the only budget implementation
  the pipeline calls
- **Honesty labels**: every result records `metric_source` and `token_source`
  (`measured` vs `estimated(word-count)`) so no number pretends to be measured when it
  was estimated
- **Deterministic research**: seeded generation (`seed 42+i`), SHA-based manifest
  filenames, regenerable gold sets and artifacts

## Layers

**Presentation:**

- Purpose: Render live memory state, head-to-head comparisons, and research-job output
- Location: `server.py` (API) + `ui/index.html` (SPA)
- Contains: FastAPI routes, SSE streaming, CORS middleware, one self-contained HTML/JS dashboard
- Depends on: orchestration layer functions in `server.py`, `memory_optimizer/*`
- Used by: browser at `http://127.0.0.1:9002/dashboard`

**Orchestration:**

- Purpose: Drive turns through the pipeline, maintain session state, build comparison payloads, spawn research jobs
- Location: module-level functions in `server.py` (`_ingest_turn`, `_prepare_chat_turn`, `_build_comparison`, `_run_llm_demo`, `_run_research_job`) and `experiments/paper.py` (`run_seed`, `run_paper`)
- Contains: turn lifecycle, baseline window simulation, token calibration (`_calibrated_constants`), demo message construction (`_build_demo_messages`)
- Depends on: `memory_optimizer/*`, `baselines/*`
- Used by: presentation layer endpoints; `run_experiments.py`

**Policy Components:**

- Purpose: Pure-ish policy logic with no API knowledge — each stage of the adaptive method is one module
- Location: `memory_optimizer/` (7 modules)
- Contains: `scoring.py`, `decay.py`, `compression.py`, `budget.py`, `retrieval.py`, `extraction.py`, `embeddings.py`
- Depends on: `requests` (HTTP to Ollama), `numpy`; delivery of injected context is the caller's job
- Used by: `pipeline.py`, `server.py`, `experiments/paper.py`, `experiments/live.py`, `baselines/*`

**External:**

- Purpose: Local LLM inference and embeddings
- Location: `ollama_base` from `config.yaml` (`http://localhost:11434`)
- Contains: `/api/generate` (extraction, answer, window replay, token measurement), `/api/embed` (nomic-embed-text 768-dim), `/api/tags`
- Used by: `memory_optimizer/extraction.py`, `memory_optimizer/embeddings.py`, `server.py`, `experiments/paper.py`

## Data Flow

### Primary Request Path — Live Chat

1. UI sends `POST /api/chat/stream` with `ChatMessage` (`server.py:1097`)
2. `_begin_chat()` acquires `_chat_lock`, checks demo/research/chat busy guards (`server.py:947`)
3. `_prepare_chat_turn()` (`server.py:961`) → `FactExtractor.extract_facts()` (2-retry LLM call) → `_ingest_turn()` (`server.py:594`)
4. `_ingest_turn` delegates to `AdaptiveMemoryPipeline.ingest()` (`memory_optimizer/pipeline.py:80`) — five timed stages: scoring → fact-token measurement → dedupe → decay+prune → budget evict → retrieval
5. Pipeline result merges into the `state` dict (active/pruned/budget-evicted memories, `raw_history`, token traces, timeline)
6. `retriever.retrieve(message, active_memories, ranked=True)` produces the UI ranking view (`server.py:988`)
7. `_llm_stream_answer()` (`server.py:902`) streams the LLM answer back as SSE `data:` frames; final `done` frame carries `detail` + `comparison` payload
8. `_end_chat()` releases the lock

### Secondary Flow — Demo Replay (head-to-head)

1. `POST /api/demo/scenario` (`server.py:1144`) → `_build_demo_messages()` (`server.py:742`) constructs a seeded 20–500 turn conversation with planted facts, density re-mentions, corrections and negations
2. `_reset_demo_state()` clears all memory state; `_run_llm_demo()` starts on a daemon thread (`server.py:872`)
3. Per turn: extract facts → `_ingest_turn(..., replay=True)`. Replay path additionally runs `_llm_window_replay()` (sliding-window baseline answer) and `_llm_answer()` (adaptive answer), storing both in `state["baseline_replay"]`
4. `_build_comparison()` (`server.py:380`) computes token/cost/recall deltas vs the simulated baseline window (`_baseline_window`, `server.py:249`)

### Tertiary Flow — Research Jobs (subprocess)

1. `POST /api/research/run` (`server.py:1536`) validates pipeline name against `RESEARCH_PIPELINES` (`server.py:1213`) and coerces params
2. `_run_research_job()` (`server.py:1420`) runs `[sys.executable] + argv` as `subprocess.Popen` (new session, line-buffered stdout capture), progress parsed from `seed N` log lines
3. Log cap at 500 lines; cancel via `os.killpg(proc.pid, 9)`; history persisted to `experiments/results/jobs_history.json`

### Offline Paper Flow

1. `run_experiments.py` CLI → `run_paper()` (`experiments/paper.py:545`) → `run_seed()` per seed (`server.py:322` equivalent: `paper.py:322`)
2. `stream_from_generator()` produces turns+ground truth via `SyntheticConversationGenerator` (`data/synthetic_generator.py`)
3. Adaptive arm: `replay_adaptive()` (`paper.py:129`) drives the same `AdaptiveMemoryPipeline` over the stream. Baseline arms: `BaselineRunner.run()` (`baselines/baseline_runner.py:204`)
4. `evaluate_method()` (`paper.py:182`) computes E1 (token efficiency), E2 (recall/forgetting), E3 (latency), E4 (needle-by-distance), E6 (power); `_adaptive_ablations()` (`paper.py:403`) adds E5
5. `aggregate()` (`paper.py:467`) pools per-turn series, bootstrap CIs, Cohen's d, Holm/Bonferroni multiple-comparison corrections
6. `write_manifest()` (`paper.py:585`) writes `experiments/results/manifest_{sha8}.json` (SHA of config) and updates `latest_manifest.json`

### Live Experiment Flow

1. `POST /api/experiments/run` (`server.py:1931`) requires a finished demo (state present)
2. `experiments/live.compute_all_json()` (`experiments/live.py:389`) evaluates E1–E6 as pure functions over `state`, stream, and ground truth
3. Result written to `experiments/results/live_manifest.json`

**State Management:**

- The module-level `state` dict in `server.py:85` is the in-memory database: active/pruned/budget-evicted memories, `raw_history`, token traces, latency stats, timeline (capped 10k), baseline replay, demo/research job records
- No persistence except: `jobs_history.json` (research job log), experiment manifests/JSON results, and `config.yaml` (settings rewritten by `_persist_settings()`, `server.py:189`)
- Embeddings are cached in-memory only (`memory_optimizer/embeddings.py:14`) — never persisted across restarts

## Key Abstractions

**`AdaptiveMemoryPipeline`** (`memory_optimizer/pipeline.py:42`):

- Purpose: Unify the live and offline ingest paths so the measured component is byte-for-byte the production component (D20)
- Examples: constructed lazily by `server._get_pipeline()` (`server.py:547`) and directly by `experiments/paper.py:151`
- Pattern: Constructor takes `store`/`pruned` lists by reference so the pipeline mutates the caller's containers; `on_stage(op, ms)` callback lets the server collect latency without the pipeline knowing the API

**`BaseBaseline`** (`baselines/baseline_runner.py:57`):

- Purpose: Policy interface for all baseline arms — `observe(turn)` / `held_facts()` / `retrieve(query)`; shared token fitting (`fit_entries`) and similarity ranking (`rank_by_similarity`)
- Examples: `SlidingWindowBaseline`, `MemGPTStyleBaseline`, `SummarizationOnlyBaseline`, `VanillaRAGBaseline`
- Pattern: Template method; `BaselineRunner._make()` (line 181) maps method names to classes

**`CategoryDecayEngine`** (`memory_optimizer/decay.py:5`):

- Purpose: Usage-fed exponential forgetting keyed on time-since-last-access, category-conditioned lambdas
- Formula (from docstring): `M(t) = min(1, base_score * (1 + gain * (access_count − 1))) * exp(−λ_c · (current_turn − last_access_turn))`
- Pattern: `step_decay_and_prune()` mutates each memory's `current_importance` and returns `(active, pruned)`; pruned items carry a human-readable `prune_reason`

**`MemoryRetriever`** (`memory_optimizer/retrieval.py:48`):

- Purpose: Real dense retrieval with honesty about the engine used per candidate
- Pattern: `_score_all()` scores without mutation (copies + `_source` backref); `retrieve(ranked=False)` bumps `access_count`/`last_access_turn` on selected memories (reinforcement feeding decay), `retrieve(ranked=True)` is the read-only UI view; each item stamped with `retrieval_engine` (`embedding` | `lexical`)

**`FactExtractor`** (`memory_optimizer/extraction.py:112`):

- Purpose: LLM atomic-fact extraction with a deterministic category override (`_keyword_category` wins over the model's category)
- Pattern: Up to 2 attempts; on total failure returns `([], meta["fallback"]=True)` so nothing is silently written — caller must handle the empty stream

**`token_budget_evict` / `fit_to_budget`** (`memory_optimizer/budget.py:34,71`):

- Purpose: Shared hard-cap semantics: eviction by lowest `current_importance` (oldest source turn tie-break); injection by greedy token-aware fill
- Pattern: Pure functions taking `fact_tokens` callable; word-count fallback when measurement unavailable

**`TokenMeasurer`** (`experiments/paper.py:441`):

- Purpose: Offline content-token measurement via Ollama `prompt_eval_count` (num_predict=1)
- Pattern: Requests session reused, per-text cache; `None` on failure

## Entry Points

**FastAPI server:**

- Location: `server.py:41`, run via `python server.py` (uvicorn on `HOST`/`PORT`, default `127.0.0.1:9002`)
- Triggers: HTTP requests from `ui/index.html`; endpoints — `/api/chat`, `/api/chat/stream`, `/api/chat/baseline`, `/api/demo/scenario`, `/api/demo/job`, `/api/demo/job/cancel`, `/api/research/pipelines`, `/api/research/run`, `/api/research/jobs`, `/api/research/jobs/{id}`, `/api/research/jobs/{id}/cancel`, `/api/research/board`, `/api/state`, `/api/timeline`, `/api/timeline/full`, `/api/comparison/full`, `/api/experiments/latest`, `/api/experiments/run`, `/api/models`, `/api/models/select`, `/api/settings`, `/api/gpu`, `/api/ethics/privacy_export`, `/api/ethics/purge`, `/dashboard`

**Paper experiment CLI:**

- Location: `run_experiments.py` (argparse)
- Triggers: manual shell invocation or research-job subprocess
- Responsibilities: `--quick` gate, `--seeds/--turns/--density/--methods/--write/--measured`, budget/top_k overrides, manifest writing

**Experiment scripts with `--quick` gates:** `experiments/e0_extraction_quality.py`, `experiments/e7_sensitivity_sweep.py`, `experiments/e8_external_benchmark.py`, `experiments/statistics.py`, `experiments/paper_state.py`, `experiments/qualitative_report.py`, `experiments/run_multi_model.py`, `experiments/make_paper_artifacts.py` — each independently runnable; all register in `RESEARCH_PIPELINES` for dashboard launch

**Gold set regeneration:** `data/build_gold_set.py` (seed 7, 200 turns → `data/gold_labels/extraction_gold_200.json`)

**Baseline self-test:** `baselines/baseline_runner.py` `__main__` (lexical, LLM-free wiring check)

## Architectural Constraints

- **Threading:** Single-process, threaded FastAPI app. Demo (`_run_llm_demo`, `server.py:872`) and research jobs (`_run_research_job`, `server.py:1420`) run on daemon threads, guarded by `_research_lock` / `_chat_lock`; chat is serialized via `_chat_lock` + `chat_busy` flag. The adaptive pipeline itself is not thread-safe — concurrent turns are prevented by the busy guards.
- **Global state:** Module-level `state` dict (`server.py:85`) mutated by background threads; module-level caches `_gpu_cache` (`server.py:111`), `_fact_token_cache` (`server.py:114`), `_token_constants` (`server.py:113`), lazy `pipeline` global (`server.py:545`); process-wide embedding cache `_cache` + `_lock` in `memory_optimizer/embeddings.py:14`. No persistence layer for memory state.
- **brain.md discipline:** "Nothing goes into `memory_optimizer/` or `server.py` without first landing here" — numbered design decisions (D1–D23) are the change contract; see `brain.md:1-5`.
- **Writer held fixed (D9):** All baselines consume the identical pre-extracted fact stream; only the memory policy differs.
- **Compression is dedupe-only:** `MemoryCompressor.compress_cluster` (`memory_optimizer/compression.py:38`) is an explicit placeholder — "Do not 'improve' it into a model call; consolidate via dedupe instead."
- **Budget semantics single-sourced:** Live (`pipeline.py`), offline replay (`paper.py`), and E5 replay (`live.py`) all call `memory_optimizer/budget.py` for eviction/fitting; do not implement budget logic inline elsewhere.
- **Measurement honesty:** Token counts must be labeled `measured` (Ollama `prompt_eval_count`) vs `estimated(word-count)`; E3 latency only reportable with real LLM calls.

## Anti-Patterns

### Monolithic `server.py`

**What happens:** `server.py` is 1998 lines mixing the HTTP API layer, orchestration (`_ingest_turn`, `_build_comparison`), token calibration, demo scenario generation (`_build_demo_messages` with 5 hand-written seed messages + filler templates), baseline window simulation, and the subprocess research-job runner.
**Why it's wrong:** Any change to a chat endpoint risks touching demo generation or job spawning; the file is hard to test in isolation (all state is module-global).
**Do this instead:** Split orchestration into a callable module (e.g., a `pipeline_runner`/`session` module) that both `server.py` routes and tests import — mirroring how `AdaptiveMemoryPipeline` already extracted the ingest path.

### Duplicated sliding-window / budget-fitting logic

**What happens:** Three independent implementations of the greedy window fit exist:
`server.py:_baseline_window` (`server.py:249`, with per-turn `C_label` formatting cost and eviction reasons), `experiments/live.py:_baseline_window_of` + `_per_turn_baseline_tokens` (`experiments/live.py:398,413`), and `experiments/paper.py:per_turn_window_tokens` + `final_window_ids` (`experiments/paper.py:93,105`). They differ subtly (label-cost charging, eviction-reason detail), so live E1 and paper E1 can disagree.
**Why it's wrong:** Three sources of truth drift; the paper claims comparability with the live dashboard ("same wilcoxon E1").
**Do this instead:** Centralize the baseline window in one module (e.g., `memory_optimizer/budget.py` or a new `baselines/window.py`) and have all three callers use it.

### Duplicated ablation replay

**What happens:** The four E5 ablations (full, no_decay, no_compression, equal_weight_scoring) are implemented twice: `experiments/live.py:_e5_replay`/`_e5` (`experiments/live.py:215,279`) and `experiments/paper.py:_adaptive_ablations` (`experiments/paper.py:403`).
**Why it's wrong:** Two hand-rolled copies of the ingest loop; behavior can diverge (e.g., `live._e5_replay` applies budget eviction only when `fact_tokens` is not None; `paper._adaptive_ablations` goes through the pipeline).
**Do this instead:** Run ablations through `AdaptiveMemoryPipeline` with settings toggles (as `paper.py` does) and have `live.py` reuse the same function.

### Deprecated placeholder modules left in place

**What happens:** `experiments/e1_token_efficiency.py` … `e6_power_check.py` are 8–10 line stubs returning `{"status": "deprecated", "message": "use run_experiments.py ..."}`. They remain importable and appear in the experiment listing.
**Why it's wrong:** New contributors may run the wrong entry point and get empty data; the `paper.py` docstring notes "Legacy synthetic e-modules (e1..e6_*.py) remain importable".
**Do this instead:** Either delete them or convert each into a thin re-export from `experiments/paper.py` so the entry point name stays valid but the data path is real.

### Duplicate cosine/overlap helpers

**What happens:** `_cosine`/`_overlap`/`_token_overlap` are reimplemented in `memory_optimizer/compression.py:4-24`, `memory_optimizer/retrieval.py:25-45`, and `experiments/live.py:17-23` (and `baselines/baseline_runner.py` imports the retrieval versions).
**Why it's wrong:** Divergent thresholds and normalization could change measurably (E2 recall matching uses `_overlap` in two different places with different formulas).
**Do this instead:** Keep `memory_optimizer/retrieval.py` as the canonical home (baselines already import from it) and make `live.py` and `compression.py` import those helpers.

## Error Handling

**Strategy:** Fail-soft with explicit fallback flags — the system degrades (word-count tokens, lexical retrieval, empty facts) rather than raising, while recording what mode it fell back to.

**Patterns:**

- Extraction failure → `meta["fallback"]=True`, returns `([])` — "Extraction failure must not silently make the entire raw user message durable memory" (`memory_optimizer/extraction.py:177`)
- Token measurement failure → `None` → word-count estimate (`pipeline._tok`, `server._fact_tokens`)
- Embedding failure → `retrieval_engine: "lexical"` token overlap (`memory_optimizer/retrieval.py:108-114`); transient Ollama failures never poison the embedding cache (`memory_optimizer/embeddings.py:58`)
- LLM JSON parse failure → `{}` → empty answer/facts (`server.py:326-329`)
- Research subprocess failure → `job["status"]="error"`, `exit_code` recorded, log capped at 500 lines
- NaN/Inf sanitized out of manifests (`experiments/paper.py:_sanitize`, `experiments/live.py:_sanitize`)
- Settings validation via Pydantic models (`ChatMessage`, `SettingsPayload`, `_ResearchRunPayload`) + explicit clamping on demo params

## Cross-Cutting Concerns

**Logging:** No logging framework — CLIs print to stdout (progress parsed by the research job runner via `^seed \d+` regex, `server.py:1210`); server has no logger, writes no per-request logs (`server3.log` is a leftover runtime artifact, gitignored via `*.log`).
**Validation:** Pydantic for request payloads; range-clamping in `_coerce_run` (`server.py:1319`) and demo-scenario param clamping (`server.py:1147`).
**Authentication:** None — binds `127.0.0.1` by default, CORS restricted to localhost origins (`server.py:43`); privacy endpoints (`/api/ethics/*`) expose local delete/export only.

---

*Architecture analysis: 2026-09-16*
