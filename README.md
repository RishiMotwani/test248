# Adaptive Memory Manager for Context-Constrained LLMs

> **PRIMARY CLAIM:** A category-conditioned exponential decay function, combined with importance-weighted retrieval, improves long-horizon recall accuracy per token compared to static memory-scoring methods (RAG, sliding window, summarization-only) — and does so by forgetting the right things, not just remembering more.

A research repository that implements and empirically tests a **category-conditioned decay memory architecture** for local SLMs (Llama 3.1 8B, Qwen 2.5 7B, Phi-3-mini) running under strict context limits (4k/8k windows).

Two faces share one memory-policy core:

1. **Live dashboard** — interactive head-to-head demo (adaptive vs sliding-window baseline) against a real local Ollama model, plus a research-job console that runs the offline experiment suite from the browser.
2. **Offline paper suite** — deterministic seeded experiments (E1–E6) over synthetic conversations, comparing the adaptive method against four baselines on the exact same pre-extracted fact stream (the writer is held fixed).

---

## Key Features

1. **Category-Conditioned Decay Engine** — forgetting curves grounded in Ebbinghaus and SuperMemo SM-2, with per-category decay rates (`transient`, `personal`, `technical_preference`, `project_context`). See `memory_optimizer/decay.py`.
2. **Multi-Factor Scoring** — linear/logistic weight fitting across relevance, utility, recency, and frequency (`memory_optimizer/scoring.py`).
3. **Dense Retrieval with Honesty Labels** — embedding-based cosine retrieval with importance-weighted re-ranking; degrades explicitly (and labels the fallback) to lexical overlap when embeddings fail (`memory_optimizer/retrieval.py`).
4. **Rigorous Experimental Suite (E1–E6)** — paired Wilcoxon signed-rank tests, bootstrap confidence intervals, power analysis, Cohen's d, Holm/Bonferroni multiple-comparison corrections, and needle-in-a-haystack visualizations (`experiments/paper.py`, `experiments/statistics.py`).
5. **Hard-Case Demos** — the dashboard can plant real conversation memories with **corrections** and **negations** to test whether each side "knows the current truth" (`/api/demo/scenario`).
6. **Privacy & Ethics Compliance** — zero-knowledge local storage, strict forgetfulness guarantees, and programmatic export/purge controls (`/api/ethics/privacy_export`, `/api/ethics/purge`).
7. **Honesty in Measurement** — every token count is labeled `measured` (Ollama `prompt_eval_count`) vs `estimated(word-count)`; no number pretends to be measured when it was estimated.

---

## Architecture

```text
Presentation          ui/index.html (vanilla-JS SPA)  +  server.py (FastAPI, uvicorn 127.0.0.1:9002)
Orchestration         Live: server.py state + _ingest_turn / _prepare_chat_turn
                      Paper: experiments/paper.py run_seed / replay_adaptive
                      Base:  baselines/baseline_runner.py BaselineRunner
Policy library        memory_optimizer/  (scoring, decay, compression, budget,
                      retrieval, extraction, embeddings, pipeline)
Baselines             baselines/  (sliding_window, memgpt_style, summarization_only, vanilla_rag)
External              Local Ollama — /api/generate, /api/embed, /api/tags (HTTP, no SDK)
```

The single ingest path is `AdaptiveMemoryPipeline.ingest()` in `memory_optimizer/pipeline.py` (`scoring → token measurement → dedupe → decay+prune → budget evict → retrieval`) — used byte-for-byte by both the live server and the offline replay, so the measured component is the production component.

Budget semantics are single-sourced in `memory_optimizer/budget.py` (`token_budget_evict`, `fit_to_budget`), and 4096 tokens of context is the default arena (`config.yaml`).

The governing design document is `brain.md` — every change to `memory_optimizer/` or `server.py` traces back to a numbered decision (D1…D23). *Nothing goes into `memory_optimizer/` or `server.py` without first landing there.*

---

## Repository Layout

```
memory_optimizer/     # Core adaptive memory policy library (7 modules + pipeline)
baselines/            # Comparison arms (sliding window, MemGPT, summarization, vanilla RAG)
experiments/          # Paper engine (paper.py), live metrics (live.py), e0–e9 scripts
  └── results/        # Manifests + JSON outputs (gitignored)
data/                 # Synthetic conversation generator + E0 gold set
ui/                   # Single-file dashboard SPA (index.html)
artifacts/            # Regenerable paper tables/figures (gitignored)
server.py             # FastAPI dashboard + live demo server
run_experiments.py    # Offline paper experiment CLI
config.yaml           # Single config source: model, budget, decay, retrieval
brain.md              # Living design-decision document (D1–D23)
requirements.txt      # Pinned dependencies
```

---

## Quick Start

### 1. Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires a **local Ollama** daemon with the generation and embedding models pulled:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 2. Run the dashboard

```bash
python server.py
```

Open <http://127.0.0.1:9002/dashboard>. Environment overrides:

| Env var | Default | Purpose |
|---|---|---|
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `9002` | Port |
| `CORS_ORIGINS` | `http://127.0.0.1:9002,http://localhost:9002` | Comma-separated allowed CORS origins |

> **Important:** the server is unauthenticated. Keep it on loopback; do not expose it beyond `127.0.0.1` without adding auth.

### 3. Run the offline experiment suite

```bash
python run_experiments.py --quick          # fast deterministic wiring gate (LLM-free)
python run_experiments.py                  # full suite: 5 seeds, 150 turns, oracle writer
python run_experiments.py --seeds 3 --turns 200 --methods adaptive,sliding_window
```

Common flags: `--quick`, `--seeds N`, `--turns N`, `--density PCT`, `--methods list`,
`--write oracle|extract`, `--measured` (live token measurement via Ollama),
`--model`, `--embedding-model`, `--ollama-endpoint`, `--budget`, `--top-k`,
`--conflict-density`, `--negation-density`.

### 4. Standalone experiment scripts

Each is independently runnable and carries a `--quick` wiring gate:

```bash
python experiments/e0_extraction_quality.py --quick        # extraction vs 200-turn gold set
python experiments/e7_sensitivity_sweep.py --quick         # retriever/budget geometry sweep
python experiments/e8_external_benchmark.py --quick        # LongMemEval-style needled QA
python experiments/e9_correction_isolation.py --quick      # correction/negation isolation probe
python experiments/statistics.py                            # statistics helpers self-test
python baselines/baseline_runner.py                         # baseline wiring self-test
```

> Legacy `e1_token_efficiency.py` … `e6_power_check.py` are deprecated 8-line stubs that
> redirect to `run_experiments.py`. Use `run_experiments.py` as the authoritative entry point.

---

## Dashboard Tour

The dashboard at `ui/index.html` (served by `server.py` at `/dashboard`) provides:

- **Head-to-head demo** (`/api/demo/scenario`) — a seeded 20–500 turn synthetic conversation with planted facts, density re-mentions, **corrections (0–8)** and **negations (0–4)**. Each turn streams the adaptive answer *and* a sliding-window baseline answer, and the "knows the current truth" table grades both sides against the ground truth (chain-aware: corrections supersede, negations invert).
- **Chat** (`/api/chat/stream`) — talk to the model with adaptive memory injection, SSE streaming, and a live VRAM chip.
- **Memory inspector** — active / pruned / budget-evicted memories, token traces, retrieval engine per fact (`embedding` vs `lexical`), timeline.
- **Research console** (`/api/research/run`) — launch the paper experiments from the browser as subprocesses, watch logs, cancel jobs.
- **Live experiments** (`/api/experiments/run`) — compute E1–E6 metrics over the finished demo state; results go to `experiments/results/live_manifest.json`.
- **Model & settings** — switch generation model (`/api/models`), tune budget/top-k/weights (persisted back to `config.yaml`).
- **Ethics controls** — export all stored memories (`/api/ethics/privacy_export`) or purge them (`/api/ethics/purge`).

Key API endpoints: `/api/chat`, `/api/chat/stream`, `/api/demo/scenario`, `/api/demo/job`, `/api/demo/job/cancel`, `/api/research/pipelines`, `/api/research/run`, `/api/research/jobs`, `/api/research/board`, `/api/state`, `/api/timeline`, `/api/comparison/full`, `/api/experiments/latest`, `/api/experiments/run`, `/api/models`, `/api/settings`, `/api/gpu`, `/api/ethics/*`, `/dashboard`.

---

## Configuration (`config.yaml`)

Single source of truth, loaded at import by `server.py` and via `load_settings()` in `experiments/paper.py`. Runtime UI changes are written back with `yaml.safe_dump(..., sort_keys=False)`.

```yaml
system:
  seed: 42
  ollama_endpoint: http://localhost:11434
  llm_model: llama3.1:8b
  embedding_model: nomic-embed-text
  max_context_tokens: 4096
  injection_token_limit: 0
decay_lambdas:            # per-category exponential decay rates
  transient: 0.15
  personal: 0.005
  technical_preference: 0.02
  project_context: 0.01
scoring_weights:
  w1_relevance: 0.4
  w2_utility: 0.3
  w3_recency: 0.15
  w4_frequency: 0.15
pruning:
  threshold: 0.2
retrieval:
  top_k: 5
  similarity_threshold: 0.35
compression:
  enabled: true
vram:
  bytes_per_token_estimate: {llama: 131072, qwen2: 57344, gemma4: 98304, default: 100000}
```

The only external service is local Ollama (unauthenticated HTTP). All Ollama calls use `temperature 0.1`, `num_ctx 8192`, `keep_alive "30m"`; the dashboard caches GPU snapshots (`nvidia-smi`), fact-token measurements, and embeddings in-process. Failures degrade openly — extraction falls back to empty facts with `meta["fallback"]=True`, retrieval to lexical overlap with `retrieval_engine: "lexical"`, token measurement to a word-count estimate.

---

## Experiments (E1–E17)

| ID | Experiment | Entry point |
|---|---|---|
| E0 | Extraction quality vs 200-turn gold set | `experiments/e0_extraction_quality.py` |
| E1 | Token efficiency (tokens per turn, adaptive vs baselines) | `run_experiments.py` |
| E2 | Memory accuracy / recall / forgetting (incl. hard cases) | `run_experiments.py` |
| E3 | Latency (LLM-real) | `run_experiments.py` |
| E4 | Needle-in-a-haystack (recall by distance) | `run_experiments.py` / `experiments/live.py` |
| E5 | Ablations (no-decay, no-compression, equal-weights) | `run_experiments.py` |
| E6 | Statistical power analysis | `run_experiments.py` / `experiments/live.py` |
| E7 | Sensitivity sweep (retriever/budget geometry) | `experiments/e7_sensitivity_sweep.py` |
| E8 | LongMemEval-style needled-QA benchmark | `experiments/e8_external_benchmark.py` |
| E9 | Correction/negation isolation probe (task E regression) | `experiments/e9_correction_isolation.py` |
| E10 | Context-pressure probe recall (point-of-need) | `run_experiments.py` / `experiments/paper.py` |
| E12 | Coding-context usefulness benchmark (+E13 generalization/ablations, E14 retention diagnosis, E15 retention-policy separation, E16 retention selectivity) | `experiments/e12_coding_benchmark.py`, `experiments/e16_retention_selectivity.py` |
| E17 | Long-horizon coding capability (hidden-test pass at fixed historical budgets) | `experiments/coding_benchmark.py` + `experiments/e17_coding_capability.py` |

E17 is resumable: it rewrites `e17_coding_capability.json` after every run and
skips completed `task:seed:budget:method` cells on restart. Run the full grid with
`python experiments/e17_coding_capability.py --full` (pilot = default; `--report-only`
reclassifies the saved JSON without reruns).

Baselines (each a `BaseBaseline` subclass in `baselines/`): `SlidingWindowBaseline`, `MemGPTStyleBaseline`, `SummarizationOnlyBaseline`, `VanillaRAGBaseline`. All consume the identical pre-extracted fact stream (writer held fixed, brain.md D9). E17 adds `raw_clipped` (`baselines/raw_clipped.py`) and `llm_summarization` (`baselines/llm_summarization.py`) plus a shared word-count tokenizer (`memory_optimizer/tokenizer.py`), and the shared coding harness in `experiments/coding_benchmark.py`.

Results are written to `experiments/results/` (gitignored): `manifest_<sha8>.json`, `latest_manifest.json`, `live_manifest.json`, plus per-experiment outputs and `jobs_history.json` (research-job log). Paper tables and figure data go to `artifacts/`.

---

## Testing & Verification

Verification is a **pytest suite + assert-based script gates**:

```bash
python -m pytest tests/                      # 134 tests: compression supersession, salience, retrieval ranking, retention policy/selectivity, coding-capability harness
python run_experiments.py --quick            # whole-stack wiring gate (50 turns, 1 seed, oracle)
                                             #   + hard-case gate: fails if correction_recall < 0.5
                                             #     or wrongly_retained.fraction > 0.5
python experiments/e9_correction_isolation.py --quick   # standalone correction/negation probe
python experiments/statistics.py             # assert-based statistics self-test
python baselines/baseline_runner.py          # LLM-free baseline wiring self-test
python experiments/make_paper_artifacts.py --check   # anti-drift artifact/manifest check
```

The server chains the quick gates into a single job (`RESEARCH_GATES` in `server.py`) so the dashboard can run them as a sanity check.

### Post-fix validation results (Phase 6)

| Metric | Pre-fix | Post-fix (3-seed) | Target | Status |
|---|---|---|---|---|
| `correction_recall` | 0.0 | 1.0 | >= 0.5 | PASS |
| `wrongly_retained_after_correction.fraction` | 0.8 (0.8 gate) | 0.2 (1/5) | <= 0.5 | PASS |
| `negation_recall` | 0.0 | 1.0 | honest report | PASS |
| `trap_recall` | 1.0 | 1.0 | honest report | PASS |
| E1 regression (adaptive vs vanilla) | +122% | +7.9% | < 15% | PASS |
| E2 proposed positive recall | 0.72 | 0.76 | report | — |

The 3-seed run used `--seeds 3 --turns 50 --conflict-density 0.1 --negation-density 0.1`
(`manifest_h3795b2da.json`).

---

## Conventions for Contributors

- **brain.md first.** Record the design decision (D-number) before changing `memory_optimizer/` or `server.py`.
- New baseline → subclass `BaseBaseline`, implement `observe()` / `held_facts()` / `retrieve()`, register in `BaselineRunner._make()`.
- New experiment script → reuse `load_settings` / `run_seed` / `replay_adaptive` from `experiments/paper.py`, add a `--quick` gate, register in `RESEARCH_PIPELINES`.
- Keep budget semantics in `memory_optimizer/budget.py` and overlap helpers in `memory_optimizer/retrieval.py` — do not copy `_cosine` / `_overlap` into new files.
- Module-private helpers get a leading underscore; results carry provenance labels (`metric_source`, `token_source`).
- Fail-soft with explicit fallback flags; return `None` (not `0.0`) for unknown metrics; sanitize `NaN`/`inf` at manifest boundaries.
- No secrets exist in this repo and none should be added — the only "integration" is unauthenticated localhost Ollama.

---

## Privacy & Ethics

- All memory state lives in-process (`server.py` `state` dict) — nothing leaves the machine.
- Forgetfulness is real: pruned memories are removed by the decay engine (category-conditioned thresholds), and budget evictions are deterministic by lowest importance.
- `/api/ethics/privacy_export` returns all stored and pruned memories; `/api/ethics/purge` wipes the stores.
- Bound to `127.0.0.1` by default; keep it that way (no auth layer exists).