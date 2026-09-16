---
last_mapped_commit: 58c99f010198b7e814e289d17df8cb7133f8ebdb
last_mapped_at: 2026-09-16
---
# External Integrations

**Analysis Date:** 2026-09-16

## APIs & External Services

**LLM Inference (Ollama) — the only external service:**

- Ollama local daemon at `http://localhost:11434` (configured via `config.yaml` → `system.ollama_endpoint`; overridable per-run with `--ollama-endpoint` in `run_experiments.py:43`).
- No API key/auth — plain HTTP POST/GET to the Ollama REST API via the `requests` package.
- Endpoints consumed:
  - `POST {endpoint}/api/generate` —
    - Fact extraction with `"format": "json"`, `"keep_alive": "30m"`, `"options": {"num_ctx": 8192}` at `memory_optimizer/extraction.py:135-142` (timeout 120s).
    - Baseline window replay (`WINDOW_REPLAY_PROMPT`) at `server.py:316`, answer generation (`ANSWER_PROMPT`) at `server.py:356`, streaming answer chunks (see `_llm_stream_answer`), token measurement via `prompt_eval_count` at `server.py:121-133` (`server.py:124`), model unload at `server.py:1875` (`keep_alive: 0`).
    - Paper replay path at `experiments/paper.py:456`.
  - `POST {endpoint}/api/embed` — dense embeddings (default `nomic-embed-text`, 768-dim) at `memory_optimizer/embeddings.py:46-50`; one batch call for all cache misses, timeout 60s.
  - `GET {endpoint}/api/tags` — model listing for the dashboard model picker at `server.py:1850` (timeout 5s; embedding models filtered out by name at `server.py:1856`).
- Default models (`config.yaml`): `llama3.1:8b` for generation/extraction, `nomic-embed-text` for embeddings. LLM generation is always `temperature 0.1`, `num_ctx 8192`, `keep_alive "30m"`.
- Failure behavior: non-200 responses or exceptions are swallowed — extraction falls back to empty facts with `meta["fallback"]=True` (`memory_optimizer/extraction.py:177-180`); retrieval falls back to lexical `_token_overlap` when embeddings fail (`memory_optimizer/retrieval.py:112-114`); `/api/models` returns `[]` when Ollama is down (`server.py:1851`).

**GPU Health (nvidia-smi):**

- `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits` via `subprocess.run` at `server.py:208-210`, wrapped by `_gpu_snapshot()` with a 2-second result cache (`server.py:203-217`). Feeds `/api/gpu` and the VRAM chip in the dashboard. Not a service — a local binary dependency.

## Data Storage

**Databases:**

- None. No DBMS, ORM, or persistence layer — all memory state lives in the module-level `state` dict in `server.py:85-106` (in-process, cleared on restart).

**File Storage:**

- Local filesystem only:
  - `experiments/results/*.json` — experiment manifests (`manifest_*.json`), `latest_manifest.json`, `live_manifest.json`, `e0_extraction_quality.json`, `e7_sweep.json`, `e8_e8-v1.json`, and `jobs_history.json` (research-job history persisted by `server.py:1502` / `server.py:1575`; gitignored via `experiments/results/` in `.gitignore`).
  - `data/gold_labels/extraction_gold_200.json` — 200-turn extraction gold set for E0 (committed).
  - `artifacts/` — paper outputs: `fig_data/*.json` (e.g. `e1_tokens_per_turn.json`, `e4_distance_curves.json`), `tables/table_e1_e2.tex`, `reproducibility.json` (gitignored).
  - `config.yaml` — rewritten by `_persist_settings()` (`server.py:189-200`) to persist UI-applied settings.

**Caching:**

- In-memory only, never persisted:
  - Embedding cache `(endpoint, model, text) -> vector` in `memory_optimizer/embeddings.py:14-15` (process-wide dict, thread-locked; transient Ollama failures do not poison it — `memory_optimizer/embeddings.py:58-62`).
  - Token-measurement cache `_fact_token_cache` (`server.py:114`) and per-model calibrated constants `_token_constants` (`server.py:113`, `_calibrated_constants()` at `server.py:136`).
  - GPU snapshot cache `_gpu_cache` (`server.py:111`).
- No Redis/Memcached.

## Authentication & Identity

**Auth Provider:**

- None. The server binds to localhost (`127.0.0.1:9002` default) with no auth, no sessions, no cookies. `DELETE /api/ethics/purge` (`server.py:1970`) is the only identity-adjacent endpoint — it wipes all user memory stores as a privacy control; `GET /api/ethics/privacy_export` (`server.py:1961`) dumps stored memories.

## Monitoring & Observability

**Error Tracking:**

- None (no Sentry/rollbar). Errors surface as JSON `{"status": "error", "detail": ...}` responses or are caught-and-degraded (extraction fallback, lexical fallback).

**Logs:**

- `console`-only via print/exception handling; `server.log` exists in `.gitignore` (`server3.log` present at repo root). No structured logging framework. Latency is tracked in-process in `state["latency_stats"]` and exposed via `/api/state`.

## CI/CD & Deployment

**Hosting:**

- Local single-process: `python server.py` → Uvicorn on `127.0.0.1:9002`. No container, no cloud target.

**CI Pipeline:**

- None (no GitHub Actions, no CI config).

## Environment Configuration

**Required env vars:**

- `HOST` (optional, default `127.0.0.1`) — bind address (`server.py:1997`)
- `PORT` (optional, default `9002`) — port (`server.py:1998`)
- `CORS_ORIGINS` (optional, default `http://127.0.0.1:9002,http://localhost:9002`) — comma-separated allowed origins for the CORS middleware (`server.py:43-51`); allows `GET, POST, DELETE` methods and `Content-Type` header

**Secrets location:**

- None used — no API keys, tokens, or credentials anywhere. Ollama is unauthenticated localhost. Never add secret-scraping here: there is no auth layer to protect.

## Webhooks & Callbacks

**Incoming:**

- None.

**Outgoing:**

- None. (The only "push" channel is the server's own SSE stream, `POST /api/chat/stream` → `StreamingResponse(..., media_type="text/event-stream")` at `server.py:1097-1120`, consumed by `ui/index.html:375` — internal, not an external integration.)

## Model & Runtime Dependencies

**Ollama model requirements:**

- Generation model (default `llama3.1:8b`) must be pulled locally; the dashboard can switch via `/api/models` + `/api/models/select` (`server.py:1847-1885`).
- Embedding model (default `nomic-embed-text`) must be present; retrieval silently degrades to lexical overlap if `/api/embed` fails.
- Token measurement is done live via `prompt_eval_count` from `/api/generate` with `num_predict: 1` (`server.py:129`), so a responsive Ollama is required for measured-token experiments (`--measured` in `run_experiments.py:39-40`).

---

*Integration audit: 2026-09-16*
