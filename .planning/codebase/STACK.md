---
last_mapped_commit: 58c99f010198b7e814e289d17df8cb7133f8ebdb
last_mapped_at: 2026-09-16
---
# Technology Stack

**Analysis Date:** 2026-09-16

## Languages

**Primary:**

- Python 3.14.7 - All backend logic: server, memory pipeline (`memory_optimizer/`), baselines (`baselines/`), experiments (`experiments/`), data generators (`data/`). Confirmed via `venv/bin/python --version` and `cpython-314` bytecode in `__pycache__/`.

**Secondary:**

- HTML/CSS/JavaScript (vanilla, no framework) - Single-page dashboard at `ui/index.html`, served verbatim by `GET /dashboard` in `server.py:1990`. One inline `<script>` block (`ui/index.html:265`), zero CDN references, `fetch()` calls to the FastAPI endpoints.

## Runtime

**Environment:**

- Local Python 3.14 virtualenv at `venv/` (committed `.gitignore`d; activate via `venv/bin/activate`).

**Package Manager:**

- pip with exact-pinned `requirements.txt` (all 11 packages pinned, e.g. `fastapi==0.141.1`). No `pip-tools` lockfile, no `pyproject.toml`.
- Install: `pip install -r requirements.txt` (see `README.md`).

## Frameworks

**Core:**

- FastAPI 0.141.1 - REST + SSE API in `server.py` (`app = FastAPI(title="Adaptive Memory Manager API", version="1.0.0")` at `server.py:41`).
- Uvicorn 0.52.4 - ASGI server; `uvicorn.run(app, host=HOST, port=PORT)` at `server.py:1996-1998` (env `HOST` default `127.0.0.1`, `PORT` default `9002`).
- Pydantic 2.13.5 - Request validation: `ChatMessage` (`server.py:117`), `SettingsPayload` (`server.py:1893`).

**Testing:**

- None declared — no pytest/unittest test files exist in the repo. Sanity gates exist as `--quick` / `--no-llm` flags and inline `_selftest()` functions (e.g. `experiments/statistics.py:111`, `baselines/baseline_runner.py:261`).

**Build/Dev:**

- None — no bundler, transpiler, or task runner. Python runs as-is; the dashboard is static HTML.

## Key Dependencies

**Critical:**

- requests 2.34.2 - The sole HTTP client for the Ollama API (generate/embed/tags) across `server.py`, `memory_optimizer/extraction.py`, `memory_optimizer/embeddings.py`, `experiments/paper.py`.
- numpy 2.5.3 - Cosine similarity (`memory_optimizer/retrieval.py`, `memory_optimizer/compression.py`), stats (`experiments/statistics.py`, `experiments/live.py`).
- scipy 1.18.1 - Paired Wilcoxon signed-rank test: `scipy.stats.wilcoxon` in `experiments/statistics.py:80` and `experiments/live.py:7`.
- statsmodels 0.15.0 - Power analysis `TTestIndPower.solve_power` in `experiments/live.py:320-321`.
- pyyaml 6.0.3 - `config.yaml` loading (`server.py:53`, `experiments/paper.py:522`) and settings persistence (`server.py:189-200`).
- rouge-score 0.1.2 - Paraphrase matching in E0 (`experiments/e0_extraction_quality.py:40-41`, `RougeScorer(["rougeL"], use_stemmer=True)`).

**Infrastructure:**

- bert-score 0.3.13 - Declared in `requirements.txt` line 10 and in the E0 docstring (`experiments/e0_extraction_quality.py:5`) as a paraphrase match level, but **never imported** by project code (verified: no `from bert_score` / `import bert_score` outside `venv/`). Installable/available but currently dormant.
- scikit-learn 1.9.0 - Declared in `requirements.txt` line 9 but **never imported** anywhere in project code (verified: no `import sklearn` outside `venv/`). Dead dependency.
- Standard library only elsewhere: `threading` (locks in `server.py:108-114`), `subprocess` (`nvidia-smi` at `server.py:209`), `difflib.SequenceMatcher` (`server.py:423`), `uuid`, `json`, `random`, `math`.

## Configuration

**Environment:**

- No `.env` file (existence check only — not read).
- `config.yaml` at repo root is the single source of configuration:
  - `system`: `seed: 42`, `ollama_endpoint: http://localhost:11434`, `llm_model: llama3.1:8b`, `embedding_model: nomic-embed-text`, `max_context_tokens: 4096`, `injection_token_limit: 0`
  - `decay_lambdas`: per-category exponential decay rates (`transient: 0.15`, `personal: 0.005`, `technical_preference: 0.02`, `project_context: 0.01`)
  - `scoring_weights`: `w1_relevance: 0.4`, `w2_utility: 0.3`, `w3_recency: 0.15`, `w4_frequency: 0.15`
  - `pruning.threshold: 0.2`, `retrieval.top_k: 5`, `retrieval.similarity_threshold: 0.35`, `compression.enabled: true`
  - `vram.bytes_per_token_estimate`: per-family KV-cache byte estimate (`llama: 131072`, `qwen2: 57344`, `gemma4: 98304`, `default: 100000`)
- Env vars read by `server.py`: `HOST` (`server.py:1997`), `PORT` (`server.py:1998`), `CORS_ORIGINS` (`server.py:43`).
- Runtime settings are mutated in memory and **written back into `config.yaml`** by `_persist_settings()` (`server.py:189-200`) on `/api/settings` and `/api/models/select` — config.yaml doubles as the persistence store for UI-applied settings.

**Build:**

- No build step. Entry points: `python server.py` (dashboard), `python run_experiments.py` (paper suite), `python experiments/run_multi_model.py`, `python experiments/make_paper_artifacts.py`.

## Platform Requirements

**Development:**

- Linux (repo developed and run on Linux; `nvidia-smi` queried at `server.py:209`).
- Python 3.14.x (venv built with 3.14.7).

**Production:**

- Local single-machine deployment only: FastAPI on `127.0.0.1:9002` (or `HOST`/`PORT` overrides) + a reachable Ollama daemon. GPU optional but `nvidia-smi` must exist to populate the VRAM chip (`/api/gpu`).

---

*Stack analysis: 2026-09-16*
