---
last_mapped_commit: 58c99f010198b7e814e289d17df8cb7133f8ebdb
last_mapped_at: 2026-09-16
---
# Coding Conventions

**Analysis Date:** 2026-09-16

## Naming Patterns

**Files:**

- Python modules: `snake_case.py` matching the class/concern, e.g. `memory_optimizer/scoring.py` (class `ImportanceScorer`), `memory_optimizer/decay.py` (class `CategoryDecayEngine`), `baselines/vanilla_rag.py` (class `VanillaRAGBaseline`).
- Experiment modules: `e{n}_descriptor.py` (`experiments/e1_token_efficiency.py`, `experiments/e4_needle_in_haystack.py`) plus task-lettered utilities (`experiments/paper.py`, `experiments/paper_state.py`, `experiments/live.py`). Legacy e1–e6 modules are deprecated stubs kept for import compatibility.
- Single-file UI: `ui/index.html` (CSS + JS inline, no separate `.js`/`.css` files).
- Config: `config.yaml` at repo root.

**Functions:**

- `snake_case` for all Python functions (e.g. `token_budget_evict`, `fit_to_budget`, `step_decay_and_prune`, `build_extraction_prompt`).
- Module-private helpers always get a leading underscore: `_token_overlap`, `_cosine`, `_word_tokens`, `_sanitize`, `_recall_of`, `_wilcoxon_effect`, `_keyword_category`, `_normalize_category`.
- Legacy experiment entry points follow `run_e{n}_<name>`: `run_e1_token_efficiency`, `run_e4_needle_in_haystack` in `experiments/e1_token_efficiency.py` / `experiments/e4_needle_in_haystack.py`.
- JavaScript in `ui/index.html` uses `camelCase`: `sendMessage()`, `loadDemo()`, `renderState()`, `applyBudget()`, `demoParams()`, `cancelDemoJob()`. Private helpers use a leading underscore: `_normText`, `_h2hMatch`, `_build_comparison` (Python, in `server.py`).
- FastAPI routes are `snake_case` under `/api/...`: `chat_endpoint`, `demo_scenario`, `research_run`, `privacy_purge` in `server.py`.

**Variables:**

- `snake_case` throughout; short names used in hot loops (`mem`, `gt`, `ex`, `pr`, `t`).
- Server state lives in a single module-level `state` dict with snake_case keys (`active_memories`, `pruned_memories`, `last_budget_evictions`) defined in `server.py`.
- JSON payload keys are snake_case and stable across files (e.g. `source_turn_id`, `last_access_turn`, `current_importance`, `eviction_reason` — see `memory_optimizer/budget.py`, `memory_optimizer/decay.py`).

**Types:**

- `PascalCase` type aliases for callables: `FactTokens = Callable[[str], int]` (`memory_optimizer/budget.py`, `memory_optimizer/retrieval.py`, `baselines/baseline_runner.py`, `experiments/paper.py`), `EmbedFn = Callable[[List[str]], list]` (`baselines/baseline_runner.py`), `PercentCallback = Optional[Callable[[str, float], None]]` (`memory_optimizer/pipeline.py`).
- `PascalCase` classes everywhere: `ImportanceScorer`, `CategoryDecayEngine`, `MemoryRetriever`, `MemoryCompressor`, `FactExtractor`, `AdaptiveMemoryPipeline`, `BaselineRunner` + the four baselines (`SlidingWindowBaseline`, `MemGPTStyleBaseline`, `SummarizationOnlyBaseline`, `VanillaRAGBaseline`), `TokenMeasurer`, `SyntheticConversationGenerator`, Pydantic request models `ChatMessage` / `SettingsPayload` / `_ResearchRunPayload` in `server.py`.

**Constants:**

- `UPPER_SNAKE_CASE` module-level constants: `VALID_CATEGORIES`, `CATEGORY_MAP`, `TECH_WORDS`, `PROJECT_PHRASES` (`memory_optimizer/extraction.py`); `WINDOW_REPLAY_PROMPT`, `ANSWER_PROMPT`, `FILLERS`, `SEED_MESSAGES` (`server.py`); `BASE_DIR`, `RESEARCH_PIPELINES`, `RESEARCH_GATES`, `_RESEARCH_LOG_CAP` (`server.py`); `MODEL_SET` (`experiments/run_multi_model.py`); `SUITE_VERSION` (`experiments/e8_external_benchmark.py`); `BASELINE_POINT`, `DIMENSIONS` (`experiments/e7_sensitivity_sweep.py`); `MATCH_OVERLAP = 0.7` in `baselines/baseline_runner.py`.

## Code Style

**Formatting:**

- No formatter configured: no `pyproject.toml`, `.editorconfig`, `.prettierrc`, `setup.cfg`, or `tox.ini` exist in the repo.
- PEP 8 conventions are followed loosely. Line lengths exceed 100 chars in places — longest run in prompt strings (`memory_optimizer/extraction.py` line 89: 174 chars; line 93: 200 chars) and `server.py` SSE yield (line 1114).
- Indentation: 4 spaces Python, no tabs. HTML/JS in `ui/index.html` uses 4-space indentation in `<script>` and 8-space for CSS rules.
- No semicolons in JS; single quotes for JS strings except template literals (backticks). Python uses double quotes for f-strings consistently (`f"...{var}..."`), single quotes for plain strings.

**Linting:**

- No linter config or tooling installed in `venv/` (checked: no ruff/flake8/black/mypy/isort).
- The codebase self-suppresses lint noise with inline `# noqa` comments where deferred imports follow a `sys.path.insert`: `# noqa: E402` (`experiments/paper.py` line 24, `experiments/paper_state.py` line 24, `run_experiments.py` line 24) and `# noqa: F401` for re-exports (`baselines/__init__.py` lines 7–12) and intentionally-unused imports (`experiments/paper.py` line 474).
- `from __future__ import annotations` is used in newer modules (`memory_optimizer/pipeline.py`, `experiments/paper.py`, `experiments/statistics.py`, `baselines/baseline_runner.py`, `experiments/e0_extraction_quality.py`, `experiments/e7_sensitivity_sweep.py`, `experiments/e8_external_benchmark.py`, `experiments/run_multi_model.py`, `experiments/qualitative_report.py`, `experiments/make_paper_artifacts.py`, `experiments/paper_state.py`, `data/build_gold_set.py`) but NOT in older ones (`memory_optimizer/scoring.py`, `memory_optimizer/extraction.py`, `memory_optimizer/budget.py`, `memory_optimizer/decay.py`, `memory_optimizer/retrieval.py`, `memory_optimizer/compression.py`, `experiments/live.py`, `server.py`, `data/synthetic_generator.py`). Existing string-based forward refs use quoted annotations: `-> "AdaptiveMemoryPipeline"` in `memory_optimizer/pipeline.py` line 59.

## Import Organization

**Order:**

1. Standard library modules first (alphabetically within): `import json, os, random, re, subprocess, sys, threading, time, uuid` (`server.py` lines 1–9); `from pathlib import Path` separated into a second stdlib group.
2. Third-party imports next: `import yaml`, `from fastapi import FastAPI`, `from pydantic import BaseModel, Field`, `import requests`, `import numpy as np`, `from scipy.stats import wilcoxon`, `from statsmodels.stats.power import TTestIndPower`.
3. Local package imports last: `from memory_optimizer.budget import token_budget_evict`, `from baselines.baseline_runner import BaselineRunner, fact_matches`, `from experiments.paper import load_settings, run_paper, write_manifest`.

**Path Aliases:**

- No `pyproject.toml`/`package.json` path aliases. Path resolution is via `Path(__file__).resolve().parent` rooted at `BASE_DIR` (`server.py` line 27, `run_experiments.py` line 21, every `experiments/*.py` script).
- Scripts not run as part of a package insert the repo root on `sys.path` first: `sys.path.insert(0, str(BASE_DIR))` (`run_experiments.py` line 22, `experiments/e0_extraction_quality.py` line 24). Use the same pattern in new standalone scripts.

## Error Handling

**Patterns:**

- Broad `try / except Exception` with graceful degradation to `None` or a default — pervasive for network I/O. Examples: `_measure_prompt_tokens` returns `None` on failure (`server.py` lines 121–133); `_lookup_or_measure` returns `None` (`server.py` line 185); `embed_ollama` returns `None` entries per text on failure (`memory_optimizer/embeddings.py` lines 53–54).
- Fallback state is *explicit and observable*, never silent: `FactExtractor.extract_facts` sets `meta["fallback"] = True` and returns `([], meta)` after two attempts (`memory_optimizer/extraction.py` lines 129–180), and the caller counts it: `job["extraction_fallback"] = ... + (1 if emeta.get("fallback") else 0)` (`server.py` line 889).
- Cache poisoning is explicitly protected against: an embedding failure does not write to `_cache` (`memory_optimizer/embeddings.py` lines 58–62).
- Timeouts are always set on outbound requests: `timeout=60` (`memory_optimizer/embeddings.py`, `experiments/paper.py` TokenMeasurer), `timeout=120` (`memory_optimizer/extraction.py`), `timeout=180` for LLM calls (`server.py` lines 323, 361, 912).
- `resp.raise_for_status()` is used only in `memory_optimizer/embeddings.py` line 51; most other call sites check `res.status_code != 200: continue` (`memory_optimizer/extraction.py` lines 149–150) or assume 200.
- `assert` is reserved for self-tests and gates only: `experiments/statistics.py::_selftest` (lines 118–133), `experiments/make_paper_artifacts.py::check` (lines 154–163), `baselines/baseline_runner.py::_quick_selftest`. Do not use `assert` for production validation; use fallback defaults.
- JSON sanitization at result boundaries: `experiments/paper.py::_sanitize` and `experiments/live.py::_sanitize` convert `NaN`/`inf` floats to `None` so manifests stay JSON-safe for `json.dumps` (duplicated in both files — keep either one if you touch it).
- Pydantic model validation for API request bodies: `ChatMessage` enforces `min_length=1, max_length=20000` (`server.py` lines 117–118); `SettingsPayload` uses optional fields with server-side clamping in the handler (`server.py` lines 1893–1920).

## Logging

**Framework:** None. The codebase uses `print()` for script output (`print(f"-> {out}")` in every experiment script) and a captured-subprocess log ring buffer for the research job runner (`job["log"]`, capped at `_RESEARCH_LOG_CAP = 500` lines, `server.py` lines 1211, 1415–1417). There is no `logging` module usage anywhere.

**Patterns:**

- Scripts print a `===...===` banner with run configuration at start (`run_experiments.py` lines 69–75), then a short result summary and the artifact path (`print(f"\n[SUCCESS] manifest {run_id} -> {path}")`).
- The server surfaces errors to the UI as `{"status": "error", "detail": ...}` or `{"status": "busy", "detail": ...}` responses rather than log lines (`server.py` lines 1101, 1157, 1540).
- Research job progress is derived from parsing the subprocess stdout for `seed N` markers: `_RE_SEED_MARKER = re.compile(r"^\s*seed \d+", re.I)` (`server.py` line 1210).

## Comments

**When to Comment:**

- Module docstrings carry the design rationale and research grounding; every module in `memory_optimizer/` and `baselines/` opens with one, citing brain.md sections and arXiv papers (e.g. `memory_optimizer/budget.py` cites arXiv:2607.22562, AAAI 2024 MemoryBank, MemGPT arXiv:2310.08560; `memory_optimizer/compression.py` cites "deduplicate, don't summarize" research).
- Inline comments explain *why*, not *what*: "A transient Ollama failure must not poison the process-wide cache" (`memory_optimizer/embeddings.py` line 58), "Extraction failure must not silently make the entire raw user message durable memory" (`memory_optimizer/extraction.py` line 177), "ranked=True is a read-only inspection view..." (`memory_optimizer/retrieval.py` line 158).
- Placeholder stubs carry an explicit warning not to "improve" them: `MemoryCompressor.compress_cluster` says "Concatenative placeholder — NOT a real summarizer... Do not 'improve' it into a model call" (`memory_optimizer/compression.py` lines 39–45).
- Deprecated modules explain the replacement: `experiments/e1_token_efficiency.py` returns `"status": "deprecated"` pointing at `run_experiments.py`.

**JSDoc/TSDoc:**

- Not used. HTML/JS in `ui/index.html` has a handful of `//` comments (e.g. `// ===================== Research / Pipeline Control ====================` line 1152, `// experiments table builder (shared by dashboard + detail page)` line 1409) and inline `title` attributes on inputs. No JSDoc blocks.

## Function Design

**Size:** Functions range from 1-line helpers (`_tok` in `memory_optimizer/pipeline.py`, `_dot` in `baselines/baseline_runner.py`) to large orchestrators (`server.py::_build_comparison` ~160 lines; `server.py::_ingest_turn` ~95 lines). Orchestrators are tolerated in `server.py`; the `memory_optimizer/` library keeps functions small and single-purpose.

**Parameters:** Optional collaborators and measurement functions are injected as keyword params with `None` defaults that trigger fallbacks — the core pattern for testability and for the paper pipeline: `fact_tokens=None, embed_fn=None` threaded through `AdaptiveMemoryPipeline.ingest` (`memory_optimizer/pipeline.py` lines 80–81), `BaselineRunner(..., embed_fn, fact_tokens, embedding_model)` (`baselines/baseline_runner.py` lines 167–179), `evaluate_method(..., fact_tokens=None, extraction_ms=None)` (`experiments/paper.py` lines 182–192). Private `__init__` params use underscore-prefixed keyword style in places (`MemoryRetriever.__init__(top_k, sim_threshold, embed_fn, ...)`).

**Return Values:**

- Consistent dict shapes with stable keys; metric bundles always carry provenance: `metric_source` (`experiments/paper.py` line 576, `experiments/live.py` line 368), `token_source` in `"measured"` vs `"estimated(word-count)"` (`baselines/baseline_runner.py` line 103, `experiments/paper.py` line 568).
- Tuples for multi-value returns: `token_budget_evict -> (kept, evicted, used)` (`memory_optimizer/budget.py` line 39), `fit_to_budget -> (selected, skipped, used)` (line 75), `extract_facts -> (parsed, meta)` (`memory_optimizer/extraction.py` line 119), `generate_conversation -> (turns, ground_truth)` (`data/synthetic_generator.py` line 27).
- Failure returns a documented empty/default rather than raising: `extract_facts` returns `([], meta)` with `meta["fallback"]=True`; `_llm_window_replay` returns a dict with `"error": str(e)` on exception (`server.py` lines 338–340).
- Metrics that cannot be computed return `None` (not `0.0`) when the value is unknown — e.g. `_recall_baseline_real -> None`, `wilcoxon_paired -> None` when `n < 2` (`experiments/statistics.py` line 76) — and `NaN`/`inf` are sanitized to `None` at manifest boundaries.

## Module Design

**Exports:** Packages re-export their public surface through `__init__.py`: `baselines/__init__.py` re-exports `BaseBaseline, BaselineRunner, fact_matches, matched_gt_keys` with an explicit `__all__` (lines 7–13); `memory_optimizer/__init__.py` is a one-line module docstring only (no re-exports — import modules directly, e.g. `from memory_optimizer.scoring import ImportanceScorer`).

**Barrel Files:** Only `baselines/__init__.py` acts as a barrel. `memory_optimizer`, `experiments`, `data` do not barrel their contents.

**Layering:** `memory_optimizer/` is the pure library (no HTTP server imports; only `requests`, `numpy`); `baselines/` depends only on `memory_optimizer/`; `experiments/` and `server.py` orchestrate both. `server.py` imports private helpers from the library when needed: `from memory_optimizer.retrieval import MemoryRetriever, _token_overlap` (`server.py` line 22), `from memory_optimizer.extraction import FactExtractor, build_extraction_prompt` (line 19).

**Single-source-of-truth modules:** The same semantics are deliberately shared between live server and offline replay — `token_budget_evict`/`fit_to_budget` ("the single source of truth", `memory_optimizer/budget.py` docstring), `AdaptiveMemoryPipeline` ("byte-for-byte the one measured in the paper", `memory_optimizer/pipeline.py` docstring). The live server passes its own state lists into the pipeline: `store=state["active_memories"], pruned=state["pruned_memories"]` (`server.py` lines 551–553).

## Other Conventions

**CLI pattern (experiments):** Every runnable script defines `main()` guarded by `if __name__ == "__main__":`, uses `argparse`, and supports a `--quick` flag that mutates defaults into a small deterministic LLM-free wiring gate (`run_experiments.py` lines 55–57, `experiments/e0_extraction_quality.py` lines 185–186, `experiments/e7_sensitivity_sweep.py` lines 107–108). Keep the `--quick` gate on any new experiment script; the server chains them in `RESEARCH_GATES` (`server.py` lines 1309–1316).

**Determinism:** Seeded RNGs everywhere: `random.seed(seed)` / `random.Random(seed)` in `data/synthetic_generator.py` line 24, `server.py::_build_demo_messages` line 751, `data/build_gold_set.py` line 34; seeds default to `42` in `config.yaml` (`system.seed: 42`) and the paper pipeline uses `seeds = [42 + i for i in range(n)]` (`run_experiments.py` line 59).

**Settings handling:** `config.yaml` is the single settings source, loaded with `yaml.safe_load` once at import in `server.py` (lines 53–54) and via `load_settings(overrides)` in `experiments/paper.py` (lines 521–542). Runtime changes are persisted back with `yaml.safe_dump(doc, f, sort_keys=False)` in `_persist_settings` (`server.py` lines 189–200). Use `sort_keys=False` when writing config to preserve key order.

**JSON artifact output:** Results are written as pretty-printed JSON with `indent=2` and `default=str` (e.g. `experiments/paper.py::write_manifest` lines 591–592, `experiments/e0_extraction_quality.py` line 194). Manifests are named `manifest_<sha8>.json` + a `latest_manifest.json` symlink-equivalent copy; never clobber per-model manifests (`experiments/run_multi_model.py` docs).

---

*Convention analysis: 2026-09-16*
