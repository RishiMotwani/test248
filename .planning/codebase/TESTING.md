---
last_mapped_commit: 58c99f010198b7e814e289d17df8cb7133f8ebdb
last_mapped_at: 2026-09-16
---
# Testing Patterns

**Analysis Date:** 2026-09-16

## Test Framework

**Runner:**

- None. There is **no unit test framework** in this repo: no `pytest`/`unittest` usage, no `pytest.ini`, `pyproject.toml`, `setup.cfg`, `tox.ini`, or `flake8` config. The project venv at `venv/` contains no pytest/tox/coverage binaries (verified via `venv/bin/pip list`).
- `.gitignore` lists `.pytest_cache/` (line 4), indicating pytest was run at some point, but no test files remain and no test directory exists (`glob **/*test*` and `**/tests/**` return nothing).

**Assertion Library:**

- Plain builtin `assert` statements, used only inside self-test gates described below.

**Run Commands:**

```bash
./venv/bin/python run_experiments.py --quick          # Fast wiring gate across the whole pipeline stack
./venv/bin/python experiments/statistics.py            # Statistics helpers self-test (_selftest)
./venv/bin/python baselines/baseline_runner.py         # Baseline infrastructure quick self-test
./venv/bin/python experiments/make_paper_artifacts.py --check   # Assert-based anti-drift artifact check
./venv/bin/python server.py                            # The live server — no startup tests
```

## Test File Organization

**Location:**

- No dedicated test files or test directories exist anywhere in the repo (`tests/`, `test_*.py`, `*_test.py`, `*.spec.*` all absent).
- Verification is embedded in the scripts themselves as `if __name__ == "__main__":` self-test blocks, and in the experiment CLI via `--quick`.

**Naming:**

- Self-test functions: `_selftest()` (`experiments/statistics.py` lines 118–133), `_quick_selftest()` (`baselines/baseline_runner.py` lines 267–288).
- Gate-check function: `check()` (`experiments/make_paper_artifacts.py` lines 154–163).

**Structure:**

```
[project-root]/
├── run_experiments.py              # --quick gate (50 turns, 1 seed, oracle, no measured tokens)
├── experiments/
│   ├── statistics.py               # _selftest() under __main__ guard
│   ├── make_paper_artifacts.py     # --check gate (reload artifacts, assert numbers match)
│   ├── e0..e8_*.py                 # experiment scripts, each with a --quick wiring gate
└── baselines/
    └── baseline_runner.py          # _quick_selftest() under __main__ guard
```

## Test Structure

**Suite Organization:**
There is no suite organization — no test discovery, no fixtures, no setup/teardown. Verification exists as three complementary mechanisms:

**1. `--quick` gate (every experiment script).** Argparse `--quick` collapses each run into 50 turns / 1 seed, oracle writer, no measured tokens — a deterministic, LLM-free smoke test of the wiring:

```python

# run_experiments.py (lines 55–57)

parser.add_argument("--quick", action="store_true",
                    help="fast deterministic wiring gate (50 turns, 1 seed, oracle, no measured tokens)")
```

The server chains all quick gates into one job and treats the chain result as its sanity check — `RESEARCH_GATES` (`server.py` lines 1309–1316) runs `e0 --quick`, `e1/e2 --quick`, `e7 --quick`, `e8 --quick`, etc. sequentially.

**2. `_selftest()` inside the module.**

```python

# experiments/statistics.py (lines 118–133)

def _selftest():
    assert bootstrap_ci([1.0]*10) == (1.0, 1.0, 1.0)
    assert wilcoxon_paired([1.0]*5, [2.0]*5)[0] == 1.0          # degenerate perfect
    assert wilcoxon_paired([3.0]*3, [1.0]*3) is None            # n < 2 conservatism
    ...
if __name__ == "__main__":
    _selftest()
    print("statistics selftest OK")
```

Run directly: `./venv/bin/python experiments/statistics.py`.

**3. Assert-based artifact anti-drift check.** `make_paper_artifacts.py --check` reopens the emitted CSV/JSON artifacts and asserts the numbers match the manifest — a regression guard against hand-edited or regenerated artifacts:

```python

# experiments/make_paper_artifacts.py (lines 154–163)

def check(lp):
    files = sorted(Path("experiments/results").glob("manifest_*.json"))
    assert len(files) >= 1, "no manifest found"
    ...
    for row in rows:
        assert row[0] == m["methods"][row[1]]["mean_E1_proposed_tokens"], "manifest drift"
```

## Mocking

**Framework:** None (no `unittest.mock`, no pytest monkeypatch, no `responses`/`vcr`).

**Patterns:**

- **Dependency injection instead of mocking.** The library takes collaborators as optional keyword params that default to live implementations, and callers substitute deterministic stubs:

```python

# experiments/paper.py evaluate_method(..., fact_tokens=None, extraction_ms=None)

# baseline_runner.py

runner = BaselineRunner(..., embed_fn=None, fact_tokens=None, embedding_model=model)
```

- **Oracle writer as a fake LLM.** `--no-llm` mode replaces the LLM writer with a deterministic function that writes ground-truth facts, and E0 checks it scores 1.0:

```python

# experiments/e0_extraction_quality.py (lines 148–180)

if args.no_llm:
    facts = [gt["text"] for gt in turn.get("ground_truth", [])]   # oracle
```

- **Injected token measurer.** `TokenMeasurer(fact_tokens, rate=TOKES_PER_CHAR)` in `experiments/paper.py` wraps a deterministic counting function with an optional measured overhead — requires no network in `--quick` runs.

**What to Mock:**

- Any external dependency in a verification gate: Ollama `embed_ollama` calls, LLM done replies, live token measurement. Avoid real network calls in `--quick` paths — the codebase design already routes these through `fact_tokens`/`embed_fn` params; use them.

**What NOT to Mock:**

- The memory algorithms themselves. `token_budget_evict`, `fit_to_budget`, decay, retrieval, and compression are meant to be exercised for real — the offline replay suite `experiments/paper.py::replay_adaptive` runs the actual `AdaptiveMemoryPipeline` and `BaselineRunner` code paths end-to-end with deterministic synthetic data.

## Fixtures and Factories

**Test Data:**

- Synthetic conversation generation is the fixture factory: `SyntheticConversationGenerator(seed, tech_words=..., project_phrases=...)` in `data/synthetic_generator.py` produces (turns, ground_truth) pairs with configurable decoding-challenge ratios and fact injection rates:

```python

# data/synthetic_generator.py (lines 24–30)

rng = random.Random(seed if seed is not None else MODULE_SEED)
def generate_conversation(seed=MODULE_SEED, turns=60, ...) -> (turns, ground_truth)
```

- `data/build_gold_set.py` builds the E2 ground-truth gold set of elicitation statements, seeded and deterministic.

**Location:**

- Fixture-generating code lives in `data/`; fixture *data* is emitted to `experiments/results/` (`manifest_*.json`, `latest_manifest.json`, `live_manifest.json`, `board.json`, `gold.json`) by the experiment scripts. E7 grid sweeps determinize over `BASELINE_POINT` and per-dimension arrays in `experiments/e7_sensitivity_sweep.py`.

## Coverage

**Requirements:** None enforced — no coverage tool installed, no thresholds anywhere.

**View Coverage:**

```bash

# no coverage tooling exists; nothing to run

```

## Test Types

**Unit Tests:**

- None for `memory_optimizer/*.py` (scoring, decay, budget, retrieval, compression, extraction), `server.py`, or `ui/index.html`.
- The closest thing to unit tests are the two `__main__`-guarded self-tests (`experiments/statistics.py::_selftest`, `baselines/baseline_runner.py::_quick_selftest`), which cover only statistics helper math and baseline matching helpers.

**Integration Tests:**

- The experiment suite is the de facto integration harness. `experiments/paper.py` replays the full adaptive pipeline vs baselines with deterministic conversation streams, computes paired Wilcoxon (`experiments/statistics.py::wilcoxon_paired`), Cohen's d, bootstrap 95% CIs, and multiple-comparison corrections (Holm/Bonferroni) — then writes manifests. E5 ablations turn off pipeline components to verify each one matters; E8 runs a deterministic needled-QA benchmark; E6 checks statistical power with `statsmodels.stats.power.TTestIndPower`.
- Verification is **statistical** (are results significant?) rather than behavioral (does the code return the right thing?).

**E2E Tests:**

- None for the FastAPI endpoints. The server (`server.py`, 1998 lines) is only smoke-tested by manual browser use of `ui/index.html` — no TestClient, no httpx tests, no Playwright/Selenium.
- The 1460-line `ui/index.html` has no UI tests at all.

## Common Patterns

**Async Testing:**

- Not applicable — no async test tooling. The server is synchronous FastAPI (def, not async def, throughout `server.py`); the UI polls with `setInterval`/`clearInterval` and `fetch(...).then()` but has no tests.

**Error Testing:**

- The codebase's error *handling* is the test: fallback semantics are asserted via wiring gates and offline replays only. Examples of the contract you should preserve in any new verification:

```python

# memory_optimizer/extraction.py — fallback returns [] + meta["fallback"]=True (2 attempts then give up)

# memory_optimizer/embeddings.py — failed embed returns None entries, does NOT poison _cache

# experiments/make_paper_artifacts.py --check — asserts artifact/manifest equality

```

- No tests exercise error paths (e.g., server down, Ollama down, timeout expiry) — that is the biggest gap.

## Test Coverage Gaps

| Gap | Where | Risk | Priority |
|-----|-------|------|----------|
| No unit tests at all for `memory_optimizer/` algorithms | `memory_optimizer/{scoring,decay,budget,retrieval,compression,extraction}.py` | Token-budget fitting, decay math, and dedupe merge logic change without regression protection | **High** |
| No API tests for the FastAPI server | `server.py` (1998 lines: chat, demo, settings, research job runner, privacy purge) | Endpoint regressions undetected; `RESEARCH_GATES` chain breaks silently | **High** |
| No error-path tests | network timeouts, Ollama down, disk-full | Fallback semantics (`meta["fallback"]`, None returns) unverified | Medium |
| No UI tests | `ui/index.html` | Polling/SSE rendering regressions manual-only | Medium |
| No determinism regression guard | replay suite, `synthetic_generator.py` | A seed change silently shifts all E1–E8 numbers | Medium |

**Recommended path for new tests:** add pytest + FastAPI `TestClient`; cover `memory_optimizer` units first (pure functions, inject `fact_tokens`/`embed_fn` stubs exactly as the paper pipeline does), then `server.py` endpoints with `TestClient(app)` and patched research-command execution, never hitting real Ollama in tests.

---

*Testing analysis: 2026-09-16*
