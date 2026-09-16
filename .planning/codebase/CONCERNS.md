---
last_mapped_commit: 58c99f010198b7e814e289d17df8cb7133f8ebdb
last_mapped_at: 2026-09-16
---
# Codebase Concerns

**Analysis Date:** 2026-09-16

## Tech Debt

### Duplicated overlap/matching implementations (5 variants, drifting thresholds)

**Issue:** Fact-to-fact and query-to-fact similarity is re-implemented independently in at least five places, each with its own threshold and semantics. A fix to one does not propagate.

**Files:**

- `memory_optimizer/retrieval.py:25` — `_token_overlap` (query→fact, word-set, substring fallback)
- `memory_optimizer/compression.py:4` — `_overlap` (copy of the same function)
- `experiments/live.py:17` — `_overlap` (third copy)
- `baselines/baseline_runner.py:26` — imports `_token_overlap` (the only consumer aligning with `retrieval.py`)
- `ui/index.html:269` — JS `overlap` (word-set only, no substring fallback)

**Impact:** Threshold drift is already observable: ground-truth recall uses `>= 0.7` (`experiments/live.py:32`, `baselines/baseline_runner.py:31`), baseline-real recall uses `>= 0.6` (`experiments/live.py:51`), dedupe uses `0.85` lexical / `0.90` semantic (`memory_optimizer/compression.py:109`), and the server's `_held`/`_text_match` use `SequenceMatcher` ratios `>= 0.7` / `>= 0.85` (`server.py:423`, `server.py:934`). The h2h grader in the dashboard can therefore disagree with E2's reported recall on the same memory store.

**Fix approach:** Add a single shared matcher (e.g., in `memory_optimizer/retrieval.py` or a new `memory_optimizer/matching.py`) with named modes (`gt_recall`, `dedupe`, `answer_verdict`) and make `compression.py`, `experiments/live.py`, `baselines/baseline_runner.py`, and the server import it. Keep the JS `overlap` in `ui/index.html` as the only intentionally separate copy (client-side), but mirror the same normalization rules.

### Unbounded in-memory caches

**Issue:** Three caches grow without eviction for the lifetime of the process.

**Files:**

- `server.py:114` — `_fact_token_cache` keyed by `(model, fact_text)`, never cleared except on model switch (`server.py:1883`)
- `memory_optimizer/embeddings.py:14` — `_cache` of 768-dim float vectors per unique `(endpoint, model, text)`; documented "no persistence," but also no cap — a 500-turn demo with re-mentions stores hundreds of vectors permanently
- `server.py:592` — `state["timeline"]` is capped at 10000, but `state["raw_history"]` (`server.py:623`) and `state["baseline_replay"]` (`server.py:661`) are uncapped; only the API response slices `[-200:]` (`server.py:1842`)

**Impact:** Slow memory growth over many demos; embedding vectors dominate (768 floats ≈ 6 KB each pre-float32). Unbounded `raw_history` also makes every `_baseline_window()` call (`server.py:249`) linearly more expensive.

**Fix approach:** Cap `raw_history`/`baseline_replay` (e.g., last 2000 turns like the timeline) or purge on `_reset_demo_state()`; add LRU caps to `_fact_token_cache` (e.g., 5000 entries) and `embeddings._cache` (e.g., 10k vectors), evicting oldest.

### `compress_cluster` concatenative placeholder

**Issue:** `MemoryCompressor.compress_cluster` (`memory_optimizer/compression.py:38-56`) is a documented stub that emits `"Summarized Context: <joined text>"`. It is never called on the live ingest path (`pipeline.py:105` calls `dedupe_incremental`), but remains public API.

**Impact:** Any future caller that uses `compress_cluster` instead of `dedupe_*` silently gets a degenerate "summary" that is worse than doing nothing — and the docstring explicitly warns not to convert it into an LLM call (compression drift).

**Fix approach:** Either delete `compress_cluster` and its `compressed_from` bookkeeping, or raise `NotImplementedError` at the top so misuse fails loudly instead of producing garbage data.

### Module-level mutable global state and a typo-adjacent formatting defect

**Issue:** `server.py` holds the entire application state in a module-level dict (`server.py:85-106`) that is mutated from multiple threads (demo thread, research thread, request handlers) and read from API handlers without any lock. Also, `server.py:98` — `"baseline_replay": [],` — is mis-indented at column 0 inside the `state` literal (valid Python, clearly unintended formatting; the neighboring keys are indented).

**Impact:** `RuntimeError: list changed size during iteration` and torn reads are possible when `/api/state` or `/api/timeline` reads `state["timeline"]`/`state["active_memories"]` while the demo thread appends (`server.py:574`, `server.py:623`). The formatting defect is cosmetic but signals the dict was hand-edited.

**Fix approach:** Guard reads with the existing `_research_lock`/`_chat_lock` (or a dedicated `_state_lock`) and iterate over copies (`list(...)`) in API paths. Fix the indentation at `server.py:98`.

### `config.yaml` rewritten as the settings persistence mechanism

**Issue:** Every `/api/settings` POST and model switch rewrites the repo's `config.yaml` in place (`server.py:189-200`), clobbering comments and key order (`sort_keys=False` keeps order but `yaml.safe_dump` drops comments). Two concurrent settings posts can interleave read/write and lose an update (no lock around `_persist_settings`).

**Impact:** Config drift and comment loss; a crashed write leaves a truncated `config.yaml` that fails `yaml.safe_load` at server startup (`server.py:53-54`).

**Fix approach:** Persist runtime settings to a separate file (e.g., `runtime_settings.yaml`) or a small JSON sidecar, and treat `config.yaml` as read-only defaults. Add a lock around the read-modify-write in `_persist_settings`.

### Workspace clutter at repo root

**Issue:** `server3.log` (215 lines of uvicorn access logs) sits in the repo root; `__pycache__/` exists at root and inside `data/`. All are gitignored (`*.log`, `__pycache__/` — `.gitignore:2,9`), so it is pure workspace pollution, but the log's constant `GET /api/timeline` + `GET /api/state` pairs show the dashboard polls these endpoints about once per second.

**Fix approach:** Move logs to `.planning/` or a `logs/` dir and add `logs/` to `.gitignore`; remove the stray `__pycache__/` dirs. Consider throttling the dashboard poll interval.

## Known Bugs

### E2 `baseline_forgetting_precision` is computed on the wrong turn set (live dashboard only)

**Issue:** `experiments/live.py:101` computes `baseline_forgetting_precision` as `_forgetting_precision(ground_truth, set(included_turn_ids) - set(range(0)), categories=("transient",))`. `set(range(0))` is the empty set, so the whole expression evaluates to `included_turn_ids` — the turns *still inside* the window. `_forgetting_precision` (`live.py:55`) then credits the baseline for every transient fact whose source turn is still in the window, and marks in-window transient facts as "correctly forgotten."

**Impact:** The live dashboard's E2 panel reports a baseline forgetting precision that is semantically inverted (it rewards keeping transient noise), and it disagrees with the offline paper path, which correctly computes the same metric over *evicted* turns: `experiments/paper.py:212-215` (`g["source_turn"] not in baseline_included_ids`).

**Fix approach:** Change `live.py:101` to compute over evicted turns: `set(gt_turns) - included_turn_ids` (mirror `paper.py:213`), then re-run `/api/experiments/run` and refresh `experiments/results/live_manifest.json`.

### h2h reference-memory grader uses coarse word-overlap matching (known flaw)

**Issue:** `ui/index.html:308-326` (`_h2hMatch`) decides whether the adaptive side "holds" a planted fact via:

- exact turn match (`f.turn === can.source_turn`), or
- `Math.min(overlap(a, b), overlap(b, a)) >= 0.7` where `overlap` is word-set Jaccard-over-query (`ui/index.html:269-276`).

**Impact (concrete failure modes):**

1. **Negation false-positive:** a store entry like `"no cat named Whiskers"` intersects `"User has a cat named Whiskers"` on 3 of 4 words → `min(0.75, 0.75) = 0.75 >= 0.7` → the grader verdicts the adaptive side as "still keeps the negated positive" (`ui/index.html:319`) even though the store holds the correct negated truth. Word-order and negation syntax are invisible to the matcher.
2. **Short facts inflate overlap:** the denominator is the smaller set (`min` over both directions), so short facts match at 0.7 easily — a two-word paraphrase of a four-word fact often clears the bar.
3. **Correction path:** after a correction, dedupe may merge the revised fact into the original memory entry (keeping the old `source_turn_id`, `memory_optimizer/compression.py:110-117`); the grader's turn check then fails and the text-overlap check must rescue it — borderline paraphrases flip the verdict row.
4. **Asymmetric baseline row:** negations are always displayed as "transcript keeps both wordings" for the baseline (`ui/index.html:340`) while the adaptive column is judged — the h2h table compares a judged score against an unjudged pass.

**Fix approach:** Replace word-set overlap with the same semantic matcher used in E2, or at minimum (a) strip negation markers (`not`, `no`, `never`, `isn't`) before comparing, (b) use a higher threshold (>= 0.85) plus containment for short facts, and (c) judge the baseline negation row with the same rule instead of skipping it. Verify against the live E2 per-category recall so the table and E2 agree.

### Baseline-real recall judges all ground truth against only the latest window replay

**Issue:** `experiments/live.py:45-52` (`_recall_baseline_real`) and `experiments/live.py:168` (`_e4`) consume `state["baseline_replay"][-1]` — the *last* per-turn replay only — and treat its `facts_seen` as the baseline's memory for the whole run.

**Impact:** When `replay=True` (demo mode), `baseline_positive_recall` and E4 baseline accuracy silently evaluate every planted fact (many turns old) against facts the model saw in the final window. Facts deliberately evicted from the final window are all counted as "forgotten" even if they were recalled at their own turn — the metric is a single-window snapshot dressed as per-turn coverage, and it disagrees with the offline `paper.py:202-205` semantics (per-turn window inclusion).

**Fix approach:** Either aggregate `facts_seen` across all replays (`baseline_replay` per-turn union), or compute baseline recall as `_recall_baseline` per turn (`live.py:38-42`) with replay used only as the tie-check, and label the metric source accordingly (`baseline_recall_real` currently claims `True` — `live.py:97`).

### `_gpu_snapshot` re-spawns `nvidia-smi` on every call when the GPU is unreachable

**Issue:** `server.py:203-217` caches only non-None results: `if now - _gpu_cache["ts"] < 2 and _gpu_cache["data"] is not None`. On a machine without `nvidia-smi` (or with `nvidia-smi` failing), `data` stays `None`, the cache is never served, and every `/api/state` poll (`_build_comparison` → `_vram_block` → `_gpu_snapshot`) forks a `subprocess.run` with a 3 s timeout.

**Impact:** On CPU-only dev boxes the dashboard's ~1 poll/sec (see `server3.log`) spawns ~1 nvidia-smi subprocess per second, each ~50-100 ms of overhead, in a request hot path.

**Fix approach:** Cache the None case too (e.g., cache `{"data": None}` for 10 s, or set a `_gpu_unavailable` flag after the first failure).

### Ollama `num_ctx` hard-coded below the selectable budget

**Issue:** All Ollama generate calls fix `num_ctx: 8192` (`server.py:129`, `server.py:322`, `server.py:361`, `server.py:911`; `memory_optimizer/extraction.py:141`), while `/api/settings` allows any `max_context_tokens` with only a lower clamp of 256 (`server.py:1905`) and the research pipelines advertise budgets up to 16384 (`server.py:1230`). `config.yaml` default is 4096.

**Impact:** A user raising the budget above ~7-8k tokens gets transcript replay prompts (`_llm_window_replay` builds the full window transcript, `server.py:312-313`) that exceed the Ollama context window; the model silently truncates, and E1's window token measurement (`prompt_eval_count`) no longer matches the configured budget — the comparison becomes invalid without any warning.

**Fix approach:** Clamp `max_context_tokens` to `min(16384, num_ctx - slack)` server-side, and derive `num_ctx` from the configured budget (e.g., `num_ctx = max(8192, budget + 2048)`) instead of a constant.

## Security Considerations

### No authentication on the dashboard — personal data exposure if bound beyond loopback

**Risk:** `server.py:1996-1998` defaults to `127.0.0.1`, which is safe, but `HOST` is env-overridable and `PORT` too. If anyone runs with `HOST=0.0.0.0` (documented pattern in the code, `server.py:1998`), every endpoint is unauthenticated: `/api/state` returns the full live memory store including `personal`-category facts (`server.py:1817-1844`), `/api/ethics/privacy_export` dumps all memories and pruned memories (`server.py:1961-1967`), and `/api/ethics/purge` destroys them (`server.py:1970-1987`).

**Files:** `server.py:1996`, `server.py:1817`, `server.py:1961`, `server.py:1970`

**Current mitigation:** Loopback bind default; CORS origin allow-list defaulted to `http://127.0.0.1:9002,http://localhost:9002` (`server.py:43-45`).

**Recommendations:** Refuse to serve outside loopback with a warning log unless an explicit opt-in env var (e.g., `ALLOW_NON_LOCAL=1`) is set; add a lightweight shared-secret header (`X-Dash-Token`) for non-loopback binds. Note the prompt-injection angle: LLM-extracted facts are rendered in the dashboard — see XSS below.

### Stored-XSS escape gaps in the dashboard

**Risk:** LLM-extracted fact text is consistently escaped (`esc()` at `ui/index.html:851`; used in `renderThisTurn`, `renderBothSides`, h2h rows, timeline). But three template slots interpolate server/LLM data unescaped:

- `ui/index.html:498-499` — model names from Ollama `/api/tags` go raw into `<option value="${m.name}">`
- `ui/index.html:814` — `v.active_model` raw into the VRAM chip
- `ui/index.html:1184` — pipeline `choices` rendered raw (self-controlled, low risk)

**Files:** `ui/index.html:498`, `ui/index.html:814`, `ui/index.html:1184`

**Current mitigation:** Everything else — facts, answers, prune reasons, eviction details — flows through `esc()` or `TL.esc`.

**Recommendations:** Wrap the three slots in `esc()`. Low urgency (loopback-only tool, model names come from the local Ollama), but it is the one place an adversarial model response could reach the DOM.

### Research runner spawns subprocesses on dashboard request

**Risk:** `/api/research/run` (`server.py:1536`) launches `subprocess.Popen([sys.executable] + argv, ...)` (`server.py:1439`) with CLI args built from caller-supplied params (`_build_args`, `server.py:1359`). Argv-list construction means no shell injection, and the pipeline/field schema (`RESEARCH_PIPELINES`, `server.py:1213`) constrains which flags exist — a caller cannot add arbitrary new flags beyond the schema's fields (unknown params are ignored by `_coerce_run`, `server.py:1319`). The remaining risk is a *trusted-client* assumption: anyone who can reach the port can launch heavy multi-seed LLM runs and pin the GPU for long stretches.

**Files:** `server.py:1536-1556`, `server.py:1439-1447`

**Current mitigation:** Loopback bind; single-job `research_busy` gate.

**Recommendations:** Keep loopback-only (see auth item); optionally add a `--dry-run` mode for untrusted callers. No shell=True anywhere in the repo (verified).

### No rate limiting or CSRF protection

**Risk:** The API has no throttling; `/api/chat` accepts up to 20000 chars (`server.py:117-118`) and each call LLC-runs extraction + streamed answer. There are no cookies, so CSRF is not practically exploitable from a different origin (CORS blocks it, and content-type is restricted to `Content-Type` only, `server.py:50`).

**Files:** `server.py:50`, `server.py:117`, `server.py:1071`

**Current mitigation:** Loopback bind + CORS allow-list.

**Recommendations:** Acceptable for a local research tool; document that it is not safe to expose.

## Performance Bottlenecks

### E5 ablation run executes 4 full pipeline replays inside one HTTP request, unguarded

**Problem:** `/api/experiments/run` (`server.py:1931-1958`) calls `compute_all_json` synchronously; `_e5` (`experiments/live.py:279-307`) replays the (sampled, up to 800-turn) stream **four times** — full, no-decay, no-compression, equal-weights — each replay re-embedding every fact and running retrieval per turn (`_e5_replay`, `live.py:215-276`). When `measured` tokens are on, every fact measurement is a separate Ollama call (`server.py:121`).

**Cause:** `compute_all` → `_e5` is pure and synchronous; the embedding cache (`embeddings.py`) absorbs repeat embeddings, but the 4× replay loop is still O(4 × turns × store) with per-turn embedding calls for new facts, plus `injection_token_limit` gated re-measurement.

**Improvement path:** Cache or dedupe the per-turn facts across the four replays (they share the same input stream and differ only in toggles) — e.g., precompute `scored`/embeddings once and replay only the policy deltas; also add a `state["research_busy"]`-style guard and cancellation checkpoints so a stray run can be cancelled.

### `/api/experiments/run` bypasses the single-GPU guard

**Problem:** The only guards in `run_experiments()` are `demo_job.running` and `raw_history` emptiness (`server.py:1935-1938`). It does **not** check `chat_busy`, does **not** set `research_busy`, and does **not** take `_research_lock`. A user can click "Run Experiments" while a chat is streaming and both hammer the same Ollama GPU, halving throughput and inflating E3 latency numbers recorded for the chat turn.

**Files:** `server.py:1931-1958`, `server.py:937-944` (guard helpers)

**Improvement path:** Route `run_experiments` through the same begin/end guard used by `research_run` (`server.py:1541-1556`), and have it return 409 busy like the chat path.

### `keep_alive: "30m"` pins models in VRAM

**Problem:** Every Ollama call sets `keep_alive: "30m"` (`server.py:128`, `server.py:321`, `server.py:360`, `server.py:910`, `memory_optimizer/extraction.py:140`). After a 150-turn demo, the model stays resident for 30 minutes; switching models (`server.py:1873-1879`) explicitly unloads the old one, but model-vs-embedding models both stay loaded.

**Cause:** Chosen to avoid per-call model reload latency; there is no VRAM pressure feedback.

**Improvement path:** Lower to `keep_alive: "5m"` or make it configurable in `config.yaml`; the 2-second-cached `_gpu_snapshot` could also refuse to launch new demos when VRAM headroom is below the estimated KV footprint (`_vram_block`, `server.py:233`).

### Unbounded embedding cache (memory)

**Problem:** `memory_optimizer/embeddings.py:14` caches every unique `(endpoint, model, text)` vector forever; 768-dim float32 vectors ≈ 3 KB each. The E5 replays alone embed each sampled fact once per replay variant path (cache hit after first), so the working set stays bounded by unique facts, but across many demos/messages the cache never shrinks.

**Improvement path:** LRU cap (e.g., 20k vectors ≈ 60 MB) with `collections.OrderedDict` or a simple `maxsize` check in `embed_ollama`.

## Fragile Areas

### Thread-safety of the global `state` dict

**Files:** `server.py:85-106` (state), `server.py:872-899` (`_run_llm_demo` thread), `server.py:1420-1509` (`_run_research_job` thread), `server.py:1817-1844` (reads)

**Why fragile:** The demo thread appends to `raw_history`, `timeline`, `active_memories`, `baseline_replay` while request threads iterate them. The `_research_lock`/`_chat_lock` only guard the *busy flags*, not the collections. A demo + concurrent `/api/state` poll (the dashboard polls continuously — see `server3.log`) can race: `for m in state["active_memories"]` during `_ingest_turn`'s `step_decay_and_prune` (`pipeline.py:113` reassigns `self.memories = active`) — list reassignment is atomic, but `state["budget_evicted_memories"].append` (`server.py:615`) and timeline append are not safe to iterate concurrently.

**Safe modification:** Access state collections only under a new `_state_lock` in API handlers (or iterate `list(...)` copies); keep mutations inside the demo/research threads' own critical sections.

**Test coverage:** None — see Test Coverage Gaps.

### Recall semantics depend on matcher choice for corrected/negated facts

**Files:** `experiments/live.py:32`, `baselines/baseline_runner.py:38-42`, `ui/index.html:321`, `server.py:419-425`

**Why fragile:** Recall for hard cases is decided by `source_turn` match OR text overlap. After a correction, the revised fact is anchored to a *new* source turn, and the superseded original may still match via overlap — so "wrongly_retained_after_correction" (`live.py:117-125`) hinges entirely on the 0.7 overlap threshold and dedupe behavior (`compression.py` merges at 0.85-0.90). Changing the dedupe threshold silently changes E2 hard-case figures without touching E2 code.

**Safe modification:** When changing `compression.py` thresholds, re-run E0 quick gate (`experiments/e0_extraction_quality.py --quick`), E2 (`run_experiments.py --quick`), and the dashboard's h2h table on the same scenario, and confirm E2 hard-case and h2h verdicts move together.

### Demo ground-truth/supersede bookkeeping

**Files:** `server.py:742-852` (`_build_demo_messages`), `ui/index.html:280-306` (`gtChains`)

**Why fragile:** Correction and negation turns are placed into `free` slots (`server.py:805-807`) obtained by excluding planted/density turns; if `n_corrections + n_negations` approaches `len(free)` or the last free slot equals `num_turns`, the code silently drops cases (`if ct is None: continue`, `server.py:820`). The `gtChains` reconstruction in the UI re-derives chains from `supersedes_turn`/`is_negation` flags and can disagree with the server's `ground_truth` when `planted_count` collides with `positions[-1] == num_turns` handling (`server.py:763-764`).

**Safe modification:** Add an assertion in `_build_demo_messages` that the requested `n_corrections`/`n_negations` were actually placed, and surface a warning in `ground_truth` if any were dropped; keep the UI chain logic in sync with any `ground_truth` field renames (it currently reads `is_correction_target` but never uses `obsolete`).

### `experiments/results/` is gitignored but is the server's write target

**Files:** `.gitignore:5` (`experiments/results/`), `server.py:1500-1504` (jobs history write), `server.py:1953-1957` (live manifest write), `server.py:1771-1784` (board reads)

**Why fragile:** The research board (`research_board`) silently returns `None` for each missing artifact via `_safe_read_json` (`server.py:1598`), so a deleted or partially-written `latest_manifest.json` renders as an empty board — and the `JOBS_HISTORY_FILE.write_text` failure is swallowed (`except Exception: pass`, `server.py:1503-1504`). If someone `git clean`s the results dir while the server runs, history silently stops persisting.

**Safe modification:** Persist `jobs_history.json` under a non-gitignored path (e.g., `.planning/`) and log (not swallow) write failures.

### Extraction failure silently empties a turn

**Files:** `memory_optimizer/extraction.py:173-180`, `server.py:889` (demo tally), `server.py:974` (chat)

**Why fragile:** `extract_facts` retries twice, then returns `([], meta)` with `meta["fallback"] = True`. If the local llama3.1:8b unloads or the prompt format drifts, a whole demo turn stores nothing — the demo still counts it as processed (`turns_done`), and the raw message is still appended to `raw_history` (`server.py:623`) — so the adaptive side's recall can quietly degrade between two runs with no error surfaced beyond a `fallback` counter the UI does not display.

**Test coverage:** The quick gates (e0 `--no-llm`, `run_experiments.py --quick`) all run oracle writes, so LLM-extraction failure paths are never exercised in CI-style checks.

## Scaling Limits

**Single-GPU concurrency:** The exclusivity guard is in-process only: `_chat_guard` (`server.py:937`) + `_research_lock` (`server.py:1155`, `server.py:1424`, `server.py:1541`) + demo checks. It does not cover `/api/experiments/run` (see Performance), and it cannot stop a second `uvicorn` worker/process, a CLI `run_experiments.py` invocation, or an unrelated Ollama client from sharing the GPU. If this dashboard ever runs behind `--workers > 1`, the in-memory state itself splits across workers (each worker would have its own empty `state`) — the current design is single-process only.

**Ollama context cap:** All requests use `num_ctx: 8192` while budgets up to 16384 are selectable (see Known Bugs) — beyond ~8k tokens the transcript truncates silently.

**Turn counts:** `_baseline_window` is O(n) per call and called per API poll and per demo turn; at 500-turn demos this is fine, but uncapped `raw_history` accumulation across repeated demos makes the board and `/api/comparison/full` (`server.py:1801`) increasingly slow.

## Dependencies at Risk

**`bert-score` is a dead dependency:** listed in `requirements.txt:10` but never imported anywhere in the repo (`e0_extraction_quality.py`'s docstring mentions BERTScore at `experiments/e0_extraction_quality.py:5` but the code only uses `rouge_score` at line 41). It pulls in `torch`-family wheels indirectly on some platforms — significant install weight for nothing. Remove it or actually use it in the paraphrase scorer.

**Exact pins, no Dependabot style tracking:** `requirements.txt` pins exact versions (`fastapi==0.141.1`, `numpy==2.5.3`, `scipy==1.18.1`, `statsmodels==0.15.0`, `requests==2.34.2`, `scikit-learn==1.9.0`, `pydantic==2.13.5`). Deterministic installs, but there is no upgrade cadence; `statsmodels` and `scipy` are only used for `TTestIndPower` (`experiments/live.py:320`) and bootstrap routines (`experiments/statistics.py`) — if those APIs deprecate, the E6 power computation breaks silently (`except Exception: req_n = None`, `live.py:322`).

**Ollama is an implicit runtime dependency:** everything LLM/embedding goes to `http://localhost:11434` (agent default from `config.yaml:3`). No health check or fallback beyond per-call `except Exception` paths that return `None`/empty — a stopped Ollama makes the dashboard answer every chat with `NO_LOCAL_ANSWER` (empty context) and report `embed_ms` spikes.

## Missing Critical Features

**No test suite at all:** the repo has zero `test_*.py`/`*_test.py` files. The only "tests" are self-checks: `experiments/statistics.py:_selftest` (`statistics.py:111`), `baselines/baseline_runner._quick_selftest` (`baseline_runner.py:261`), the E0 `--quick` oracle gate (`e0_extraction_quality.py:185-186`), and `run_experiments.py --quick`. None of these exercise `server.py` endpoints, the h2h grader (`ui/index.html`), or the decay/budget/dedupe edge cases (boundary thresholds 0.2/0.35/0.6/0.7/0.85/0.9, empty store, all-transient pruning, budget exactly at a fact size).

**Blocks:** metric regressions like the `live.py:101` inversion and the `_h2hMatch` negation false-positive ship unnoticed because nothing asserts E2/h2h consistency.

**No server-side validation errors surfaced to the UI for research jobs:** job failures are recorded in `job["error"]` and the log tail, but the board (`research_board`) returns no error summary; users must open the job log to notice a run died (e.g., Ollama offline mid-run).

## Test Coverage Gaps

**Untested area:** `memory_optimizer/` core (decay math, budget eviction tie-breaks, dedupe merge semantics, retrieval ranking + `_bump` reinforcement), `server.py` API and concurrency guards, and the dashboard h2h grader.

**Files:** `memory_optimizer/decay.py`, `memory_optimizer/budget.py`, `memory_optimizer/compression.py`, `memory_optimizer/retrieval.py`, `server.py`, `ui/index.html:308`

**Risk:** Silent metric corruption (already present in `live.py:101`), XSS regression in the four escape-gap slots, and race-condition crashes during demo+chat concurrency.

**Priority:** High — the `live.py:101` bug and h2h grader flaws both fall in this gap and both currently ship in the dashboard's headline E2/h2h panels.

---

*Concerns audit: 2026-09-16*
