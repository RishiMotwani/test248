# Adaptive Memory Manager — Research Brain

Living document. Every file edit after a research session should trace back to a
numbered design decision (D1, D2 …) here, with the citation that motivated it.
Nothing goes into `memory_optimizer/` or `server.py` without first landing here.

## 0. The one-sentence takeaway

Across ~15 papers, the single most consistent empirical result is:

> **Retrieval quality matters far more than how you write facts.** The gap
> between a weak and strong retriever (14–23 accuracy points) dwarfs the gap
> between a good writer (LLM fact extraction) and a naive writer (raw chunks,
> 3–8 points). Constructed/extracted memory is *not* inherently better than
> direct retrieval over raw turns — and contextual compression ("summarizing
> memory") often *hurts* by silently drifting.

Our current system inverts this priority: it spends all its complexity on
extraction/scoring/deduping and none on retrieval (lexical `_token_overlap` in
`memory_optimizer/retrieval.py:5`). The highest-leverage change is therefore the
shift from lexical to real embedding retrieval — exactly the follow-up we
planned.

---

## 1. Big picture / framework papers

### 1.1 MemGPT — "LLMs as Operating Systems"
- Packer et al., "MemGPT: Towards LLMs as Operating Systems", arXiv:2310.08560. (search: par.nsf.gov/servlets/purl/10524107)
- Virtual context management as OS paging: a **main context** window plus
  external memory tiers (**recall database** for recent conversation, **archival
  vector store** for facts), plus a **search function** the model itself calls.
- Model issues *function calls* to its own memory manager: `memory_insert`,
  `memory_search`, `cursor_search`, etc. Governance of what to save is explicit
  "self-editing" by the agent, not implicit.
- Eviction: FIFO queue + a **warning token count** — the model is told "you are
  approaching context limit, must evict" and chooses what to drop.
- Perf: beats fixed-context on document analysis + multi-session chat; ~10× cost
  savings; the *utility* (roughly our adaptive-vs-baseline measurement) was
  measured both subjectively and via DMR (Deep Memory Retrieval) benchmark.
- **For us**: our `top_k:5` fixed injection is MemGPT's shallow end. The lesson
  is that memory management should be *adaptive to context pressure*. Our
  min(20,500)-turn demo + 4096 `max_context_tokens` is already a MemGPT-style
  paging arena; we lack the "warning count / evict-under-pressure" mechanic on
  the adaptive path (see D1/D2).

### 1.2 "Memory for Autonomous LLM Agents: Mechanisms, Evaluation, Emerging Frontiers" — arXiv:2603.07670
- Organizing framework: **write → manage → read** loop, with memory taxonomy by
  temporal scope, representational substrate, control policy.
- Five mechanism families: context-resident compression; retrieval-augmented
  stores; reflective self-improvement; hierarchical virtual context (MemGPT);
  policy-learned management (RL).
- Repeated finding: **"gap between has-memory / no-memory > gap between
  backbones"** — memory is the dominant driver of agent quality, larger than the
  model itself.
- Key pathologies:
  - **Summarization drift**: context-resident compression (rolling summaries)
    silently discards low-frequency facts; errors compound turn over turn.
    RET-LLM & A-Mem respond with extractive atomic facts / triplets instead.
  - Debugging difficulty: when a memory system fails, you cannot tell whether
    the write path, retrieval, compression, or down-stream reasoning caused it.
    *Our `metric_source` flags + live-real-token E-values are a partial answer
    (we can attribute failures to E1 retention vs E3 cost vs E5 reuse).*
- Retrieved memories mix **recency (exponential decay) + relevance**; solid
  baseline, matches our decay formulation.

### 1.3 Cognitive-architecture framing (LightMem, Sumers et al. survey)
- LightMem (ICLR 2026; zjunlp/LightMem) is the Atkinson–Shiffrin model made
  concrete: **sensory** (cheap fast filter) → **short-term** (topic grouping +
  consolidation) → **long-term** (offline "sleep-time" consolidation that decouples
  the expensive writes from the online path).
- Reports accuracy gains up to 10.9% on LongMemEval while cutting token *usage*
  up to 117× and API calls up to 159× (they "summarize once, look up sorted" so
  recall is cheap) — but see 1.7: its substantive win is contested.
- **For us**: our ingest-time extraction + dedupe is the "online" write, and we
  have no offline consolidation pass. A cheap consolidation target = the decay
  prune + re-embed pass that happens on a *replay* timer, not inline.

---

## 2. Priority finding: retrieval > write, and constructed memory ≠ better

### 2.1 "Diagnosing the Retrieval vs. Utilization Bottleneck in LLM-based Generative Memory Systems" — arXiv:2603.02473
- Ablation over write strategy × retrieval method × downstream reasoning.
- **Retrieval method drives 14–23 accuracy points; write strategy contributes
  3–8 points.**
- **Basic RAG over raw chunks ≥ Mem0-style extraction and ≥ MemGPT rolling
  summaries** across retrieval methods. Simpler "writer" loses little or
  nothing.
- Cosine / BM25 / hybrid distinction matters more than whether your write step
  used an LLM.
- **For us**: our entire ingest pipeline (LLM extraction, scoring, dedupe,
  keyword categories) is optimizing the *low-leverage* half. Budget real effort:
  swap `_token_overlap` for real embeddings first.

### 2.2 "Reproducing LightMem: Naive RAG Is Just as Good for Memory Management" — arXiv:2607.29104
- Matched retrieval depths, token budgets, oracle eval. Result: **constructed
  memories (LightMem MemBank entries) don't beat direct retrieval over raw
  dialogue turns** ("Naive RAG").
- Retriever choice is huge: same store, Recall@10 = 0.390 (BM25) vs 0.587
  (Qwen3-Embedding-4B); answer accuracy 58.1% → 75.5%.
- LightMem = a *context-efficiency* trade, not an accuracy win.
- **For us**: our `baseline_replay` (raw window) vs `adaptive_replay` (fact
  store) is exactly this contest, and in our one real run adaptive won on
  accuracy at ~71 vs ~496 tokens (D1 baseline) — but our retrieval is the weak
  BM25-equivalent, so we have headroom to keep the win *and* the small context.

### 2.3 Agent memory reranking evidence (MemReranker — arXiv:2605.06132; Zep — arXiv:2501.13956)
- MemReranker: retrieval quality is "the bottleneck of agent memory"; a small
  (0.6B) *reasoning-aware reranker* matches GPT-4o-mini / Gemini-3-Flash on
  LoCoMo at ~200ms. Reranking = second pass over vector candidates.
- Zep: hybrid recall via cosine + BM25 + graph BFS, then rerank (RRF, MMR,
  cross-encoder, and a graph "episode-mentions" reranker that boosts whatever
  was *frequently mentioned* — i.e., reinforcement from use).
- **For us**: `nomic-embed-text` is already installed in Ollama (verified
  `GET /api/tags`). Target retrieval pipeline = embedding cosine candidate
  gen → rerank within candidates by our existing importance blend. Zep's
  frequency-boost validates our "re-mention reinforcement" idea (D4).

---

## 3. Compression and summarization: mostly a trap

### 3.1 Summarization drift (survey 2603.07670; MemGPT companion findings)
- Rolling/contextual summarization loses low-frequency facts and compounds
  error. RET-LLM, A-Mem, Mem0 all choose **atomic self-contained facts +
  embeddings** over summaries.
- **For us**: our `compress_cluster` (memory_optimizer/compression.py:16) is a
  *fake* summarizer — it concatenates facts under a "Summarized Context:"
  prefix and never calls a model. Research says: (a) don't pretend to summarize,
  (b) real LLM summarization is the highest-risk maneuver in the whole memory
  stack, (c) the honest alternative is harder dedupe + entity linkage, not
  compression. **Recommendation: keep compress_cluster concatenative at most,
  or better, make dedupe the consolidation mechanism and delete the
  Summarized-Context block** (see D6). A "real" model summarizer is a
  *risky* feature, not a safe upgrade, on 8GB VRAM.

### 3.2 Human-inspired alternatives to summarization
- **Human-Inspired Memory Models** (arXiv:2605.08538): sleep-phase
  consolidation, interference-based forgetting, **engram maturation** (new
  memories start as latent "silent" traces at activation_strength 0.0 and only
  become recallable after repeated reinforcing retrieval) — a different, nicer
  model than our one-shot `base_score`. Reconsolidation: *retrieval rewrites the
  memory*. Entity knowledge graphs + hybrid multi-cue retrieval (episodic vector
  + semantic KG).
- **NEMORI / "What deserves memory"** (ACL 2026): distil on **prediction error**
  — retain what the agent *fails to predict* from existing knowledge; "what is
  predictable is redundant." Training-free. Critique: heuristic importance tags,
  emotional tags, rigid factual templates are all wrong signals.
- **Mem0** (Chhikara et al., "VanillaRAG/RAG2"): facts extracted as
  self-contained, embedded, and **conflict-resolved (add / update / noop)** —
  a memory write is a semantic *merge*, not an append. We have token-overlap
  dedupe but not semantic conflict resolution (full dedupe happens only within
  turn; `dedupe_incremental` compares to the store but with lexical overlap).
- **For us** (D4-D6): replace one-shot-save with: engram-strength activation on
  re-mention (memory is born latent, grows with reinforcement, decays without);
  count retrieval-reinforcement (Zep episode mentions + Oblivion); treat dedupe
  as the compression story. Our `access_count` field exists but nothing uses it
  for score — that's the hook.

---

## 4. Forgetting: static decay is the weakest part

### 4.1 SF-AMS — "Strategic Forgetting for Structured Memory in LLM Agent" — arXiv:2607.22562
- Replaces static retrieval + **heuristic decay** with a **utility-driven
  survival** mechanism: the importance of a memory is *updated online from usage
  redundancy + temporal access signals*, not assigned once at write.
- Composite Importance Scoring (semantic + entity). Outperforms LightMem, Mem0,
  A-Mem on LoCoMo / LongMemEval-s; largest gain on multi-hop under
  Qwen2.5-7B (+9.65 F1).
- **For us**: our decay (memory_optimizer/decay.py: `M(t) = base_score·e^(−λΔ)`)
  is a *static* curve, exactly the thing SF-AMS replaces. `access_count` is
  already stored — wire it into importance so re-mentions slow decay and
  one-shot facts age out fast (D4/D5). This is the strongest research backing we
  have for a change to scoring/decay.

### 4.2 Oblivion — EMNLP 2026
- **Decay-driven activation**, not deletion: memories decay along the Ebbinghaus
  forgetting curve by `n_turns_since_last_access`; every *access* reinforces.
- Utility proxy `U_t`, access-frequency proxy `F_t`, decay temperature `T`
  (comfortably configurable). Read path uses *uncertainty* to decide whether to
  query memory at all → 73% token-cost cut at 120K.
- **For us**: validate that discounting should key on **time-since-last-access**
  (reinforcement resets the clock) rather than time-since-creation. Also: an
  "uncertainty-gated retrieval" would let the adaptive path skip injection on
  easy/quasi-redundant turns → direct token savings. Nice E3 lever (D7).

### 4.3 MemoryBank (Zhong et al., AAAI2024) & ACT-R vector learners
- MemoryBank: Ebbinghaus-curve decay for chatbot memory, self-consistent memory
  updates — the original "fade unless reviewed" design.
- ACT-R-style recalling (button-maze / generalisation study): activation =
  **temporal decay + semantic similarity + noise**; retrieval is biased by both
  recency *and* associativity, with probabilistic noise. "Forgetting as a
  feature" — old, unused memories are cheap to recover *if* you keep a
  low-cost trace (embedding) and let the value decay.

### 4.4 RecMem — ACL 2026 Findings
- **Subconscious memory layer**: stores lightweight embeddings only; only invokes
  the LLM to *consolidate* when it observes **sustained recurrence**. Lazy
  consolidation recovers facts plain extraction omits, at large token savings.
- **For us (D8)**: don't run expensive LLM extraction every single turn. Two-tier
  write: cheap embedding-index on every turn; LLM fact-extraction + consolidation
  only on recurrence (or deterministically, the demo's planted turns). Cuts
  extraction tokens + latency and mirrors our E3 numbers (extraction 513ms/turn
  is our dominant cost).

### 4.5 "How Memory Management Impacts Agents" (hit from search #3)
- Experience-following behavior correlates with manager decision quality;
  memory *addition* and *deletion* both measurably change an agent's behavior on
  future tasks — use future-task performance as *free quality labels* for the
  manager. Matches our idea of treating E1 retention-vs-used as a signal.

---

## 5. Context / representation mechanics

### 5.1 "Choosing How to Remember" — arXiv:2602.14038 (STIM/MTEM/LTSM)
- Three-tier sensorimotor→intermediate→long-term store, retrieval+fuse on the
  read path. Fine-grained tiering yields better trace separation than one flat
  store.

### 5.2 "Structural Memory of LLM Agents" — arXiv:2412.15266
- Retrieval methods compared: single-step, iterative (query-refining, RepoCoder
  style), multi-hop rerank. Embeddings matter; structure of memory affects
  results across 4 tasks.

### 5.3 "Active context compression" / Focus — arXiv:2601.07190
- Sawtooth pattern: **consolidate → withdraw** from context. Once useful
  content is confirmed, summarize into a knowledge block and *delete the raw
  logs*. StreamingLLM / LLMLingua as sub-components.
- **For us**: legalizes our `baseline_replay` window (raw recent turns) + a
  separate injected fact set: the baseline is the fresh context, the adaptive
  block is the consolidated knowledge. Cleanest framing for the demo narrative.

### 5.4 'Active context compression' trade-off we already measure
- E3 shows window_replay 1140ms vs adaptive answer 228ms — both real.
  Compression in the literature is about *prompt* cost; we measure both token and
  ms. Keep `debug.timing` per stage so history is auditable (already stored as
  fact_token_measurement, extraction ms, replay ms, answer ms).

---

## 6. Coding-agent-specific memory

### 6.1 RepoCoder (Zhang et al., EMNLP2023 — arXiv:2303.12570)
- Repo-level completion via **iterative retrieval–generation**: generated output
  itself rounds-trips as the retrieval query → converge to the relevant code.
  Two real takeaways: (1) relevance query = last observation, not the whole
  history; (2) recursion beats one-shot top-k for code.
- **For us**: our adaptive replay currently uses the *last turn* as the query
  (server `_adaptive_replay`), which matches "query = recent observation."
  Unevaluated: one refinement loop to re-query with the focused question.

### 6.2 CODEMEM — ACL2026 Findings ("AST-Guided Adaptive Memory for Repository-…")
- Memory manager for iterative repo-level code gen; guided by AST structure
  instead of raw text. Structural indexing (code graph / identifiers) > flat
  fact list for code contexts.
- **For us**: only if we ever ingest real code; our demo is conversation-shaped.
  Note the *principle*: index structure, not prose. Our `category` field is the
  proto-structure.

### 6.3 Evaluating AGENTS.md — arXiv:2602.11988 ("Are repository-level context files helpful?")
- **Context files don't generally improve success rates** and raise inference
  cost >20%; repository overviews are the least helpful section; instructions
  are followed but don't help. *Any* static context injection must be evaluated
  before keeping it.
- **For us**: the demo's "planted facts → re-mentioned" makes prefetchable static
  context and lets us *measure* whether injected context actually helps recall —
  exactly this paper's demanded discipline. Keep the comparison honest: report
  turns where injected facts changed the answer vs merely added tokens (E2
  exact-match machinery).

### 6.4 AceCoder (arXiv:2303.17780) / CODEAGENT (ACL2024)
- AceCoder: guided code generation + **example retrieval** (similar problem →
  few-shot is memory used as prompt). CODEAGENT: 5 integrated tools + tool-use
  strategies for repo-level tasks. Both: retrieval skills ≥ more model compute.

---

## 7. The five numbered design decisions this doc exists to justify

All trace to citations above. Decisions are written as "D# : current → target".
Pending items are marked ⏳ (need a human answer); recommended targets marked ★.

### D1 ⏳★ Shared token budget on the adaptive store
- *Evidence*: MemGPT warning-count eviction (§1.1); AGENTS.md cost discipline
  (§6.3); our first real run's 71-vs-496 comparative claim only stays
  comparable if both paths are bounded.
- *Current*: `baseline_replay` enforces `max_context_tokens:4096` on raw turns;
  adaptive store is unbounded (only dedupe + decay-prune + `top_k:5` cap).
  Optimized bar in UI saturates at 100% (`Math.min`), losing the comparison.
- *Target*: new `memory_optimizer/budget.py` with `token_budget_evict(memories,
  budget, fact_tokens)` shared by live pipeline and E5 replay; eviction policy
  **lowest `current_importance`, oldest tie-break** ★ (SF-AMS utility-driven
  survival, §4.1; MemoryBank fade, §4.3). Live injector also stays within budget
  using measured per-fact tokens (never overshoot).
- *Open question to user*: eviction order (a) lowest-importance/oldest ★,
  (b) oldest-first, (c) largest-token-first.

### D2 ⏳★ Explicit evict-under-pressure alerting (MemGPT "warning token count")
- *Target*: when the computed injection would exceed the budget, tag the
  comparison `budget_squeezed: true` and show *what had to be dropped* — turns
  lost-context counting into an honest metric. This converts the saturated-bar
  bug into a first-class signal.

### D3 ★ Real embedding retrieval (the highest-leverage fix)
- *Evidence*: §2.1 (14–23 pts from retrieval), §2.2 (0.390→0.587 Recall@10),
  §2.3 rerankers; §5.2 embeddings-as-structure.
- *Current*: `_token_overlap` lexical Jaccard only;
  `MemoryRetriever.__docstring__` claims "vector embedding cosine similarity"
  that doesn't exist (retrieval.py:15).
- *Target*: embed each fact ONCE at ingest via `nomic-embed-text` (already in
  Ollama) into a `fact_embedding` field; retrieve by cosine over candidates;
  rerank by `0.6·importance_blend + 0.4·similarity` (calibrate). Keep lexical as
  a *fallback* and record `retrieval_engine: embedding|lexical` so E5 stays
  comparable (`metric_source` precedent). `nomic-embed-text` gives 768-dim,
  ~0-dim token cost per embedding prompt — measure real latency, it's cheap.
  **Store embedding only as the trace (RecMem, §4.4); do NOT re-extract every
  turn** — see D8.

### D4 ⏳★ Re-mention reinforcement (engram activation + usage-fed importance)
- *Evidence*: SF-AMS dynamic importance (§4.1), Oblivion access-reinforcement
  (§4.2), Zep episode-mentions rerank (§2.3), Human-Inspired engram maturation
  (§3.2).
- *Current*: `access_count` recorded but unused by scoring; decay uses
  `source_turn_id`+`base_score` once, never refreshed; re-mentions do nothing.
- *Target*: on any retrieval/re-mention, **refresh the decay clock** and grow
  `current_importance` (usage-redundancy term). Budget-fed eviction (D1) then
  drops un-mentioned stale facts before reinforcement — matching
  MemoryBank/Oblivion forgetting curves. Fine-grained `injection_token_limit`
  ties in: injections are what *count* as reinforcement.
- *Open question*: exact reinforcement semantics — (a) refresh + recompute
  base_score from current `confidence`/relevance (★ recommended), (b) pure clock
  refresh only (Oblivion-style), (c) capped signal (prevent runaway).

### D5 ★ Dedupe as the compression story; kill the fake summarizer
- *Evidence*: summarization drift (§3.1); Human-Inspired "deduplicate, don't
  summarize" (§3.2); Mem0 semantic add/update/noop conflict resolution (§3.2).
- *Current*: `compress_cluster` concatenates under "Summarized Context:"
  (compression.py:23) — a mislabeled non-summarizer that can even *grow*
  token count; dedupe is lexical within-category only.
- *Target*: (a) `dedupe_incremental` upgraded to semantic merge using fact
  embeddings (D3) — conflict-resolve add/update like Mem0; (b) `compress_cluster`
  either removed from the live path or relabeled `concatenate_cluster` and never
  called during ingest; a *true* LLM summarizer is explicitly a later,
  opt-in, high-risk experiment.
- *Status*: **COMPLETED** (Phase 1, commit 0880f1b). `dedupe()` and
  `compress_cluster()` deleted; dead e1-e6 stubs removed. Dedupe is now the
  sole compression story; `dedupe_incremental` handles supersession (D24).

### D6 ★ Category reintrospection / multi-cue retrieval
- *Current*: LLM category is overridden by a keyword sniff; category is a
  flat string used only by dedupe gating.
- *Target* (light): use category as an *entity/index tag* (Zep KG-lite,
  §2.3) in embedding + rerank; enables "topic fan-out" retrieval
  (multi-query per §5.2). Do NOT build a real KG yet.

### D7 ⏳★ Uncertainty-gated injection (read-path economy, Oblivion)
- *Evidence*: Oblivion §4.2 (73% token cut by not always querying);
  AGENTS.md §6.3 (needless context is a cost, not a benefit).
- *Target*: when top-1 embedding similarity is below an *adaptive* threshold,
  skip injection for that turn (still report `injected: "skipped-low-rel"` in
  the comparison). Direct E3/E5 win; must be toggleable for scenario parity.
  Threshold should auto-tune per model from calibration pass, like our
  `_calibrated_constants`.

### D8 ★ Lazy two-tier write (RecMem, LightMem sleep consolidation)
- *Evidence*: RecMem §4.4 (consolidate only on sustained recurrence);
  LightMem §1.3 (decouple expensive consolidation from online path).
- *Target*: cheap embed-on-every-turn (D3); full LLM extraction consolidated
  *only* when a fact recurs or (demo mode) deterministically on planted/filler
  schedule. Extraction is our dominant ms cost (513ms/turn) — this is the E3
  cost lever. Keep `extraction_fallback` counter so we can verify no quality
  loss versus full-extraction runs.

### D9 ★ Writer-held-fixed comparison baselines (task A)
- *Evidence*: retrieval-vs-write decomposition (arXiv:2603.02473) — write
  strategy contributes 3–8 pts vs retrieval's 14–23; "naive RAG as good as
  constructed memory" (arXiv:2607.29104) only holds when the writer is identical
  across arms.
- *Current*: baseline stubs existed but were never instantiated or called; they
  had no uniform interface and no token-budget parity with the adaptive store.
- *Target*: `baselines/baseline_runner.py` + four live baselines
  (sliding_window / memgpt_style / summarization_only / vanilla_rag), all
  consuming the *same* pre-extracted fact stream as the adaptive system and
  bounded by the same D1 budget. Tokens are measured when the caller supplies
  `fact_tokens`; otherwise the manifest says `estimated(word-count)` — never
  passed off as measured. E2/E4 baseline matching reuses the exact live.py rule
  (source_turn match or `_token_overlap` ≥ 0.7).

### D10 ★ Seeded offline paper pipeline (task B/E8/C/F/H)
- *Evidence*: AGENTS.md discipline (arXiv:2602.11988) — any injected/static context
  must be evaluated; the live demo cannot produce rerunnable, multi-seed numbers.
- *Current*: `run_experiments.py` called fake e1–e6 functions; the only real
  measurement path was the live dashboard, which is single-run and not seeded.
- *Target*: `experiments/paper.py` + CLI `run_experiments.py`:
  deterministic seeded replays of E1–E6 per method (adaptive + D9 baselines).
  The writer is held fixed (`--write oracle` = planted facts,
  `--write extract` = real LLM extraction); tokens are `measured` only if
  `--measured` supplies an Ollama token measurer, else the manifest says
  `estimated(word-count)`. E1/E2/E4/E6 reuse live.py's exact semantics so paper
  and dashboard numbers are comparable. E3 is `not_measured` offline and reports
  extraction latency only under `--write extract`. Manifests land in
  `experiments/results/manifest_h<sha>.json` + `latest_manifest.json`, and
  `/api/experiments/latest` falls back to the paper latest so the dashboard can
  render seeded runs when no live run exists.

### D11 ★ Local needled-QA benchmark (task E8)
- *Evidence*: external-data discipline from AGENTS.md (§6.3) — all injected
  context must be evaluated; LongMemEval-style needled QA gives a formalised
  recall-under-distance metric beyond free-form chat.
- *Current*: no standalone benchmark; the only recall numbers come from the
  live dashboard or the seeded paper pipeline (D10).
- *Target*: `experiments/e8_external_benchmark.py` (suite version e8-v1): a
  local needled-QA suite across six categories (who_am_i / project_knowledge /
  statement_on_question / question_on_statement / correction / trap_remains), each
  item a short injection + filler + question conversation. Methods are scored on
  whether the expected fact is retrievable at the question turn (recall) plus the
  tokens paid. Tagged ``synthetic`` (not the real LongMemEval dataset); each run
  is self-contained and rerunnable with `--quick`.

### D12 ★ Sensitivity sweep (task C)
- *Evidence*: any headline number is meaningless without its operative
  configuration (AGENTS.md §6.3); recall-vs-token trade-offs must be read one
  dimension at a time to see which lever matters.
- *Current*: no knob-grid exists; the demo hardcodes turns/density/budget.
- *Target*: `experiments/e7_sensitivity_sweep.py` sweeps turns, signal density,
  token budget, and top_k from a fixed baseline point, one dimension at a time,
  over the D10 seeded oracle pipeline with identical writer facts. Output is a
  stable grid JSON (`e7_sweep.json`) with per-cell per-method mean recall and
  tokens, plus a one-at-a-time view.

### D13 ★ Extraction quality gold set (task D)
- *Evidence*: the writer is the shared upstream of every downstream metric (E1–E8),
  yet no quality baseline exists for its extraction itself; arXiv:2603.02473
  shows 3–8 pts ride on the writer and must be quantified separately.
- *Current*: 2-entry extraction gold file; no recall/F1 metric; no trap or
  correction coverage.
- *Target*: `data/gold_labels/extraction_gold_200.json` (200 turns, ~220 facts)
  generated deterministically by `data/build_gold_set.py` (seed 7). Categories:
  personal / project_context / technical_preference / transient + trap and
  correction turns. `experiments/e0_extraction_quality.py` scores early
  precision/recall/F1 at exact and paraphrase match levels (ROUGE-L F1 >= 0.6
  or BERTScore when available), per-category recall, trap-only recall, and
  correction recall. `--no-llm` is the wiring gate; real runs use the LLM
  extraction pipeline.

### D14 ★ Hard-case scenarios: corrections & negations (task E)
- *Evidence*: arXiv:2603.02473 shows memory systems fail specifically on
  supersession and negation; the demo generator only had trap facts, so no
  metric could see it.
- *Current*: `SyntheticConversationGenerator` gains `conflict_density`
  (durable facts later corrected, original marked `superseded_by`, new fact
  `is_correction_target`) and `negation_density` (`is_negation`). `paper.py`
  threads both via `--conflict-density` / `--negation-density`.
- *Target*: E2 adds `per_category_recall`, `trap_recall`, `correction_recall`,
  `negation_recall`, and `wrongly_retained_after_correction` (fraction of
  superseded facts still in the store — lower is better; 0.80 on the sanity
  gate, honestly reported as a known limitation). E4 adds
  `hard_case_accuracy_by_distance` for trap+correction facts only.
- *Status*: **COMPLETED** (Phases 2+5). Correction supersession fix (D24)
  drives `wrongly_retained` from 0.80 down to 0.0 (single-seed) / 0.2
  (3-seed average, 1/5 wrongly retained). `correction_recall` = 1.0,
  `negation_recall` = 1.0, `trap_recall` = 1.0. Quick gate enforces
  correction_recall >= 0.5 and wrongly_retained <= 0.5.

### D15 ★ Statistical rigor in the aggregate (task F)
- *Evidence*: single-point means ignore seed variance; unadjusted pairwise
  tests inflate family-wise error when several methods are compared at once.
- *Current*: per-seed wilcoxon p-values only; no CI, effect size, or correction.
- *Target*: `experiments/statistics.py` (selftest gate) provides
  `bootstrap_mean_ci`, `bootstrap_ci_mean_diff`, `cohens_d_paired`,
  `holms_correct`, `bonferroni_correct`. `paper.aggregate` now emits per-method
  pooled-per-turn 95% CIs and Cohen's d vs baseline, plus a
  `multiple_comparisons` block: paired wilcoxon per method with Holm and
  Bonferroni adjustment over the compared methods. On the sanity gate:
  adaptive E1 CI [16.0, 21.8], d = -1.654, Holm-adjusted p = 0.0 for every
  method (all `>`-significant vs baseline because injection `<=` baseline tokens).

### D16 ★ Qualitative narrative report (task G)
- *Evidence*: tables of E1–E6 means don't tell the reader *what the system did*;
  a trace-level narrative (phases, spikes, prunes, verbatim passages) is what
  the paper's qualitative section needs.
- *Current*: no narrative output; manifests carry only per-run aggregates.
- *Target*: `experiments/qualitative_report.py` re-runs the manifest's seeded
  adaptive replay and emits `qualitative_report.json`: a narrative paragraph
  (mean injected vs raw window, budget-saturation share), a three-phase
  contrast (under / near / at budget) with per-phase mean tokens and retrieved
  categories, and verbatim edge passages (injection spike turns quoted; prune
  events with their source turn quoted; final store composition). Numbers in
  this report are tagged `metric_source` (G0: never quote a number without its
  operative config).

### D17 ★ Multi-model corpus run (task H)
- *Evidence*: single-writer results overstate robustness; writer variance across
  models is quoted in the write-up only if measured (arXiv:2603.02473 checks
  exactly this).
- *Current*: all experiments ride llama3.1:8b alone.
- *Target*: `experiments/run_multi_model.py` runs the D10 pipeline with real LLM
  extraction + measured tokens across `{llama3.1:8b, qwen2.5:7b, mistral:7b-instruct}`
  (seeds default 3, 150 turns), writing per-model manifests under
  `results/models/<slug>/` so `latest_manifest.json` stays intact. Summary table
  in `models_summary.json` (per-model mean E1 tokens + CI95, E2 recall,
  extraction ms). `--quick` gate = 1 model, 6 turns, 1 seed via real extraction
  (verified: E1 14.0 tok, extraction 700.8 ms).

### D18 ★ Paper artifact generator (task I)
- *Evidence*: tables hand-typed from stale output are the classic reproducibility
  failure; every paper number must be generated, not transcribed.
- *Current*: numbers live only inside gitignored manifests; no LaTeX/figures.
- *Target*: `experiments/make_paper_artifacts.py` emits `artifacts/` (gitignored,
  always regenerable): `tables/table_e1_e2.tex` (cells copied verbatim from the
  manifest aggregate + CI + Cohen's d + Holm-adjusted p), `fig_data/`
  (`e1_tokens_per_turn.json` mean-over-seeds series, `e4_distance_curves.json`),
  and `reproducibility.json` (git revision, python/numpy/scipy versions, config
  hash, seeds, manifest provenance). `--check` gate reloads artifacts and asserts
  table cells equal the manifest values.

### D19 ★ Claim-to-evidence tracker (task J)
- *Evidence*: a paper states claims; a repo should prove each claim is backed by
  an artifact, or say which artifact is missing — otherwise claims drift silently.
- *Current*: no inventory of what evidence exists across the results tree.
- *Target*: `experiments/paper_state.py` scans `results/` (manifests, e0/e7/e8,
  models summary) and emits `paper_state.json`: run inventory (each tagged with
  config + token_source + metric_source) plus a claims table (E1–E6, C7, E0,
  E8, MM) with supported/missing flags. MA/MB gates are self-testing; on the
  current quick artifacts only MM (needs the real 3-model run) is missing.

### D20 ★ Unified adaptive pipeline (task L)
- *Evidence*: server._ingest_turn and the paper replay duplicated the same five
  stages in two divergent implementations; measurements could silently diverge
  from the production path.
- *Current*: server and `experiments/paper.replay_adaptive` both inline the
  stages.
- *Target*: `memory_optimizer/pipeline.py` — `AdaptiveMemoryPipeline` with a
  clean store API (ingest / retrieve / active_memories / pruned / budget
  evictions), `on_stage` latency callback, and `from_settings` factory. The
  server's `_ingest_turn` and `replay_adaptive` now share this one runner;
  gate numbers unchanged after the refactor (E1 19.2 tok, E2 recall 0.72).
  Gotchas fixed during refactor: store identity must be re-synced after each
  ingest (dedupe/decay/budget replace the store list), and demo reset now
  clears list objects in place so the pipeline retains the same list.
- *Status*: server on :9002 restarted on the refactored runner; /api/chat
  ingest, /api/timeline, /api/settings verified live.

### D21 ★ Bug sweep from logs & live paths (task M)

- *Sweep*: grep of `server3.log` + original `server.log` shows no runtime
  errors (0 error/traceback lines). Live paths exercised on the refactored
  :9002 server after [L]: POST /api/chat (fact ingested into store), POST
  /api/experiments/run (E1–E6 live path OK), full 30-turn LLM demo via
  /api/demo/scenario completed 30/30 with error:null and 0 extraction
  fallbacks. All /api/* routes answer with correct codes.
- *Fixes landed in this sweep*: the two store-identity divergences found while
  unifying the pipeline in [L] (paper replay + server reset both re-syncing
  the store list), plus the decimal-rounding call signature bug in the
  statistics module and the `rollup` guard in the J tracker. No further issues
  surfaced from the logs.

### D22 ★ Dashboard is the control center (user directive U4)

Everything the pre-existing dashboard could not do (run pipelines, watch logs,
select multi-model, tune budget/top_k, toggle hard cases) is now controllable
and visible from the UI, not the CLI:

- **Generic research JobRunner** (`server.py`): whitelisted pipelines
  (paper, multimodel, e7, e8, e0, qualitative, artifacts, paper_state, stats,
  `gates` = chained quick gates) spawned as subprocesses with `sys.executable`
  from `BASE_DIR`, argv built from typed field schemas (no shell, no injection),
  per-stage progress markers, cancel via SIGKILL to the process group, and a
  persisted history (`experiments/results/jobs_history.json`, gitignored).
- **Single GPU slot**: research jobs are mutually exclusive with each other and
  with the live demo (both directions guarded).
- **`/api/research/{pipelines,jobs,run,cancel,board}`** — the board adapter
  normalizes the gitignored result files (paper aggregate w/ CI + Cohen's d +
  Holm p, per-category + trap/correction/negation recall, wrongly-retained,
  multi-model `models_summary`, E8/E0/C7, claim coverage) into one tagged
  payload; each block keeps its `metric_source`.
- **Live demo hard cases**: `_build_demo_messages(hard_cases=True)` injects
  correction turns (`is_correction_target`/`supersedes_turn`/`superseded_by`)
  and a negation turn (`is_negation`), mirroring the paper generator's
  semantics; `/api/demo/scenario` takes `hard_cases`; `experiments/live.py`
  E2 now also emits `hard_case` metrics so the live Experiments panel reflects
  them (honest output: correction_recall=0.0, wrongly-retained 1.0 in the
  1-correction 20-turn case).
- **Budget/top_k overrides** reach the paper path via `run_experiments.py
  --budget/--top-k` → `paper.load_settings(overrides)`; defaults unchanged.

### D23 ★ Eviction forensics + live chat microscope (user directive U5)

Every evicted fact/turn now carries a machine-readable *why*, and you can chat
with the bot and watch it pull + store memories live:

- **Per-evicted-turn reasons (both models)**: `_baseline_window()` now charges
  content tokens + `C_label` (per-turn `"user (turn N): "` transcript label +
  separator, measured on the active model, fallback 16) per turn and returns a
  4-tuple `(included, evicted, used, evicted_reasons)`; each reason states how
  full the window already was (`window_used_before`/budget) and what adding the
  turn would need. `baseline_memory_preview` gains `evicted_turns_detail`,
  `format_tokens_per_turn`, `used_tokens`; lost facts carry `lost_reason` via
  `collect_window_facts(turns, cap, reason_map)`. Adaptive side: `budget.py`
  stamps `eviction_reason` on budget-squeeze victims (reason `budget_squeeze`,
  detail with used/budget + which importance key evicted), `decay.py` prune
  reason now embeds M(t), threshold, decayed-over-turns, and lambda; `_ingest_turn`
  persists each budget eviction into `state["budget_evicted_memories"]`
  (`evicted_at_turn`, capped 500, cleared on reset/purge). `/api/state` exposes
  it; both-sides Forgotten lists, the MEM view (Budget-evicted section), the
  barebones-window view, and the pruned list all render the reasons.
- **Token-honesty fix (was 4,920 > 4,096)**: the measured window
  (`prompt_eval_count − C_replay`) exceeded the sized window because sizing
  ignored the per-turn label overhead that the replay prompt actually contains.
  Charging `C_label` per turn in the greedy fit makes the sized window match a
  real measured prompt; verified under a 150-token budget: sized 87 ≤ 150,
  measured 66 ≤ 150, per-turn label 27 tokens.
- **Live chat microscope**: SSE `/api/chat/stream` (stage → token → done events
  with the full turn detail + fresh comparison), enriched `/api/chat`,
  on-demand `/api/chat/baseline` (replays just the raw window, not persisted);
  a busy guard (demo running / research running / another chat in flight → 409)
  keeps the single-GPU slot. Per message the UI shows extraction/ingest/
  retrieval/answer latency chips, the ranked retrieval table (engine, sim,
  M(t), tokens, ✓injected flags), storage rows (new / reinforced / merged with
  counters), and this-turn evictions (decay-pruned + budget-squeezed, each with
  its reason). `retrieval._score_all` extracted so `retrieve(..., ranked=True)`
  returns clean ranked copies (default path unchanged). "New chat (purge)"
  wipes memory via `/api/ethics/purge`.
- **Chat model**: first "live" run exposed that `gemma4:26b` needed 5+ min per
  message on the 5060 Ti (2 LLM calls each); active `llm_model` switched to
  `llama3.1:8b` (~0.3-0.8 s/stage; calibrated constants re-measured on load).
  `gemma4:26b` remains installed for explicit research runs.
- *Verified*: py_compile + node --check on the UI script; chat SSE produced a
  real cached answer; retrieval injected the Whiskers fact with sim 0.675 /
  M 0.937; on-demand baseline recovered "loves salmon" that adaptive extraction
  had dropped; demo>>chat returned 409 busy; 60-turn + 45-turn demos under
  500/150 budgets produced 44-56 barebones evictions all with per-turn reasons,
  24/24 lost facts with `lost_reason`, and 10 persisted adaptive budget-squeeze
  evictions. Config restored to `max_context_tokens: 4096`, `llama3.1:8b`.
- *No change*: adaptive prune-by-decay reason is coded + rendered but never
  fired in these short demos (importances stayed above threshold); it surfaces
  in longer chats when decay actually prunes.

### D24 ★ Correction supersession fix (Phase 2)

Corrections and negations were not handled by the dedupe path — they were
appended as new memories without removing the stale originals, causing
`wrongly_retained_after_correction` to hit 0.80 in the hard-case gate.

- *Evidence*: arXiv:2603.02473 §4.1 (supersession as the core memory-maintenance
  challenge); the existing generator marks `superseded_by` on ground truth but
  `dedupe_incremental` never saw the correction→original link.
- *Fix*: `_is_supersession(old, new)` heuristic — at least 0.6 of old words
  present in new AND new adds marker words (Revised/Correction/NOT/…). When a
  supersession is detected, `dedupe_incremental` replaces the old fact's text
  with the corrected text, stores `superseded_prior_fact`, inherits the new
  fact's confidence, and preserves access bookkeeping. The generator injects
  marker words so `_is_supersession` fires; a bare semantic similar fact
  (without markers) does NOT trigger supersession.
- *Result*: `correction_recall` = 1.0, `wrongly_retained` fraction = 0.0
  (single-seed) / 0.2 (3-seed, 1/5 spurious sibling). Quick gate enforces
  correction_recall >= 0.5, wrongly_retained <= 0.5.
- *Negative control*: applied correction → 0.0 retained; ignored correction →
  1.0 still flagged as stale. Cross-entity template overlap (6/7 = 0.86)
  correctly NOT treated as supersession.

### D25 ★ Write-time salience replaces hardcoded query_relevance=0.8 (Phase 3)

All memories were ingested with a hardcoded `query_relevance=0.8` in
`pipeline.ingest`, regardless of how relevant the fact actually was to the
user message. This inflated the initial importance of low-relevance filler
facts and compressed the score distribution.

- *Evidence*: the scoring formula `M = α·rel + β·util + γ·rec + δ·freq`
  weights relevance at 40% — so `query_relevance=0.8` made every fact look
  equally important at birth, defeating the scoring system's ability to
  differentiate signal from noise.
- *Fix*: `_write_time_salience()` in `memory_optimizer/scoring.py` computes
  cosine similarity (when `embed_fn` available) or token overlap (lexical
  fallback) between the user message and the fact, clamped [0, 1].
  `pipeline.ingest()` uses this per-fact salience when no explicit override
  is supplied; `live.py` `_e5_replay` routes through the pipeline instead of
  calling the scorer directly. `query_relevance=0.8` removed from all code.
- *Result*: facts born from relevant user messages get higher initial
  importance; filler/procedural facts start lower. Quick gate: 24 tests pass,
  E2 proposed recall unchanged at 0.760, E1 adaptive 44.0 vs baseline 323.0.

### D26 ★ Evaluation fixes, manifest versioning, dashboard caveats, audit (Phases 4-6)

- **E2 forgetting precision computed over evicted turns** (Phases 4): the
  baseline window for `*_forgetting_precision` changed from in-window turns to
  the actual evicted-turn set (`_baseline_window_of`), so "proposed forgets
  things the baseline keeps" is measured correctly.
- **`forgetting_horizon_note`** added to E2: states when the horizon is too
  short / nothing evicted to make precision non-discriminating. The dashboard
  renders a caveat row whenever the note is non-discriminating (DASH-01).
- **Manifest versioning** (`pipeline_fix_version: 2`, EVAL-03): `_paper_board`
  exposes `scoring_and_correction_fix_applied = pipeline_fix_version >= 2`, and
  the dashboard shows a banner on historical manifests missing the corrected
  pipeline (DASH-02).
- **Quick-gate enforcement** (TEST-04, Phase 5): `run_experiments.py --quick`
  exits non-zero when `correction_recall` is missing or < 0.5 or
  `wrongly_retained.fraction > 0.5`.
- **Regression tests** (TEST-03, Phase 5): `test_retrieval_ranking.py` guards
  ranking contract, corrected-fact-priority, and reinforcement bookkeeping.
- **E9 correction isolation probe** (TEST-05, Phase 5): standalone hand-rolled
  correction/negation stream replayed through the adaptive pipeline only;

### D27 ★ Context-pressure experiment: point-of-need probe recall (task E)
  Probes are verbatim repetitions of durable facts inserted after the conversation
  ends, evaluated against the method's *final stored* memories using strict
  entity-token matching (key/port number within category). Probes are never appended
  to the replay stream, so turn-window baselines cannot be credited merely because
  the probe message sits in their window, and template re-use cannot fake a hit
  (key_19 does not count as recalling key_25). This metric isolates actual retained
  content from budget-efficiency effects and template inflation; it is the primary
  evaluator for "useful information per active-context token."
  `experiments/e9_correction_isolation.py`, exit code = isolation pass.
- **3-seed revalidation** (REVAL-01..04, Phase 6): transformer validation with
  conflict/negation density 0.1/0.1 — correction_recall 1.0, wrongly_retained
  0.2, E1 regression 7.9% vs vanilla (< 15% gate).

### D28 ★ Coding-context usefulness benchmark — NEGATIVE result for adaptive (E12)
  **Question:** at the SAME active-context token budget, does the adaptive memory
  keep the information needed to answer coding questions better than simpler
  context management? **Answer: no — clearly worse under a fair workload.**
  Workload (`data/coding_workload.py`): 84 semantically diverse facts (24 distinct
  predicate families — see the dedupe-fairness fix below) introduced across a
  400-turn session, with corrections/obsolete decisions, filler, and one ground-truth
  question per fact asked at the end (required token must be injected, forbidden
  obsolete token must be absent). Budget is genuinely binding (natural store 1024
  tok = 2–8× the budget). Embeddings: nomic-embed-text (production retrieval path).
  **3-seed means, query-time injected context (top-k=5):**

  | budget | method | ctx_recall | store_recall | long_range | corr | obsc_ret | ctx_tok |
  |---|---|---|---|---|---|---|---|
  | 128 | adaptive | **0.155** | 0.194 | 0.153 | 0.50 | 0.0 | 58 |
  | 128 | vanilla_rag | **0.905** | 0.905 | 0.926 | 0.00 | 1.0 | 60 |
  | 256 | adaptive | **0.167** | 0.345 | 0.157 | 0.50 | 0.0 | 56 |
  | 512 | adaptive | **0.167** | 0.532 | 0.157 | 0.50 | 0.0 | 56 |
  | 512 | summarization | 0.583 | 0.583 | 0.591 | 0.00 | 0.0 | 511 |
  | any | sliding_window | 0.048 | 0.048 | 0.037 | 0.00 | 0.0 | 125 |

  **Two independent causes (both verified, neither is workload artifact):**
  1. **Bounded store + decay** limit how much is retained: adaptive's store grows
     with budget (10→20→31 facts) and store_recall rises 0.19→0.35→0.53, but it
     still holds only ~37% of facts at the top budget — vanilla keeps all 86.
  2. **Importance-dominant retrieval** wastes what *is* retained. Production
     ranking is `0.6·importance + 0.4·similarity` (D3 marked this "calibrate").
     For a given query across the store, importance spans 0.47 (0.6×0.21..0.99)
     while nomic cosine similarity spans only 0.069 (0.427..0.600), so ranking is
     ~7:1 query-independent. Read-only ablation at budget 512:
     production **0.167**, pure-similarity **0.421**, pure-importance **0.115**.
     Context recall is FLAT (0.155→0.167) even as store_recall nearly triples —
     the signature of query-insensitive injection.
  **Decision:** per the "do not optimise adaptive to beat the benchmark" directive,
  the retrieval weights were NOT tuned. D3's "calibrate" is now a concrete,
  evidence-backed remediation for external review (recalibrate similarity vs
  importance, or make the blend query-conditional). Even with pure similarity
  (0.421) adaptive would still trail vanilla_rag (0.905) at ~60 injected tokens;
  the bounded store is the harder limit.
  **Benchmark-fairness fixes made (not tuning):** (a) facts were packed into the
  first half then decay-pruned before end-of-session queries — now spread across
  the whole session (`build_coding_session`); (b) the old single-skeleton template
  expansion made facts near-duplicates that adaptive's dedupe (cosine ≥0.90)
  legitimately collapsed (332→189) — replaced with 24 distinct predicate families,
  now dedupe retains ~86–88%. `tests/test_coding_benchmark.py` locks both in.

### D29 ★ Separate memory capacity from active context; make retrieval query-first
  **Problem:** the E12 diagnosis revealed two structural issues:
  1. `max_context_tokens` simultaneously capped the *long-term memory store* AND
     the *active LLM context* — they are conceptually different resources.
  2. Retrieval ranking `0.6·importance + 0.4·similarity` made injection
     query-insensitive (importance span 0.47 vs similarity span 0.069).

  **Fix:** three coordinated changes.

  **29.1 Separate budgets (config.yaml, pipeline.py, paper.py):**
  - `max_context_tokens` → hard cap on active LLM context (unchanged default 4096)
  - `injection_token_limit` (default 0 = use max_context_tokens) → explicit
    active injection cap
  - `memory_store_token_budget` (default 0 = no hard cap) → independent
    long-term memory capacity; retention governed by compression + decay
  - Pipeline ingest now evicts to `memory_store_token_budget` (if > 0) instead of
    `max_context_tokens`. Active injection uses `injection_token_limit`.

  **29.2 Query-first retrieval (retrieval.py):**
  - Default weights changed from `imp_weight=0.6, sim_weight=0.4, cat_bonus=0.1`
    to `imp_weight=0.15, sim_weight=0.85, cat_bonus=0.02`
  - Ranking formula: `score = sim_weight * sim + imp_weight * importance + bonus`
    (bonus no longer multiplied by sim_weight)
  - Conceptual separation: **historical importance = retention signal**,
    **query similarity = retrieval signal**.

  **29.3 E12 now measures retrieval loss separately:**
  - `store_recall` = answerable from full store
  - `context_recall` = answerable from injected context
  - `retrieval_loss = store_recall - context_recall` (info existed but retrieval failed)
  - `mean_context_utilization = mean_context_tokens / active_context_budget`
  - Explicit `memory_store_token_budget: max(4096, budget*4)` in E12 settings so
    the store is NOT the bottleneck; we measure retrieval quality.

  **Status:** implementation complete; targeted validation pending (Change 8).

### D30 ★ Phase 9 Generalization and Causal Ablation Validation
  **Objective:** determine whether the Phase-8 architectural improvements
  (separate store/active budgets, query-first retrieval) generalize across
  seeds, budgets, query families, and retrieval modes — and isolate which
  change caused the improvement.

  **30.1 E13 Generalization Experiment (experiments/e13_generalization.py):**
  - 5 seeds × 5 budgets × 5 methods with Phase-8 defaults
  - Budget range expanded: [64, 128, 256, 512, 1024]
  - 5 seeds: [42, 43, 44, 45, 46]
  - Metrics per cell: context_recall, store_recall, retrieval_loss, long_range_recall,
    correction_recall, obsolete_retention, mean/max_context_tokens,
    mean/max_context_utilization, budget_violation_count, store_tokens,
    store_count, store_budget_binding, budget_eviction_count
  - Query-family breakdown by qtype: requirements, architecture, constraints,
    implementation, bug_fix, module_relation, feature_flag, correction, obsolete
  - All results deterministic and machine-readable

  **30.2 Causal Ablation A — Store-Capacity Separation:**
  - Four store policies tested on identical workload/active budgets:
    1. **coupled_old_behavior**: store_budget = active_budget (pre-Phase-8)
    2. **matched_budget**: store_budget = active_budget (direct comparison)
    3. **separated_4x**: store_budget = max(4096, active × 4) (Phase-8 default)
    4. **unbounded**: store_budget = 0 (no hard cap; decay/dedupe only)
  - Key comparison: store_recall and context_recall at each policy
  - Workload scale verified: natural_store_tokens (1024) < 4x budget at 1024 → unbounded ≠ separated_4x
  - *Finding*: coupled/matched policies collapse store_recall to ~0.18 at 128 budget;
    separated_4x/unbounded achieve ~0.79 store_recall with only ~0.08 retrieval loss

  **30.3 Causal Ablation B — Retrieval Policy Profiles:**
  - Four fixed profiles on separated_4x store (128, 256, 512 budgets; seeds 42-44):
    1. **phase8**: imp_weight=0.15, sim_weight=0.85, cat_bonus=0.02 (default)
    2. **legacy_phase7**: imp_weight=0.60, sim_weight=0.40, cat_bonus=0.10
    3. **pure_similarity**: imp_weight=0.00, sim_weight=1.00, cat_bonus=0.00
    4. **pure_importance**: imp_weight=1.00, sim_weight=0.00, cat_bonus=0.00
  - Admission rule unchanged: `sim >= threshold or importance > 0.6`
  - *Finding*: Phase-8 (0.15/0.85) achieves 0.70 ctx_recall vs legacy 0.27;
    pure similarity reaches 0.86 ctx_recall; pure importance drops to 0.18
    → query-first retrieval is the primary driver of Phase-8 gains

  **30.4 Embedding vs Lexical Sensitivity:**
  - Embeddings (nomic-embed-text): 0.70 ctx_recall, 0.08 retrieval_loss
  - Lexical fallback: 0.57 ctx_recall, 0.20 retrieval_loss
  - Effect does not disappear without embeddings but is substantially weaker
  - Phase-8 retrieval relies on embedding similarity discrimination

  **30.5 Architectural Regression Tests Added:**
  - `test_token_limit_exceeds_top_k`: token_limit allows > top_k when budget permits
  - `test_token_limit_never_exceeded`: hard cap never violated
  - `test_store_budget_independent_of_active_context`: store can exceed active budget
  - `test_unbounded_store_allows_growth`: memory_store_token_budget=0 removes hard cap

  **30.6 Evidence Summary:**
  | Comparison | ctx_recall Δ | store_recall Δ | retrieval_loss Δ |
  |------------|-------------|---------------|------------------|
  | coupled vs separated_4x (128) | +0.53 | +0.61 | -0.29 |
  | legacy_phase7 vs phase8 (128) | +0.43 | +0.14 | -0.29 |
  | pure_sim vs phase8 (128) | +0.16 | +0.08 | -0.07 |
  | embedding vs lexical (128) | +0.13 | +0.01 | -0.12 |

  **30.7 Limitations:**
  - Natural store tokens (1024) < 4× budget at 1024 → unbounded ≈ separated_4x at high budgets
  - 64-token budget may be too restrictive for meaningful retrieval
  - Only nomic-embed-text tested; other embeddings untested
  - Decay policy fixed per directive; store pressure × decay interaction unexplored

  **Status:** Validation complete. Phase-8 improvements **generalized** across seeds,
  budgets, and query families. Store-capacity separation and query-first retrieval
  are **both causal drivers**; query-first retrieval is the larger contributor.

### D31 ★ Phase 10 Retention Diagnosis — Causal Analysis of Store Losses
  **Motivation:** Phase 8 separated long-term memory capacity from active context
  and changed retrieval to query-first. Phase 9 confirmed generalization and
  isolated two causal drivers. However, at low active-context budgets,
  adaptive still trails vanilla RAG, and a significant fraction of remaining
  loss occurs **inside the memory store** before query-time retrieval.

  **Objective:** Identify exactly **why useful task facts are disappearing from
  the long-term store** — before changing the memory policy.

  **31.1 Instrumentation Enhancements (pipeline.py):**
  - Per-ingest store-pressure tracking: `pre_store_tokens`, `store_budget`,
    `store_over_budget_before_eviction`, `evicted_count`, `post_store_tokens`,
    `eviction_occurred` — recorded every ingest turn.
  - Per-fact survival diagnostics fields added to memory records:
    `initial_importance`, `write_time_salience`, `max_importance`,
    `final_importance`, `dedupe_count`, `superseded`, `last_access_turn`,
    `access_count`, `decay_steps_survived`, `pruned_by_decay`,
    `removed_by_store_budget`, `retrieval_candidate`, `retrieval_selected`.
  - Budget-eviction count now recorded from actual `token_budget_evict()` calls.

  **31.2 Fact Lifecycle Classification:**
  Every ground-truth coding fact is classified into exactly one terminal state:
  - `PRESENT_AND_RETRIEVED` — in store and retrieved at query time
  - `PRESENT_BUT_NOT_RETRIEVED` — in store but retrieval failed to select it
  - `REMOVED_BY_DECAY` — pruned by decay engine (M(t) < threshold)
  - `REMOVED_BY_STORE_BUDGET` — evicted by token_budget_evict
  - `MERGED_BY_DEDUPE` — merged into another fact via compression
  - `SUPERSEDED_CORRECTLY` — replaced by a correction fact
  - `SUPERSEDED_INCORRECTLY` — stale fact not properly superseded
  - `NEVER_STORED` — never entered the store
  - `OTHER` — unclassified

  **31.3 E14 Retention Diagnosis Experiment (experiments/e14_retention_diagnosis.py):**
  - Coding workload: 400 turns, scale=3, budgets [64, 128, 256, 512],
    seeds [42, 43, 44, 45, 46], nomic-embed-text
  - Store budget: max(4096, budget × 4) — intentionally larger than active context
  - Per-fact survival diagnostics tracked from extraction through retrieval
  - Category survival: requirements, architecture, constraints, implementation,
    bug_fix, module_relation, feature_flag, correction, obsolete, long-range
  - Retention-time curves: 8 age buckets (0–49, 50–99, ..., 350–399 turns)
  - Ablation A: no decay (decay_enabled=false)
  - Ablation B: no pruning threshold (decay runs but no threshold deletion)
  - Ablation C: score component diagnosis (relevance, utility, recency, frequency)
  - Ablation D: write-time salience diagnosis (writer relevance vs query-time similarity)
  - Correction/obsolete safety gate maintained in all ablations

  **31.6 Observed Results (E14, 5 seeds × 4 budgets):**

  Dominant cause-of-loss (mean across seeds):
  | Budget | stored+retr | stored/not | decay | budget_evict | dedupe | super_ok | super_bad | corr_recall |
  | 64 | 43.4 | 5.2 | 31.4 | 0.0 | 4.0 | 2.0 | 0.0 | 0.5 |
  | 128 | 54.6 | 5.0 | 20.4 | 0.0 | 4.0 | 2.0 | 0.0 | 0.7 |
  | 256 | 64.4 | 5.0 | 10.6 | 0.0 | 4.0 | 2.0 | 0.0 | 1.0 |
  | 512 | 73.8 | 6.0 | 0.2 | 0.0 | 4.0 | 2.0 | 0.0 | 1.0 |

  Ablation: no_decay → 0.0 decay removals at all budgets; store_recall = 0.917;
  corr_recall = 1.0 at all budgets (including 64, vs 0.5 with decay).
  Ablation: no_prune_threshold → 0.0 decay removals (decays but never deleted);
  store_recall = 0.917 at all budgets; ctx_recall at 64 drops to 0.862 vs 0.893
  (decayed importance scores lower retrieval scores but no facts deleted).

  Write-time salience vs query-time similarity:
  - mean write_time_salience ≈ 0.98 for all retrieved/missed (near-ceiling, no discrimination)
  - mean retrieval_sim ≈ 0.48 for all; spearman ≈ -0.08 (near-zero/negative)

  **31.7 Observed Facts vs Inferences:**

  OBSERVED:
  1. Decay + pruning threshold is the dominant loss mechanism (31.4/86 facts at 64 budget)
  2. Store-budget eviction = 0 (store 4096+ never binds with 86 facts)
  3. Dedupe removes 4 template facts consistently (4.6%), non-preventable given workload design
  4. Supersession always correct (0 incorrectly retained)
  5. Correction recall degrades at low budget (0.5 at 64) because correction facts also decay
  6. Write-time salience is near-ceiling (~0.98) and does NOT predict which facts are retrieved

  INFERRED:
  7. Retrieval reinforcement creates a stable attractor: facts repeatedly selected never decay
     (once importance > 0.6 + sim < 0.35, they cycle through admission/bump)
  8. Budget constraint matters only indirectly: lower budget → fewer facts admitted → less
     reinforcement → faster decay; the budget itself does not evict
  9. Dedupe losses are fixed regardless of budget (template collision by design, not a bug)

  UNRESOLVED:
  10. Interaction of decay with write-time salience: all facts written with high relevance
      to their turn, so write-time salience provides no discrimination
  11. The 5 PRESENT_BUT_NOT_RETRIEVED facts per budget (~6%) are consistently missed;
      root cause (insufficient sim for admission vs insufficient budget for injection)
      not isolated

  **Status:** Diagnosis complete. No memory-policy changes made in this phase.

---

### D32 ★ Phase 11 Retention-Policy Separation — Activation vs Survival
  **Motivation:** D31 showed decay→prune is the dominant store-loss mechanism.
  The fix must separate **activation** (how "hot" a memory is right now —
  ranking for retrieval) from **survival** (how much a memory should be kept —
  eviction). Decay should lower retrieval ranking, NOT delete the only surviving
  representation. Deletion should be reserved for genuine store-capacity
  pressure, decided by a retention-oriented priority. (Terminology: "activation-
  aware retrieval vs retention-aware survival", NOT "task-aware memory".)

  **32.1 New Scores:**
  - `current_importance` — unchanged dynamic activation (category decay lambdas;
    decays with time since last access; reinforced on retrieval). Ranking only.
  - `retention_priority` — NEW stable evidence score:
    `min(1.0, base_score * (1 + access_gain * (access_count - 1)))`.
    Deterministic (no LLM/embedding cost), no temporal decay, grows with repeated
    use. Eviction ONLY.

  **32.2 Retention Modes (`config.yaml` `retention`, `decay.py`):**
  - `hard_threshold` (LEGACY / Phase 10): prune when `current_importance` <
    pruning.threshold (0.2). Preserved for backward compatibility + E14 replication.
  - `soft_decay`: importance decays, never threshold-prunes; memory flagged
    `retained_below_threshold=True`; store survives unless the store-token budget
    actually evicts.
  - `dual_score`: soft_decay + store-budget eviction uses `retention_priority`.
  - `no_decay`: ablation diagnostic (lambdas 0, threshold 0.0) — upper bound.
  - Unknown mode → `ValueError` (validated before the loop).

  **32.3 E15 Experiment (experiments/e15_retention_policy.py):**
  - Primary grid: 4 policies × budgets [64,128,256,512] × 5 seeds, 400 turns,
    scale 3, store_budget 4096, nomic-embed-text.
  - Stress grid (mandatory — dual_score only differs under real store pressure):
    scale auto-grown until natural store ≥ 8192 tokens (measured 8272 @ scale 27);
    store budgets [1024,2048,4096] × active [64,128,256] × seeds [42,43,44],
    1200 turns × 27 cells. Policies: soft_decay, dual_score, hard_threshold.
  - Lifecycle registry adds a retention transcript per fact (current_importance,
    minimum, retention_priority, maximum, retained_below_threshold, terminal state).
  - Retrieval path FROZEN during the grid: ranking uses `current_importance`
    only (causal isolation — separation can only act through survival/eviction).

  **32.6 Observed Results (E15 primary, 5 seeds × 400 turns):**

  Comparison (mean context_recall / store_recall / decay_removals / corr_recall):
  | Policy | 64 | 128 | 256 | 512 |
  | hard_threshold | ctx 0.576 store 0.672 rm 31.4 | 0.705/0.774/20.4 | 0.800/0.852/10.6 | 0.917/0.917/0.2 |
  | soft_decay | ctx 0.862 store 0.917 rm 0 | 0.883/0.917/0 | 0.888/0.917/0 | 0.917/0.917/0 |
  | dual_score | identical to soft_decay (no store evictions in primary) |
  | no_decay | ctx 0.893 store 0.917 rm 0 | (flat) |

  - soft/dual correction_recall = 1.0 at ALL budgets (legacy 0.5/0.7 at 64/128)
  - obsolete_retention = 0.0 everywhere; supersession always correct
  - Revived (would-be-pruned, later retrieved): 30.0/20.4/9.6/0.2 at 64/128/256/512

  **32.7 Observed Facts vs Inferences:**

  OBSERVED (stress grid, 1200 turns, ~8272 natural store tokens):
  1. soft vs hard at store 2048/4096: store_recall 0.775–0.889 vs 0.762–0.869;
     hard loses up to ~409 facts to decay-pruning even before eviction binds
  2. Tightest store (1024): dual ctx 0.368/0.434/0.503 vs soft 0.322/0.405/0.484;
     dual correction_recall 0.94 vs soft 0.28; dual evicts 430 vs soft 518 facts
     (more stable store); dual archival store_recall 0.629 vs soft 0.638 (−1 pt)
  3. dual evicts by retention_priority: keeps once-retrieved/reinforced facts,
     evicts never-retrieved facts → task recall improves, archival completeness dips
  4. No stale resurrection: obsolete_retention 0 under every policy/budget

  INFERRED:
  5. Under hard_threshold, decay→prune destroys facts that store capacity could
     have preserved; the "memory loss" is policy-caused, not capacity-caused
  6. Retention-priority eviction is the causation-complete version of soft_decay
     under pressure: fewer evictions, better context + correction safety

  UNRESOLVED:
  7. Archival store_recall at the tightest store (−1 pt) — never-retrieved facts
     are evicted first; whether this matters depends on long-horizon reuse that
     the 1200-turn grid does not exercise
  8. `PRESENT_BUT_NOT_RETRIEVED` ≈ 6 facts persists across all policies (D31 #11)

  **Decision (reviewer-confirmable auto-labels):**
  - SEPARATING ACTIVATION FROM SURVIVAL IS SUPPORTED → SUPPORTED
  - RETENTION PRIORITY FOR STORE EVICTION IS SUPPORTED → SUPPORTED
  - SOFT RETENTION RECOVERS FACTS BUT CREATES A STALENESS TRADEOFF → NOT
    SUPPORTED (obsolete 0, correction improved — no staleness cost observed)

  **Advanced production default:** `retention: {mode: dual_score,
  eviction_priority: retention_priority}`. Decay no longer deletes; store
  eviction prefers low retention_priority. Pipeline fallback for settings
  without the key keeps `hard_threshold` (backward-compatible; E14 tests pin
  legacy mode explicitly). All other defaults unchanged.

  **Status:** Phase 11 complete. Committed + pushed. Candidate advanced.

---

### D33 ★ Phase 12 Retention Selectivity — task_affinity does NOT beat dual_score (E16)
  **Motivation:** E15 (D32) showed `retention_priority` eviction is the
  causation-complete survival policy, but left two gaps: (a) E15's evaluation was
  *token-based*, and tokens (`memcache`, `redis`) legitimately occur in unrelated
  facts, so `obsolete_retention > 0` misread correct supersession as staleness;
  (b) `retention_priority` is fed by `access_count`, which increments on every
  retrieval — a fact that is retrieved survives, so "retention-priority survival"
  may trivially equal "retrieval-driven survival" (a feedback echo), leaving no
  un-contaminated evidence that survival tracks task relevance.

  **33.1 Identity-safe evaluation (data/coding_workload.py):**
  - Every gt fact gets a deterministic `fact_id =
    category:qtype:source_turn:SHA256(precise_fact)[:16]` (SHA-256, NOT Python
    `hash()`); queries carry `target_fact_ids` / `forbidden_fact_ids`; corrections
    assert the new fact's id present and the old fact's id absent
    (`superseded_prior_fact_id`).
  - Metrics: `fact_identity_context/store_recall`,
    `fact_identity_correction_recall`, `fact_identity_obsolete_retention` —
    immune to token collisions. Legacy token metrics kept for comparability.

  **33.2 Access separation (retrieval.py, compression.py):**
  - `retrieval_access_count` increments ONLY on retriever selection (`_bump`);
    `ingest_reinforcement_count` increments ONLY on compressor merge/supersession;
    legacy `access_count` unchanged (backward compat). The feedback loop can now
    be diagnosed into retrieval- vs ingest-fed components
    (`diagnostics.spearman_vs_future_use` per cell).

  **33.3 Task-state signal + causal baselines (task_state.py, budget.py, pipeline.py):**
  - `TaskStateTracker(window=32):` FIFO of fact-carrying turn embeddings;
    `task_affinity = mean(top-4 cosine(fact, tracker))`, clamped [0,1]; observed
    ONLY at the current turn's ingest (causal, future-blind).
  - Eviction policies (all `retention.mode=dual_score`, retrieval FROZEN):
    `dual_score` (retention_priority), `base_score_only` (no access/importance/
    recency signals), `task_affinity` (lexicographic `(task_affinity,
    retention_priority)`), `random` (seeded lower bound), `oracle_future_use`
    (OFFLINE-only future-peeking upper bound; future labels reach only oracle +
    metric code; no-future-leakage assertions in `replay_session`).
  - `protect_corrections=True` in E16: a current correction is never evicted
    while its superseded predecessor exists; `protected_capacity_conflict`
    records when the store budget is too tight to honour it.

  **33.4 E16 Grid (experiments/e16_retention_selectivity.py):**
  - 5 policies × STORE[256,512,1024,2048] × active[64,128,256] × seeds[42..46] =
    300 cells, 1200 turns, scale 27 (662 gt facts, natural store far above
    budget = genuine pressure), nomic-embed-text. No-pressure diagnostic
    (store_budget=0) = fingerprints identical across policies (True); window
    ablation 16/32/64 (sensitivity only).

  **33.5 Observed Results (E16):**

  - E15 raw-JSON reconciliation (81 stress cells): token-based
    `obsolete_retention > 0` in 27 cells, `correction_recall < 1` in 44, both in
    11, clean 21; `SUPERSEDED_INCORRECTLY = 0` in ALL cells — the store never
    resurrects a corrected fact; the real loss is the NEW corrected fact missing
    from the final store (dual 0/0/0, hard_threshold 18/14/13, soft_decay
    18/13/9 at store 1024/2048/4096). Token census confirms collisions (scale 27:
    "memcache" in 3 authoritative facts, "redis" in 8, "100"/"250" only in the
    superseded/revised constraint).

  - identity_store_recall (mean over active/seeds): dual_score 0.057/0.103/
    0.129/0.253 vs task_affinity 0.049/0.093/0.126/0.252 vs base_score_only
    0.045/0.088/0.120/0.241 vs random 0.048/0.095/0.126/0.250 vs oracle
    0.057/0.103/0.129/0.253 at store 256/512/1024/2048. All policies ~equal at
    every budget; oracle ≈ dual at the grid's store range (future-use facts are
    a small faction of retained facts, so eviction precision cannot separate).

  - 8-criterion decision: **task_affinity does NOT advance (0/8 passes)** —
    c1 FAIL (future-use retention recall 0.0709 vs dual 0.0802 at low store,
    < +0.03 margin), c6 FAIL (0/5 seeds at tightest store), c7 FAIL (0.0709 vs
    base_score_only 0.0662, < +0.03), c8 FAIL (oracle store gap task +0.009 vs
    dual −0.0003). c2–c5 PASS (precision 1.0 both; correction recall 1.0;
    obsolete retention 0; context unchanged).

  INFERRED:
  - The retrieval-feedback hypothesis is NOT falsified, but `task_affinity`
    (future-blind task similarity) is no better than random at predicting future
    use in this workload — the future-relevant signal is the fact's *own* text
    similarity to future queries, which naive cosine-to-recent-turns does not
    capture, and which `dual_score`'s retrieval-feedback already approximates.
  - Identity-safe metrics close the E15 token-collision gap: fact-level obsolete
    retention is 0 everywhere; the residual E15 signal was a metric artifact,
    but the correction-content loss (new fact evicted) was real and is now
    guarded by correction protection.

  UNRESOLVED:
  - Whether any future-blind survival signal can beat retrieval-feedback driven
    survival in this workload remains open; `task_affinity` is not it.
  - At the low store range (256–512) oracle ≈ causal policies: eviction picks
    among many never-retrieved facts and future-use is a tiny fraction, so the
    policy cannot express much separation; a workload where future use is more
    concentrated (fewer distinct target facts, denser repeats) may discriminate
    better.

  **Decision (reviewer-confirmable auto-labels):**
  - TASK_AFFINITY FOR RETENTION IS SUPPORTED → **NOT SUPPORTED** (fails c1/c6/
    c7/c8; 0/5 seeds; not even oracle-superior)
  - RETRIEVAL-FEEDBACK DOMINATES SURVIVAL → **NOT OBSERVED** (inconclusive: all
    causal policies ≈ oracle at low store; the workload's future-use density
    cannot discriminate)
  - IDENTITY-SAFE EVAL + CORRECTION PROTECTION → **SUPPORTED** (obsolete 0, corr
    recall 1.0, SUPERSEDED_INCORRECTLY 0 across all 300 cells)

  **Production default unchanged:** `retention: {mode: dual_score,
  eviction_priority: retention_priority}`. No silent flip; `task_affinity`
  remains available (opt-in) for workloads with concentrated future use.

  **Status:** Phase 12 complete. Committed + pushed. Candidate rejected
  (dual_score retained). STOP — no Phase 13.

---

## 8. Open questions for the human (blocking decisions)

1. **D1 eviction policy**: lowest-`current_importance` + oldest tiebreak ★ / oldest-first / largest-token-first.
2. **D4 reinforcement semantics**: refresh + recompute base_score ★ / clock-refresh only / capped signal.
3. **D4/D1 demo controls**: `planted_count` cap — 50 leaving ≥20 filler turns ★ vs up to `turns/2`; and the exact meaning of the density control (re-mention rate of planted facts — user's 10-Sep mental model — vs "% of turns that carry a planted fact" which the current UI tooltip claims but the code ignores).
4. **D7 fill-mode**: when relevance candidates are scarce, fill the remaining `injection_token_limit` with best remaining by rank ★ vs strict top-k only (conservative, avoids injecting irrelevant context à la §6.3).

## 9. Benchmark / source index (how we'll verify changes)

- Our own: E1 (retention, real measured tokens), E2 (exact-match answer recall),
  E3 (latency+token cost, real prompt_eval_count), E5 (measured vs fallback
  token source, `token_source: measured`).
- External papers referencing these benchmarks we should read before any model
  swap: LoCoMo (MemReranker, SF-AMS use it), LongMemEval (LightMem, Zep, SF-AMS;
  LightMem sees "up to 10.9% gains"), Deep Memory Recall DMR (MemGPT, Zep).
  **Our demo is not a benchmark substitute** — it's an honest, real-everything
  single-conversation rig that tests comparative behavior under bounded context.
- Calibration constants (`_calibrated_constants`), per-fact token cache
  (`_fact_tokens`) and `metric_source` flags are the instrumentation that keeps
  every one of these numbers comparable across changes.