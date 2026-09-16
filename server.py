import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import uuid
import yaml
from difflib import SequenceMatcher
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import requests

from memory_optimizer.extraction import FactExtractor, build_extraction_prompt
from memory_optimizer.scoring import ImportanceScorer
from memory_optimizer.decay import CategoryDecayEngine
from memory_optimizer.retrieval import MemoryRetriever, _token_overlap
from memory_optimizer.compression import MemoryCompressor
from memory_optimizer.budget import token_budget_evict
from memory_optimizer.embeddings import embed_ollama

BASE_DIR = Path(__file__).parent

RESEARCH_DIR = BASE_DIR / "experiments/results"
JOBS_HISTORY_FILE = RESEARCH_DIR / "jobs_history.json"


def _load_jobs_history():
    try:
        if JOBS_HISTORY_FILE.exists():
            return json.loads(JOBS_HISTORY_FILE.read_text())
    except Exception:
        pass
    return []

app = FastAPI(title="Adaptive Memory Manager API", version="1.0.0")

_cors_origins = [o.strip() for o in os.environ.get(
    "CORS_ORIGINS", "http://127.0.0.1:9002,http://localhost:9002"
).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

with open(BASE_DIR / "config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

extractor = FactExtractor(endpoint=cfg["system"]["ollama_endpoint"], model=cfg["system"]["llm_model"])
scorer = ImportanceScorer(weights=cfg["scoring_weights"])
decay_engine = CategoryDecayEngine(lambdas=cfg["decay_lambdas"], pruning_threshold=cfg["pruning"]["threshold"])
compressor = MemoryCompressor()
ollama_base = cfg["system"]["ollama_endpoint"]
embed_model = cfg["system"].get("embedding_model", "nomic-embed-text")


def _embed_texts(texts):
    return embed_ollama(texts, model=embed_model, endpoint=ollama_base)


retriever = MemoryRetriever(
    top_k=cfg["retrieval"]["top_k"],
    sim_threshold=cfg["retrieval"]["similarity_threshold"],
    embed_fn=_embed_texts,
    embedding_model=embed_model,
)

settings = {
    "max_context_tokens": int(cfg["system"]["max_context_tokens"]),
    "active_model": cfg["system"]["llm_model"],
    "top_k": int(cfg["retrieval"]["top_k"]),
    "similarity_threshold": float(cfg["retrieval"]["similarity_threshold"]),
    "pruning_threshold": float(cfg["pruning"]["threshold"]),
    "enable_compression": bool(cfg["compression"].get("enabled", True)),
    "injection_token_limit": int(cfg["system"].get("injection_token_limit", 0)),
}

state = {
    "current_turn": 0,
    "active_memories": [],
    "pruned_memories": [],
    "raw_history": [],
    "active_token_trace": [],
    "inject_token_trace": [],
    "ground_truth": [],
    "latency_stats": {},
    "timeline": [],
    "last_evicted_ids": [],
    "last_demo": {},
    "demo_job": {},
"baseline_replay": [],
    "last_context": {},
    "last_budget_evictions": [],
    "budget_evicted_memories": [],
    "chat_busy": False,
    "research_jobs": [],
    "research_busy": False,
    "research_history": _load_jobs_history(),
}

_research_lock = threading.Lock()
_chat_lock = threading.Lock()

_gpu_cache = {"ts": 0, "data": None}

_token_constants = {"model": None}
_fact_token_cache = {}


class ChatMessage(BaseModel):
    user_message: str = Field(min_length=1, max_length=20000)


def _measure_prompt_tokens(prompt: str, model: str = None, timeout: int = 60):
    model = model or settings["active_model"]
    try:
        resp = requests.post(f"{ollama_base}/api/generate", json={
            "model": model,
            "prompt": str(prompt),
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.1, "num_predict": 1, "num_ctx": 8192},
        }, timeout=timeout)
        return resp.json().get("prompt_eval_count")
    except Exception:
        return None


def _calibrated_constants():
    model = settings["active_model"]
    if _token_constants.get("model") == model and all(
            _token_constants.get(k) is not None for k in ("C_extract", "C_replay", "C_answer", "C_label")):
        return _token_constants
    c_extract = _measure_prompt_tokens(build_extraction_prompt("X"))
    c_extract = (c_extract - 1) if c_extract is not None else None
    replay_const = (f"{WINDOW_REPLAY_PROMPT}\n\n--- BEGIN TRANSCRIPT ---\n\n--- END TRANSCRIPT ---\n\nReturn JSON:")
    c_replay = _measure_prompt_tokens(replay_const)
    answer_const = (f"{ANSWER_PROMPT}\n\n--- CONTEXT ---\n(no context entries)\n---\n\nQuestion: X\nAnswer:")
    c_answer = _measure_prompt_tokens(answer_const)
    c_answer = (c_answer - 1) if c_answer is not None else None
    c_label = None
    if c_replay is not None:
        label = _measure_prompt_tokens("user (turn 7): ")
        newline = _measure_prompt_tokens("\n")
        if label is not None and newline is not None:
            c_label = label + newline
    if c_label is None:
        c_label = 16
    _token_constants.update(
        model=model, C_extract=c_extract, C_replay=c_replay, C_answer=c_answer, C_label=c_label,
        measured=c_extract is not None and c_replay is not None and c_answer is not None,
    )
    return _token_constants


def _extraction_content_tokens(meta: dict) -> int:
    constants = _calibrated_constants()
    count = meta.get("prompt_eval_count")
    c = constants.get("C_extract")
    if count is None or c is None:
        return None
    return max(0, count - c)


def _fact_tokens(fact_text: str, model: str = None) -> int:
    model = model or settings["active_model"]
    key = (model, fact_text)
    cached = _fact_token_cache.get(key)
    if cached is not None:
        return cached
    count = _measure_prompt_tokens(fact_text, model=model)
    if count is None:
        return None
    _fact_token_cache[key] = count
    return count


def _lookup_or_measure(text: str) -> int:
    return _fact_tokens(text)


def _persist_settings():
    with open(BASE_DIR / "config.yaml", "r") as f:
        doc = yaml.safe_load(f)
    doc["system"]["max_context_tokens"] = int(settings["max_context_tokens"])
    doc["system"]["llm_model"] = settings["active_model"]
    doc["retrieval"]["top_k"] = int(settings["top_k"])
    doc["retrieval"]["similarity_threshold"] = float(settings["similarity_threshold"])
    doc["pruning"]["threshold"] = float(settings["pruning_threshold"])
    doc["compression"]["enabled"] = bool(settings["enable_compression"])
    doc["system"]["injection_token_limit"] = int(settings["injection_token_limit"])
    with open(BASE_DIR / "config.yaml", "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def _gpu_snapshot():
    now = time.time()
    if now - _gpu_cache["ts"] < 2 and _gpu_cache["data"] is not None:
        return _gpu_cache["data"]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        if len(parts) == 2:
            _gpu_cache.update(ts=now, data={"used_mib": int(parts[0]), "total_mib": int(parts[1])})
    except Exception:
        _gpu_cache.update(ts=now, data=None)
    return _gpu_cache["data"]


def _model_family(model: str) -> str:
    m = model.lower()
    for fam in ("llama", "qwen", "gemma", "mistral", "deepseek", "moondream", "phi"):
        if fam in m:
            return fam
    return "default"


def _kv_bytes_per_token() -> int:
    table = cfg["vram"]["bytes_per_token_estimate"]
    return int(table.get(_model_family(settings["active_model"]), table["default"]))


def _vram_block(baseline_tokens: int, optimized_tokens: int) -> dict:
    sample = _gpu_snapshot()
    bpt = _kv_bytes_per_token()
    est_b = baseline_tokens * bpt / 1048576
    est_o = optimized_tokens * bpt / 1048576
    return {
        "sampled": sample,
        "kv_estimate_bytes_per_token": bpt,
        "kv_baseline_mib": round(est_b, 2),
        "kv_optimized_mib": round(est_o, 2),
        "kv_delta_mib": round(est_b - est_o, 2),
        "label": "estimate",
        "active_model": settings["active_model"],
    }


def _baseline_window():
    """Return (included, evicted, used, evicted_reasons).

    The greedy fit charges every turn its *content* tokens plus the per-turn
    transcript formatting (C_label: "user (turn N): " label + separator) so that
    the sized window matches what a real replay prompt measures
    (prompt_eval_count - C_replay). Each evicted turn carries the exact reason
    it was rejected (which newer turns already filled the window and by how much).
    """
    history = state["raw_history"]
    budget = settings["max_context_tokens"]
    label_cost = int(_calibrated_constants().get("C_label") or 0)
    included, used, evicted = [], 0, []
    evicted_reasons = []
    for h in reversed(history):
        cost = h["tokens"] + label_cost
        if used + cost <= budget:
            included.append(h)
            used += cost
        else:
            evicted.append(h)
            evicted_reasons.append({
                "turn_id": h["turn_id"],
                "tokens": h["tokens"],
                "format_tokens": label_cost,
                "window_used_before": used,
                "budget": budget,
                "reason": (f"newer turns already fill {used} of {budget} window tokens; "
                           f"adding this turn ({cost} tokens incl. {label_cost} formatting) "
                           f"would need {used + cost} > {budget}"),
            })
    included.reverse()
    evicted.reverse()
    evicted_reasons.reverse()
    return included, evicted, used, evicted_reasons


def _naive_probe(query: str):
    included, _, _, _ = _baseline_window()
    scored = []
    for h in included:
        sim = _token_overlap(query, h["user"])
        if sim > 0:
            scored.append({"turn_id": h["turn_id"], "excerpt": h["user"][:90], "sim": round(sim, 2)})
    scored.sort(key=lambda x: x["sim"], reverse=True)
    return scored[:5]


WINDOW_REPLAY_PROMPT = (
    "You are an assistant whose ONLY source of information is the conversation transcript below. "
    "The transcript may be incomplete because older turns were truncated to fit a context budget.\n\n"
    "From the transcript: 1) Answer the user's latest message using ONLY facts present in the transcript. "
    "If the transcript does not contain enough context to answer, reply exactly NO_LOCAL_ANSWER. "
    "2) List every factual claim visible in the transcript as short standalone facts.\n\n"
    "Be terse: answer in at most 12 words, each fact at most 10 words.\n\n"
    "Return strict JSON only: {\"answer\": \"...\", \"facts\": [\"fact1\", \"fact2\", ...]}"
)


def _llm_window_replay(included_turns: list) -> dict:
    if not included_turns:
        return {"answer": "", "facts_seen": [], "raw": "",
                "prompt_eval_count": 0, "eval_count": 0, "ms": 0}
    transcript = "\n".join(f'user (turn {h["turn_id"]}): {h["user"]}' for h in included_turns)
    prompt = (f"{WINDOW_REPLAY_PROMPT}\n\n--- BEGIN TRANSCRIPT ---\n{transcript}\n"
              f"--- END TRANSCRIPT ---\n\nReturn JSON:")
    try:
        resp = requests.post(f"{ollama_base}/api/generate", json={
            "model": settings["active_model"],
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": "30m",
            "options": {"temperature": 0.1, "num_predict": 300, "num_ctx": 8192},
        }, timeout=180)
        data = resp.json()
        raw = str(data.get("response", ""))
        try:
            out = json.loads(raw)
        except Exception:
            out = {}
        return {
            "answer": str(out.get("answer") or ""),
            "facts_seen": [str(f) for f in (out.get("facts") or [])][:40],
            "raw": raw,
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "ms": (resp.elapsed.total_seconds() * 1000) if hasattr(resp, "elapsed") else None,
        }
    except Exception as e:
        return {"answer": "", "facts_seen": [], "raw": "", "error": str(e),
                "prompt_eval_count": None, "eval_count": None, "ms": None}


ANSWER_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the context memory entries below. "
    "If the context entries do not contain the answer, reply exactly NO_LOCAL_ANSWER. "
    "Answer in a short sentence of at most 15 words."
)


def _llm_answer(query: str, context_facts: list) -> dict:
    if not query:
        return {"answer": "", "prompt_eval_count": 0, "eval_count": 0, "ms": 0}
    context = "\n".join(f"- {f['fact']}" for f in (context_facts or [])) or "(no context entries)"
    prompt = f"{ANSWER_PROMPT}\n\n--- CONTEXT ---\n{context}\n---\n\nQuestion: {query}\nAnswer:"
    try:
        resp = requests.post(f"{ollama_base}/api/generate", json={
            "model": settings["active_model"],
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.1, "num_predict": 80, "num_ctx": 8192},
        }, timeout=180)
        data = resp.json()
        return {
            "answer": str(data.get("response", "") or "").strip(),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "ms": (resp.elapsed.total_seconds() * 1000) if hasattr(resp, "elapsed") else None,
        }
    except Exception:
        return {"answer": "", "prompt_eval_count": None, "eval_count": None, "ms": None}


def _content_tokens(prompt_eval_count, constant) -> int:
    if prompt_eval_count is None or constant is None:
        return None
    return max(0, prompt_eval_count - constant)


def _build_comparison(context_facts: list = None, probe_query: str = None, full: bool = False) -> dict:
    if context_facts is None:
        context_facts = state["last_context"].get("facts", [])
    history = state["raw_history"]
    trace = state["active_token_trace"]
    inj_trace = state["inject_token_trace"]

    cumulative_raw = 0
    token_history = []
    for i, h in enumerate(history):
        cumulative_raw += h["tokens"]
        token_history.append({
            "turn_id": h["turn_id"],
            "raw_cumulative": cumulative_raw,
            "active_tokens": trace[i] if i < len(trace) else 0,
            "injected_tokens": inj_trace[i] if i < len(inj_trace) else 0,
        })

    included, evicted, used, evicted_reasons = _baseline_window()
    reason_by_id = {r["turn_id"]: r for r in evicted_reasons}

    last_replay = state["baseline_replay"][-1] if state["baseline_replay"] else None
    measured_win = last_replay.get("window_tokens") if last_replay else None
    baseline_tokens = measured_win if measured_win is not None else used

    active_measured = [m.get("measured_tokens") for m in state["active_memories"]]
    unmeasured_active = sum(1 for t in active_measured if t is None)
    active_tokens = sum(t for t in active_measured if t is not None)

    context_tokens = state["last_context"].get("injected_tokens") if state.get("last_context") else None

    evicted_ids = [h["turn_id"] for h in evicted]
    evicted_turn_ids_set = set(evicted_ids)

    gt_by_turn = {}
    for gt in state["ground_truth"]:
        gt_by_turn.setdefault(gt["source_turn"], []).append(gt)
    evicted_gt_turns = set(gt_by_turn) & evicted_turn_ids_set

    def _held(ev_turn, fact_text):
        for m in state["active_memories"]:
            if m.get("source_turn_id") == ev_turn:
                return m
            if SequenceMatcher(None, fact_text.lower(), m["fact"].lower()).ratio() >= 0.7:
                return m
        return None

    forgot_but_kept = []
    forgot_by_both = 0
    for ev_turn in sorted(evicted_gt_turns):
        for gt in gt_by_turn.get(ev_turn, []):
            mem = _held(ev_turn, gt["fact"])
            if mem is not None:
                forgot_but_kept.append({
                    "fact": gt["fact"],
                    "category": gt["category"],
                    "source_turn": ev_turn,
                    "current_importance": round(mem.get("current_importance", mem.get("base_score", 0)), 3),
                    "is_planted": True,
                })
            else:
                forgot_by_both += 1

    kept_extra_other = []
    for mem in state["active_memories"]:
        if mem.get("source_turn_id") in evicted_turn_ids_set - evicted_gt_turns:
            kept_extra_other.append({
                "fact": mem["fact"],
                "category": mem["category"],
                "source_turn": mem["source_turn_id"],
                "current_importance": round(mem.get("current_importance", mem.get("base_score", 0)), 3),
            })

    planted_gt = [g for g in state["ground_truth"] if g.get("planted")]
    hard_case_gt = [g for g in state["ground_truth"] if g.get("hard_case")]
    planted_evicted_ids = evicted_turn_ids_set if evicted_turn_ids_set else set()

    baseline_memory_preview = {
        "included_turns": [{"turn_id": h["turn_id"], "excerpt": h["user"][:90], "tokens": h["tokens"]} for h in included],
        "evicted_turn_ids": evicted_ids,
        "evicted_turns_detail": evicted_reasons,
        "evicted_tokens": sum(h["tokens"] for h in evicted),
        "format_tokens_per_turn": reason_by_id.get(evicted_ids[0], {}).get("format_tokens", 0) if evicted_ids else 0,
        "used_tokens": used,
        "planted_facts_in_window": sum(
            1 for g in planted_gt if g["source_turn"] not in planted_evicted_ids
        ),
        "planted_facts_in_window_vs_evicted": {
            "in_window": sum(1 for g in planted_gt if g["source_turn"] not in planted_evicted_ids),
            "evicted": sum(1 for g in planted_gt if g["source_turn"] in planted_evicted_ids),
        },
        "planted_facts_total": len(planted_gt),
        "hard_case_facts": len(hard_case_gt),
        "hard_case_corrections": sum(1 for g in hard_case_gt if g.get("is_correction_target")),
        "hard_case_negations": sum(1 for g in hard_case_gt if g.get("is_negation")),
        "ground_truth_total": len(state["ground_truth"]),
    }

    def collect_window_facts(turns, cap, reason_map=None):
        out, seen = [], set()
        for h in turns:
            for f in h.get("facts", []):
                text = f.get("fact", "")
                if not text or text in seen:
                    continue
                seen.add(text)
                src_turn = f.get("source_turn_id", h["turn_id"])
                item = {
                    "fact": text,
                    "category": f.get("category", ""),
                    "source_turn": src_turn,
                }
                if reason_map:
                    item["lost_reason"] = reason_map.get(src_turn, {}).get("reason", "")
                out.append(item)
                if len(out) >= cap:
                    return out
        return out

    baseline_memory_summary = {
        "remembered": collect_window_facts(included, 5000 if full else 200),
        "lost": collect_window_facts(evicted, 20000 if full else 500, reason_map=reason_by_id),
        "remembered_turns": len(included),
        "lost_turns": len(evicted),
        "remembered_unbounded": sum(len(h.get("facts", [])) for h in included),
        "lost_unbounded": sum(len(h.get("facts", [])) for h in evicted),
    }

    return {
        "system_token_budget": settings["max_context_tokens"],
        "cumulative_history_tokens": cumulative_raw,
        "baseline_context_tokens": baseline_tokens,
        "baseline_context_tokens_measured": measured_win is not None,
        "baseline_turns_in_context": len(included),
        "baseline_evicted_turn_count": len(evicted),
        "baseline_evicted_turn_ids": evicted_ids if full else evicted_ids[:200],
        "baseline_evicted_tokens": sum(h["tokens"] for h in evicted),
        "active_memory_tokens": active_tokens,
        "active_memory_count": len(state["active_memories"]),
        "active_memory_unmeasured_count": unmeasured_active,
        "pruned_memory_count": len(state["pruned_memories"]),
        "adaptive_budget_cap": settings["max_context_tokens"],
        "budget_squeezed": bool(state["last_budget_evictions"]),
        "adaptive_budget_evictions": state["last_budget_evictions"],
        "injection_token_limit": settings["injection_token_limit"],
        "retrieval_engines": state["last_context"].get("engines", []),
        "optimized_context_tokens": context_tokens,
        "optimized_facts_in_context": len(context_facts or []),
        "metric_source": "live model prompt_eval_count (content tokens, measured live)",
        "facts_barebones_forgot_you_still_keep": forgot_but_kept,
        "forgot_by_both": forgot_by_both,
        "kept_extra_other": kept_extra_other,
        "baseline_memory_preview": baseline_memory_preview,
        "baseline_memory_summary": baseline_memory_summary,
        "baseline_naive_probe": _naive_probe(probe_query) if probe_query else None,
        "token_history": token_history,
        "latency_stats": state["latency_stats"],
        "settings": settings,
        "vram": _vram_block(used, active_tokens),
    }


def _record_latency(stage: str, ms: float):
    state["latency_stats"].setdefault(stage, []).append(ms)

pipeline = None

def _get_pipeline():
    global pipeline
    if pipeline is None:
        from memory_optimizer.pipeline import AdaptiveMemoryPipeline
        pipeline = AdaptiveMemoryPipeline(settings, scorer, decay_engine, compressor=compressor,
                                          retriever=retriever, store=state["active_memories"],
                                          pruned=state["pruned_memories"], on_stage=_record_latency)
    return pipeline


def _record_timeline(t_id, user_message, facts, context_facts, pruned_this_turn, injected_tokens,
                     baseline_answer=None, optimized_answer=None, facts_seen=None):
    included, evicted, used, _ = _baseline_window()
    evicted_ids = [h["turn_id"] for h in evicted]
    last = set(state["last_evicted_ids"])
    newly = [i for i in evicted_ids if i not in last][-50:]
    state["last_evicted_ids"] = evicted_ids[-200:]

    newly_lost = []
    for mem in state["active_memories"]:
        if mem.get("source_turn_id") in set(newly):
            newly_lost.append({
                "fact": mem["fact"],
                "category": mem["category"],
                "current_importance": round(mem.get("current_importance", mem.get("base_score", 0)), 3),
            })

    state["timeline"].append({
        "turn_id": t_id,
        "user_message": user_message,
        "extracted_facts": [dict(f) for f in facts],
        "optimized_retrieved_this_turn": [dict(f) for f in (context_facts or [])],
        "optimized_pruned_this_turn": [dict(f) for f in pruned_this_turn],
        "optimized_injected_tokens": injected_tokens,
        "optimized_budget_squeezed": bool(state["last_budget_evictions"]),
        "optimized_budget_evictions": state["last_budget_evictions"],
        "baseline_window_turn_ids": [h["turn_id"] for h in included][-30:],
        "baseline_window_tokens": used,
        "baseline_newly_evicted_turn_ids": newly,
        "baseline_newly_lost_facts": newly_lost,
        "baseline_answer": baseline_answer or "",
        "optimized_answer": optimized_answer or "",
        "baseline_facts_seen": facts_seen or [],
    })
    state["timeline"] = state["timeline"][-10000:]


def _ingest_turn(user_message: str, facts: list, external_turn_id: int = None, replay: bool = False,
                 real_msg_tokens: int = None) -> dict:
    if external_turn_id is None:
        state["current_turn"] += 1
        t_id = state["current_turn"]
    else:
        state["current_turn"] = external_turn_id
        t_id = external_turn_id

    result = _get_pipeline().ingest(t_id, user_message, facts,
                                    fact_tokens=_lookup_or_measure, embed_fn=_embed_texts)
    state["active_memories"] = _get_pipeline().active_memories
    state["last_budget_evictions"] = result["budget_evictions"]
    for e in result["budget_evictions"]:
        entry = dict(e)
        entry["evicted_at_turn"] = t_id
        if not entry.get("eviction_reason"):
            entry["eviction_reason"] = {
                "reason": "budget_squeeze",
                "detail": "evicted to fit the adaptive token budget",
            }
        state["budget_evicted_memories"].append(entry)
    state["budget_evicted_memories"] = state["budget_evicted_memories"][-500:]
    pruned_this_turn = result["pruned"]

    if real_msg_tokens is None:
        real_msg_tokens = _fact_tokens(user_message)
    msg_tokens = real_msg_tokens if real_msg_tokens is not None else 1

    state["raw_history"].append({
        "turn_id": t_id,
        "user": user_message,
        "tokens": msg_tokens,
        "token_measured": real_msg_tokens is not None,
        "facts": [dict(f) for f in facts],
    })

    state["active_token_trace"].append(result["active_tokens"])

    context_facts = result["injected"]
    inject_trace_val = state["last_context"].get("injected_tokens") if state.get("last_context") else None
    state["inject_token_trace"].append(inject_trace_val)

    baseline_answer, optimized_answer, facts_seen = "", "", []
    window_tokens, window_turn_ids = None, []
    injected_tokens = None
    if replay:
        included, _, _, _ = _baseline_window()
        window_turn_ids = [h["turn_id"] for h in included]
        constants = _calibrated_constants()
        t0 = time.perf_counter()
        bl = _llm_window_replay(included)
        _record_latency("window_replay", (time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        ans = _llm_answer(user_message, context_facts)
        _record_latency("answer", (time.perf_counter() - t0) * 1000)
        window_tokens = _content_tokens(bl.get("prompt_eval_count"), constants.get("C_replay"))
        injected_tokens = _content_tokens(ans.get("prompt_eval_count"), constants.get("C_answer"))
        if injected_tokens is None and context_facts:
            context_text = "\n".join(f["fact"] for f in context_facts)
            injected_tokens = _fact_tokens(context_text)
        if window_tokens is None and included:
            transcript = "\n".join(f'user (turn {h["turn_id"]}): {h["user"]}' for h in included)
            window_tokens = _fact_tokens(transcript)
        baseline_answer = bl.get("answer", "")
        facts_seen = bl.get("facts_seen", [])
        optimized_answer = ans.get("answer", "")
        state["baseline_replay"].append({
            "turn_id": t_id,
            "query": user_message,
            "baseline_answer": baseline_answer,
            "optimized_answer": optimized_answer,
            "facts_seen": facts_seen,
            "window_tokens": window_tokens,
            "window_turn_ids": window_turn_ids,
            "injected_tokens": injected_tokens,
            "window_eval_count": bl.get("eval_count"),
            "answer_eval_count": ans.get("eval_count"),
            "window_latency_ms": bl.get("ms"),
            "answer_latency_ms": ans.get("ms"),
            "window_turn_count": len(window_turn_ids),
        })

    state["last_context"] = {
        "query": user_message,
        "facts": [dict(f) for f in context_facts],
        "injected_tokens": injected_tokens,
        "token_measured": injected_tokens is not None,
        "engines": sorted({f.get("retrieval_engine", "") for f in context_facts if f.get("retrieval_engine")}),
    }

    _record_timeline(t_id, user_message, facts, context_facts, pruned_this_turn, injected_tokens,
                     baseline_answer=baseline_answer, optimized_answer=optimized_answer, facts_seen=facts_seen)
    return _build_comparison(context_facts, probe_query=None)


FILLERS = {
    "short": (" Routine monitoring shows nothing unusual; the log pipelines are "
              "stable and the refactor is progressing on schedule without conflicts."),
    "normal": (" The overnight batches all finished clean; nothing unusual surfaced in the log "
               "pipelines, the deployment target is stable, and the refactor work we scheduled "
               "is progressing exactly as planned without any unhandled conflicts or open questions "
               "that would need escalation."),
    "long": (" The overnight batches all finished clean and nothing unusual surfaced in the log "
             "pipelines; the deployment target for the current sprint is stable with no pending "
             "rollback needed, and the refactor work we scheduled earlier is progressing exactly "
             "as planned without any unhandled conflicts, flaky tests, or open questions that would "
             "need escalation, so the team can stay on the committed timeline risk-free."),
}


SEED_MESSAGES = [
    (2, "personal",
     "Heads up, remember this: I have a cat named Whiskers. Please remember my cat's name and that she loves tuna.",
     "User has a cat named Whiskers"),
    (0.10, "project_context",
     "Please note for the project: key_alpha_legacy is a config key my core project depends on, so remember it stays on.",
     "User core project: config key_alpha_legacy is on"),
    (0.50, "project_context",
     "Architecture decision under discussion: each feature's flag key will segment the rollout. Remember this project decision.",
     "Project architecture decision under discussion: flag keys segment the rollout"),
    (0.90, "project_context",
     "Project update: the feature flag config key_demo_90 is enabled for the rollout. Please remember that key_demo_90 is enabled.",
     "Project feature flag config key_demo_90 is enabled"),
    (0.95, "technical_preference",
     "Important requirement: the production cluster must run PostgreSQL 15 as the primary database. No exceptions, remember this.",
     "User primary DB requirement: must use PostgreSQL 15"),
]


def _generated_planted(n: int):
    """Deterministic extra planted facts beyond the 5 hand-written seeds."""
    cats = ["project_context", "technical_preference", "personal"]
    c = cats[(n - 5) % len(cats)]
    if c == "project_context":
        fact = f"Project feature flag config key_demo_r{n} is enabled for the rollout"
        msg = (f"Project note: the rollout flag config key_demo_r{n} is being enabled. "
               f"Remember that config key_demo_r{n} stays on.")
    elif c == "technical_preference":
        fact = f"User primary DB requirement: staging cluster runs PostgreSQL 15 instance db_{n}"
        msg = (f"Requirement: the staging cluster primary database must be PostgreSQL 15 "
               f"(instance db_{n}). No exceptions.")
    else:
        fact = f"User has a recurring scheduled job that must run daily: daily_{n}"
        msg = f"Remember the daily scheduled job daily_{n} must keep running."
    return c, msg, fact


def _build_demo_messages(turns: int = 150, density_pct: float = 20.0,
                         planted_count: int = 5, filler: str = "normal",
                         hard_cases: bool = True, n_corrections: int = 2,
                         n_negations: int = 1):
    num_turns = max(20, min(500, int(turns)))
    filler_text = FILLERS.get(str(filler), FILLERS["normal"])
    density_pct = max(0.0, min(100.0, float(density_pct)))

    seed = int(cfg["system"].get("seed", 42))
    rng = random.Random(seed * 1009 ^ num_turns * 31 ^ int(planted_count) * 17 ^ int(density_pct))

    planted = max(1, min(int(planted_count), num_turns - 4))
    base_facts = [m[1:] for m in SEED_MESSAGES]

    if planted == 1:
        positions = [2]
    else:
        positions = {2}
        for i in range(1, planted):
            positions.add(min(num_turns, 3 + (num_turns - 3) * i // (planted - 1)))
        positions = sorted(positions)
    if positions and positions[-1] == num_turns and len(positions) > 1:
        positions[-1] = num_turns - 1

    planted_facts = []
    for idx, pos in enumerate(positions):
        if idx < len(base_facts):
            p_cat, p_msg, p_fact = base_facts[idx]
        else:
            p_cat, p_msg, p_fact = _generated_planted(idx + 1)
        planted_facts.append((pos, p_cat, p_msg, p_fact))

    available = [t for t in range(3, num_turns + 1) if t not in set(positions)]
    n_density = int(round(num_turns * density_pct / 100.0))
    rng.shuffle(available)
    density_turns = set(available[:n_density])
    density_map = {t: planted_facts[i % len(planted_facts)] for i, t in enumerate(sorted(density_turns))}

    messages, ground_truth = [], []
    planted_pos = {p[0] for p in planted_facts}
    for turn_id in range(1, num_turns + 1):
        if turn_id in planted_pos:
            _, p_cat, p_msg, p_fact = next(p for p in planted_facts if p[0] == turn_id)
            messages.append({"turn_id": turn_id, "user": p_msg + filler_text})
            ground_truth.append({"source_turn": turn_id, "fact": p_fact, "category": p_cat, "is_trap": False,
                                 "planted": True})
        elif turn_id in density_map:
            _, _, _, fact = density_map[turn_id]
            messages.append({
                "turn_id": turn_id,
                "user": (f"Turn {turn_id}: Discussing routine system log outputs and generic code refactoring."
                         f"{filler_text} Reconfirming an earlier note: {fact}."),
            })
        else:
            messages.append({
                "turn_id": turn_id,
                "user": f"Turn {turn_id}: Discussing routine system log outputs and generic code refactoring.{filler_text}",
            })

    if hard_cases:
        durable = [p for p in planted_facts if p[1] != "transient"]
        durable.sort(key=lambda p: p[0])
        used_turns = planted_pos | density_turns
        free = [t for t in range(5, num_turns - 1) if t not in used_turns]
        free.sort(reverse=True)
        req_corr, req_neg = max(0, int(n_corrections)), max(0, int(n_negations))
        budget = min(len(durable), len(free))
        if req_corr + req_neg > 0:
            if req_corr + req_neg <= budget:
                n_corr, n_neg = req_corr, req_neg
            else:
                n_corr = min(req_corr, round(budget * req_corr / (req_corr + req_neg)))
                n_neg = min(req_neg, budget - n_corr)
        else:
            n_corr = n_neg = 0
        corrected = durable[:n_corr]
        for i, (pos, cat, _, fact) in enumerate(corrected):
            ct = free[i] if i < len(free) else None
            if ct is None:
                continue
            new_fact = f"Revised requirement: {fact} is no longer the case"
            messages[ct - 1] = {
                "turn_id": ct,
                "user": (f"Correction to an earlier requirement (from turn {pos}): that earlier note is no longer "
                         f"right. Updated requirement: {new_fact}. Please discard the old version." + filler_text),
            }
            for g in ground_truth:
                if g["source_turn"] == pos and g.get("fact") == fact:
                    g["superseded_by"] = ct
                    g["obsolete"] = True
            ground_truth.append({
                "source_turn": ct, "fact": new_fact, "category": cat, "is_trap": False,
                "is_correction_target": True, "hard_case": True, "supersedes_turn": pos,
                "expected_needed_later": True,
            })
        remaining = [p for p in durable if p not in corrected]
        n_neg = max(0, min(n_neg, len(remaining), len(free) - n_corr))
        for j, (pos, cat, _, fact) in enumerate(remaining[:n_neg]):
            nt = free[n_corr + j] if n_corr + j < len(free) else None
            if nt is None:
                continue
            negated = f"Correction to requirement: NOT {fact.lower()}"
            messages[nt - 1] = {
                "turn_id": nt,
                "user": f"Please note this down: {negated}." + filler_text,
            }
            ground_truth.append({
                "source_turn": nt, "fact": negated, "category": cat, "is_trap": False,
                "is_negation": True, "hard_case": True, "expected_needed_later": True,
            })
    return messages, ground_truth


def _reset_demo_state():
    state["current_turn"] = 0
    state["active_memories"].clear()
    state["pruned_memories"].clear()
    state["raw_history"] = []
    state["active_token_trace"] = []
    state["inject_token_trace"] = []
    state["latency_stats"] = {}
    state["timeline"] = []
    state["last_evicted_ids"] = []
    state["ground_truth"] = []
    state["baseline_replay"] = []
    state["last_context"] = {}
    state["budget_evicted_memories"] = []
    state["chat_busy"] = False


def _run_llm_demo(messages):
    job = state["demo_job"]
    job["running"] = True
    job["started_at"] = time.time()
    total = len(messages)
    try:
        for i, m in enumerate(messages, start=1):
            if job.get("cancel_requested"):
                job["cancelled"] = True
                break
            t0 = time.perf_counter()
            extracted, emeta = extractor.extract_facts(m["turn_id"], m["user"])
            _record_latency("extraction", (time.perf_counter() - t0) * 1000)
            real_msg_tokens = _extraction_content_tokens(emeta)
            if job.get("cancel_requested"):
                job["cancelled"] = True
                break
            job["extraction_fallback"] = job.get("extraction_fallback", 0) + (1 if emeta.get("fallback") else 0)
            _ingest_turn(m["user"], extracted, external_turn_id=m["turn_id"], replay=True,
                         real_msg_tokens=real_msg_tokens)
            job["turns_done"] = i
            job["current_turn"] = m["turn_id"]
    except Exception as e:
        job["error"] = str(e)
    finally:
        job["running"] = False
        job["finished_at"] = time.time()
        job["turns_total"] = total


def _llm_stream_answer(query: str, context_facts: list):
    context = "\n".join(f"- {f['fact']}" for f in (context_facts or [])) or "(no context entries)"
    prompt = f"{ANSWER_PROMPT}\n\n--- CONTEXT ---\n{context}\n---\n\nQuestion: {query}\nAnswer:"
    t0 = time.perf_counter()
    resp = requests.post(f"{ollama_base}/api/generate", json={
        "model": settings["active_model"],
        "prompt": prompt,
        "stream": True,
        "keep_alive": "30m",
        "options": {"temperature": 0.1, "num_predict": 80, "num_ctx": 8192},
    }, stream=True, timeout=180)
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        chunk = obj.get("response", "")
        if chunk:
            yield chunk
        if obj.get("done", False):
            break
    _record_latency("answer", (time.perf_counter() - t0) * 1000)


def _text_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a.lower() == b.lower():
        return True
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.85


def _chat_guard():
    if state["demo_job"].get("running"):
        return {"status": "busy", "detail": "A demo is running; cancel it before chatting", "job": state["demo_job"]}
    if state["research_busy"]:
        return {"status": "busy", "detail": "A research job is running; wait or cancel it before chatting"}
    if state["chat_busy"]:
        return {"status": "busy", "detail": "Another chat message is still being processed"}
    return None


def _begin_chat():
    with _chat_lock:
        guard = _chat_guard()
        if guard:
            return guard
        state["chat_busy"] = True
        return None


def _end_chat():
    with _chat_lock:
        state["chat_busy"] = False


def _prepare_chat_turn(message: str) -> dict:
    """Extract + ingest (no LLM replay) + rank retrieval + storage snapshot-diff.

    Returns everything the UI needs to show what the system pulled from memory
    and wrote to it for this message.
    """
    t_id = state["current_turn"] + 1
    store_before = {
        "active": len(state["active_memories"]),
        "pruned": len(state["pruned_memories"]),
        "budget_evicted": len(state["budget_evicted_memories"]),
    }
    t0 = time.perf_counter()
    extracted, emeta = extractor.extract_facts(t_id, message)
    _record_latency("extraction", (time.perf_counter() - t0) * 1000)
    real_msg_tokens = _extraction_content_tokens(emeta)
    prior_texts = [m.get("fact", "") for m in state["active_memories"]]
    prior_pruned_count = len(state["pruned_memories"])

    t0 = time.perf_counter()
    comparison = _ingest_turn(message, extracted, replay=False, real_msg_tokens=real_msg_tokens)
    _record_latency("ingest", (time.perf_counter() - t0) * 1000)
    ctx = state["last_context"].get("facts", [])
    if state["last_context"].get("injected_tokens") is None and ctx:
        state["last_context"]["injected_tokens"] = sum(_fact_tokens(f["fact"]) or 0 for f in ctx)

    t0 = time.perf_counter()
    ranked = retriever.retrieve(message, state["active_memories"], ranked=True)
    _record_latency("retrieval", (time.perf_counter() - t0) * 1000)
    injected_texts = [f["fact"] for f in ctx]
    for r in ranked:
        r["injected"] = any(_text_match(i, r.get("fact", "")) for i in injected_texts)

    storage = _classify_storage(extracted, prior_texts, state["active_memories"])
    pruned_this_turn = state["pruned_memories"][prior_pruned_count:]
    return {
        "turn_id": t_id,
        "query": message,
        "extracted": [dict(f) for f in extracted],
        "ranks": ranked,
        "storage": storage,
        "pruned_this_turn": [dict(p) for p in pruned_this_turn],
        "budget_evictions": [dict(b) for b in state["last_budget_evictions"]],
        "store_before": store_before,
        "comparison": comparison,
    }


def _classify_storage(extracted, prior_texts, now):
    rows = []
    for f in extracted:
        text = f.get("fact", "")
        if not text:
            continue
        hit = next((m for m in now if _text_match(m.get("fact", ""), text)), None)
        was_present = any(_text_match(t, text) for t in prior_texts)
        if hit is not None and not was_present and _text_match(hit.get("fact", ""), text) and hit.get("fact") != text:
            kind, note = "merged", "merged into an overlapping cluster"
        elif hit is not None and was_present:
            kind, note = "reinforced", "re-mentioned existing memory — access refreshed"
        else:
            kind, note = "new", "written to the adaptive store"
        rows.append({
            "fact": text, "kind": kind, "note": note,
            "category": f.get("category", ""),
            "tokens": hit.get("measured_tokens") if hit else None,
            "importance": round((hit.get("current_importance") or 0), 3) if hit else None,
            "source_turn_id": hit.get("source_turn_id") if hit else None,
            "duplicates": hit.get("duplicates", 1) if hit else None,
            "access_count": hit.get("access_count", 1) if hit else None,
        })
    return rows


def _this_turn_detail(prep, answer):
    latency = {}
    for stage in ("extraction", "ingest", "retrieval", "answer"):
        arr = state["latency_stats"].get(stage, [])
        if arr:
            latency[stage] = round(arr[-1], 1)
    return {
        "turn_id": prep["turn_id"],
        "query": prep["query"],
        "answer": answer,
        "extracted_facts": prep["extracted"],
        "retrieval": {
            "ranked": prep["ranks"],
            "injected_count": sum(1 for r in prep["ranks"] if r.get("injected")),
            "injected_tokens": prep["comparison"].get("optimized_context_tokens"),
            "injection_token_limit": prep["comparison"].get("injection_token_limit"),
            "engines": prep["comparison"].get("retrieval_engines") or [],
        },
        "storage": {
            "written": prep["storage"],
            "active_before": prep["store_before"]["active"],
            "active_after": len(state["active_memories"]),
            "pruned_before": prep["store_before"]["pruned"],
            "budget_evicted_before": prep["store_before"]["budget_evicted"],
            "budget_evicted_after": len(state["budget_evicted_memories"]),
        },
        "evictions": {
            "pruned_this_turn": [{"fact": p.get("fact"), "category": p.get("category", ""),
                                  "reason": p.get("prune_reason", "")} for p in prep["pruned_this_turn"]],
            "budget_this_turn": prep["budget_evictions"],
        },
        "latency": latency,
        "comparison": prep["comparison"],
    }


@app.post("/api/chat")
def chat_endpoint(msg: ChatMessage):
    guard = _begin_chat()
    if guard:
        return guard
    try:
        prep = _prepare_chat_turn(msg.user_message)
        answer = "".join(_llm_stream_answer(msg.user_message, state["last_context"].get("facts", [])))
        detail = _this_turn_detail(prep, answer)
        return {
            "turn_id": prep["turn_id"],
            "bot_response": answer,
            "optimized_answer": answer,
            "baseline_answer": "",
            "baseline_facts_seen": [],
            "active_memories": state["active_memories"],
            "pruned_memories": state["pruned_memories"],
            "budget_evicted_memories": state["budget_evicted_memories"],
            "retrieved_context": prep["ranks"],
            "this_turn": detail,
            "comparison": prep["comparison"],
        }
    finally:
        _end_chat()


@app.post("/api/chat/stream")
def chat_endpoint_stream(msg: ChatMessage):
    guard = _begin_chat()
    if guard:
        return JSONResponse(guard, status_code=409)

    def gen():
        try:
            yield f"data: {json.dumps({'type': 'stage', 'stage': 'extracting'})}\n\n"
            prep = _prepare_chat_turn(msg.user_message)
            yield f"data: {json.dumps({'type': 'stage', 'stage': 'answering'})}\n\n"
            chunks = []
            for tok in _llm_stream_answer(msg.user_message, state["last_context"].get("facts", [])):
                chunks.append(tok)
                yield f"data: {json.dumps({'type': 'token', 'text': tok})}\n\n"
            answer = "".join(chunks)
            detail = _this_turn_detail(prep, answer)
            yield f"data: {json.dumps({'type': 'done', 'current_turn': state['current_turn'], 'detail': detail, 'active_memories': state['active_memories'], 'pruned_memories': state['pruned_memories'], 'budget_evicted_memories': state['budget_evicted_memories'], 'comparison': _build_comparison()})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
        finally:
            _end_chat()

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/chat/baseline")
def chat_baseline():
    included, _, _, _ = _baseline_window()
    t0 = time.perf_counter()
    bl = _llm_window_replay(included)
    ms = (time.perf_counter() - t0) * 1000
    _record_latency("window_replay", ms)
    window_tokens = _content_tokens(bl.get("prompt_eval_count"), _calibrated_constants().get("C_replay"))
    return {
        "turn_id": state["current_turn"],
        "baseline_answer": bl.get("answer", ""),
        "facts_seen": bl.get("facts_seen", []),
        "window_turn_ids": [h["turn_id"] for h in included],
        "window_turn_count": len(included),
        "window_tokens": window_tokens,
        "window_eval_count": bl.get("eval_count"),
        "latency_ms": round(ms, 1),
        "error": bl.get("error"),
    }


@app.post("/api/demo/scenario")
def demo_scenario(payload: dict = None):
    payload = payload or {}
    turns = max(20, min(500, int(payload.get("turns", 150))))
    density_pct = max(0.0, min(100.0, float(payload.get("density_pct", 20.0))))
    planted_count = max(1, min(50, int(payload.get("planted_count", 5))))
    filler = str(payload.get("filler", "normal"))
    hard_cases = bool(payload.get("hard_cases", True))
    n_corrections = max(0, min(8, int(payload.get("n_corrections", 2)))) if hard_cases else 0
    n_negations = max(0, min(4, int(payload.get("n_negations", 1)))) if hard_cases else 0

    with _research_lock:
        if state["demo_job"].get("running"):
            return {"status": "busy", "detail": "A demo is already running", "job": state["demo_job"]}
        if state["research_busy"]:
            return {"status": "busy", "detail": "A research job is running; wait or cancel it before starting a demo"}

    _reset_demo_state()
    state["last_demo"] = {
        "turns": turns, "density_pct": density_pct, "planted_count": planted_count,
        "filler": filler, "mode": "llm", "hard_cases": hard_cases,
        "n_corrections": n_corrections, "n_negations": n_negations,
    }
    state["demo_job"] = {
        "running": False, "mode": "llm", "turns_total": turns, "turns_done": 0,
        "current_turn": None, "cancel_requested": False, "cancelled": False,
        "error": None, "started_at": None, "finished_at": None, "eta_s": None,
    }

    messages, ground_truth = _build_demo_messages(turns, density_pct, planted_count, filler,
                                                  hard_cases=hard_cases, n_corrections=n_corrections,
                                                  n_negations=n_negations)
    state["ground_truth"] = ground_truth
    state["last_demo"].update(
        ef_corrections=sum(1 for g in ground_truth if g.get("is_correction_target")),
        ef_negations=sum(1 for g in ground_truth if g.get("is_negation")),
    )
    state["demo_job"].update(running=True, turns_total=len(messages), started_at=time.time())
    threading.Thread(target=_run_llm_demo, args=(messages,), daemon=True).start()
    return {"status": "started", "job": state["demo_job"]}


@app.get("/api/demo/job")
def demo_job():
    j = state["demo_job"]
    if j.get("running") and j.get("started_at"):
        elapsed = time.time() - j["started_at"]
        done = j.get("turns_done", 0) or 0
        per = elapsed / max(1, done)
        j["eta_s"] = int(per * max(0, j.get("turns_total", 0) - done))
        j["elapsed_s"] = int(elapsed)
    return {"job": j}


@app.post("/api/demo/job/cancel")
def cancel_demo_job():
    if state["demo_job"].get("running"):
        state["demo_job"]["cancel_requested"] = True
        return {"status": "cancel_requested", "job": state["demo_job"]}
    return {"status": "not_running", "job": state["demo_job"]}


# =====================================================================
# Research job runner: dashboard-controlled pipelines
# =====================================================================

_RE_SEED_MARKER = re.compile(r"^\s*seed \d+", re.I)
_RESEARCH_LOG_CAP = 500

RESEARCH_PIPELINES = {
    "paper": {
        "label": "Paper pipeline (E1-E6 manifest)",
        "entry": "run_experiments.py",
        "llm_warn": "measured or write=extract triggers real LLM calls",
        "fields": {
            "seeds": {"type": "int", "min": 1, "max": 10, "default": 5, "help": "number of seeded replays"},
            "turns": {"type": "int", "min": 20, "max": 500, "default": 150},
            "density": {"type": "float", "min": 0.0, "max": 100.0, "default": 20.0, "help": "% of turns that re-mention planted facts"},
            "methods": {"type": "strlist", "default": "adaptive,sliding_window,memgpt_style,summarization_only,vanilla_rag"},
            "write": {"type": "choice", "choices": ["oracle", "extract"], "default": "oracle", "help": "oracle = held-fixed writer; extract = real LLM"},
            "measured": {"type": "bool", "default": False, "help": "measure tokens via Ollama (needs GPU)"},
            "model": {"type": "str", "default": "llama3.1:8b"},
            "embedding_model": {"type": "str", "default": ""},
            "label": {"type": "str", "default": ""},
            "conflict_density": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
            "negation_density": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
            "budget": {"type": "int", "min": 256, "max": 16384, "default": None},
            "top_k": {"type": "int", "min": 1, "max": 50, "default": None},
        },
    },
    "multimodel": {
        "label": "Multi-model comparison (writer-variance E1/E2)",
        "entry": "experiments/run_multi_model.py",
        "llm_warn": "runs extraction + replay for each model",
        "fields": {
            "seeds": {"type": "int", "min": 1, "max": 5, "default": 3},
            "turns": {"type": "int", "min": 20, "max": 500, "default": 150},
            "density": {"type": "float", "min": 0.0, "max": 100.0, "default": 20.0},
            "models": {"type": "strlist", "default": "llama3.1:8b,qwen2.5:7b,mistral:7b-instruct"},
            "methods": {"type": "strlist", "default": "adaptive,vanilla_rag,sliding_window"},
            "embeddings": {"type": "bool", "default": False},
        },
    },
    "e7": {
        "label": "C7 sensitivity sweep (retriever/budget geometry)",
        "entry": "experiments/e7_sensitivity_sweep.py",
        "fields": {
            "methods": {"type": "strlist", "default": "adaptive,vanilla_rag,sliding_window"},
            "seeds": {"type": "int", "min": 1, "max": 5, "default": 2},
            "measured": {"type": "bool", "default": False},
        },
    },
    "e8": {
        "label": "E8 needled-QA external benchmark",
        "entry": "experiments/e8_external_benchmark.py",
        "fields": {
            "methods": {"type": "strlist", "default": "adaptive,sliding_window,memgpt_style,summarization_only,vanilla_rag"},
            "n_per_category": {"type": "int", "min": 1, "max": 20, "default": 4},
            "idle_turns": {"type": "int", "min": 0, "max": 50, "default": 6},
            "embedding_model": {"type": "str", "default": "nomic-embed-text"},
        },
    },
    "e9": {
        "label": "E9 correction isolation probe",
        "entry": "experiments/e9_correction_isolation.py",
        "fields": {},
    },
    "e0": {
        "label": "E0 extraction quality vs gold set",
        "entry": "experiments/e0_extraction_quality.py",
        "fields": {
            "sample": {"type": "int", "min": 1, "max": 250, "default": None, "help": "null = full gold set"},
            "no_llm": {"type": "bool", "default": False, "help": "oracle sanity gate (no LLM)"},
            "model": {"type": "str", "default": "llama3.1:8b"},
            "no_paraphrase": {"type": "bool", "default": False},
        },
    },
    "qualitative": {
        "label": "Qualitative narrative report",
        "entry": "experiments/qualitative_report.py",
        "fields": {
            "manifest": {"type": "str", "default": "", "help": "empty = latest_manifest.json"},
            "seed": {"type": "int", "min": 1, "max": 999, "default": None},
        },
    },
    "artifacts": {
        "label": "Paper artifact generator (tables/figs/tex)",
        "entry": "experiments/make_paper_artifacts.py",
        "fields": {
            "manifest": {"type": "str", "default": ""},
            "check": {"type": "bool", "default": False},
        },
    },
    "paper_state": {
        "label": "Refresh claim coverage (paper_state.py)",
        "entry": "experiments/paper_state.py",
        "fields": {},
    },
    "stats": {
        "label": "Statistics self-test (bootstrap CI / Holm)",
        "entry": "experiments/statistics.py",
        "fields": {},
    },
    "gates": {
        "label": "All quick gates (stats→paper→e7→e8→e0→paper_state)",
        "entry": None,
        "fields": {},
    },
}

RESEARCH_GATES = [
    {"label": "statistics self-test", "argv": ["experiments/statistics.py"]},
    {"label": "paper quick gate", "argv": ["run_experiments.py", "--quick"]},
    {"label": "e7 sensitivity quick gate", "argv": ["experiments/e7_sensitivity_sweep.py", "--quick"]},
    {"label": "e8 benchmark quick gate", "argv": ["experiments/e8_external_benchmark.py", "--quick"]},
    {"label": "e9 correction isolation probe", "argv": ["experiments/e9_correction_isolation.py", "--quick"]},
    {"label": "e0 extraction quality quick gate", "argv": ["experiments/e0_extraction_quality.py", "--quick"]},
    {"label": "paper_state claim coverage", "argv": ["experiments/paper_state.py", "--quick"]},
]


def _coerce_run(pipeline, params):
    spec = RESEARCH_PIPELINES[pipeline]
    out = {}
    for field, meta in spec["fields"].items():
        raw = params.get(field, meta.get("default"))
        if raw is None:
            continue
        t = meta["type"]
        try:
            if t == "int":
                v = int(raw)
                if meta.get("min") is not None:
                    v = max(meta["min"], v)
                if meta.get("max") is not None:
                    v = min(meta["max"], v)
                out[field] = v
            elif t == "float":
                v = float(raw)
                if meta.get("min") is not None:
                    v = max(meta["min"], v)
                if meta.get("max") is not None:
                    v = min(meta["max"], v)
                out[field] = v
            elif t == "bool":
                out[field] = bool(raw) if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")
            elif t == "strlist":
                if isinstance(raw, (list, tuple)):
                    out[field] = ",".join(str(x).strip() for x in raw if str(x).strip())
                else:
                    out[field] = ",".join(x.strip() for x in str(raw).split(",") if x.strip())
            elif t == "choice":
                if raw in meta["choices"]:
                    out[field] = raw
            else:
                out[field] = str(raw)
        except (TypeError, ValueError):
            continue
    return out


def _build_args(pipeline, params):
    spec = RESEARCH_PIPELINES[pipeline]
    entry = spec["entry"]
    if not entry:
        return []
    argv = [entry]
    for field, meta in spec["fields"].items():
        val = params.get(field)
        if val is None:
            continue
        flag = "--" + field.replace("_", "-")
        if meta["type"] == "bool":
            if val:
                argv.append(flag)
        elif meta["type"] == "strlist":
            argv += [flag, str(val)]
        else:
            argv += [flag, str(val)]
    return argv


def _sanitize_job(j):
    s = dict(j)
    s.pop("proc", None)
    s["log"] = list(s.get("log", []))[-80:]
    s["cancel_requested"] = bool(s.get("cancel_requested"))
    return s


def _new_research_job(pipeline, params):
    spec = RESEARCH_PIPELINES[pipeline]
    argv = _build_args(pipeline, params)
    return {
        "id": uuid.uuid4().hex[:8],
        "pipeline": pipeline,
        "label": spec["label"],
        "params": params,
        "cmd_human": (" ".join(argv) if argv else "chained quick gates (%d stages)" % len(RESEARCH_GATES)),
        "status": "queued",
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "elapsed_s": None,
        "exit_code": None,
        "error": None,
        "log": [],
        "stage_index": 0,
        "stage_total": len(RESEARCH_GATES) if pipeline == "gates" else 1,
        "stage_label": "queued",
        "progress_pct": None,
        "progress_label": "",
        "cancel_requested": False,
        "proc": None,
    }


def _cap_job_log(job):
    if len(job["log"]) > _RESEARCH_LOG_CAP + 50:
        del job["log"][:len(job["log"]) - _RESEARCH_LOG_CAP]


def _run_research_job(job):
    stages = RESEARCH_GATES if job["pipeline"] == "gates" else [
        {"label": job["label"], "argv": _build_args(job["pipeline"], job["params"])}
    ]
    with _research_lock:
        state["research_busy"] = True
    job["status"] = "running"
    job["started_at"] = time.time()
    tots = job["stage_total"]
    for i, stage in enumerate(stages, start=1):
        if job.get("cancel_requested"):
            break
        job["stage_index"] = i
        job["stage_label"] = "%d/%d: %s" % (i, tots, stage["label"])
        job["progress_pct"] = None
        job["progress_label"] = "spawning"
        job["log"].append("== [%d/%d] %s :: %s" % (i, tots, stage["label"], " ".join(stage["argv"])))
        _cap_job_log(job)
        try:
            proc = subprocess.Popen(
                [sys.executable] + stage["argv"],
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                bufsize=1,
            )
        except Exception as e:
            job["error"] = "spawn failed: %s" % e
            job["log"].append("[ERROR] %s" % job["error"])
            break
        job["proc"] = proc
        seen_seeds = 0
        try:
            for line in proc.stdout:
                _cap_job_log(job)
                job["log"].append(line.rstrip("\n"))
                if _RE_SEED_MARKER.match(line):
                    seen_seeds += 1
                    total_seeds = job["params"].get("seeds")
                    job["progress_label"] = "seed %d" % seen_seeds
                    if total_seeds and seen_seeds <= int(total_seeds):
                        job["progress_pct"] = int(min(95, seen_seeds * 90 / int(total_seeds)))
                if job.get("cancel_requested"):
                    break
            if job.get("cancel_requested"):
                try:
                    os.killpg(os.getpgid(proc.pid), 9)
                except Exception:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
            proc.wait()
        except Exception as e:
            job["error"] = str(e)
        finally:
            job["proc"] = None
        job["exit_code"] = proc.returncode if proc and proc.returncode is not None else None
        if job.get("cancel_requested"):
            job["status"] = "cancelled"
            job["log"].append("cancel requested; process terminated")
            break
        if proc.returncode != 0:
            job["status"] = "error"
            job["error"] = job.get("error") or ("exit code %s — see log" % proc.returncode)
            job["log"].append("[ERROR] %s" % job["error"])
            break
        job["progress_label"] = "done"
        job["progress_pct"] = 100
    else:
        if not job.get("cancel_requested"):
            job["status"] = "success"
            job["log"].append("[SUCCESS] %s finished" % job["label"])
            job["progress_pct"] = 100
    job["finished_at"] = time.time()
    job["elapsed_s"] = int(job["finished_at"] - (job["started_at"] or job["finished_at"]))
    state["research_history"].append(_sanitize_job(job))
    state["research_history"] = state["research_history"][-100:]
    try:
        JOBS_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        JOBS_HISTORY_FILE.write_text(json.dumps(state["research_history"], indent=2, default=str))
    except Exception:
        pass
    try:
        state["research_jobs"] = [j for j in state["research_jobs"] if j["id"] != job["id"]]
    finally:
        with _research_lock:
            state["research_busy"] = False


@app.get("/api/research/pipelines")
def research_pipelines():
    out = {}
    for name, spec in RESEARCH_PIPELINES.items():
        out[name] = {
            "label": spec["label"],
            "llm_warn": spec.get("llm_warn"),
            "fields": {k: {kk: vv for kk, vv in v.items()} for k, v in spec["fields"].items()},
            "is_gated": name == "gates",
        }
    return out


@app.get("/api/research/jobs")
def research_jobs_list():
    active = [_sanitize_job(j) for j in state["research_jobs"]]
    return {"busy": state["research_busy"], "active": active, "history": state["research_history"]}


class _ResearchRunPayload(BaseModel):
    pipeline: str
    params: dict = {}


@app.post("/api/research/run")
def research_run(payload: _ResearchRunPayload):
    p = payload.pipeline
    if p not in RESEARCH_PIPELINES:
        return {"status": "error", "detail": "Unknown pipeline '%s'. Allowed: %s" % (p, sorted(RESEARCH_PIPELINES))}
    with _research_lock:
        if state["demo_job"].get("running"):
            return {"status": "busy", "detail": "the live demo is running; cancel it before launching a research job"}
        if state["research_busy"]:
            return {"status": "busy", "detail": "a research job is already running"}
        state["research_busy"] = True
    try:
        params = _coerce_run(p, payload.params or {})
        job = _new_research_job(p, params)
    except Exception as e:
        with _research_lock:
            state["research_busy"] = False
        return {"status": "error", "detail": "failed to build job: %s" % e}
    state["research_jobs"].append(job)
    threading.Thread(target=_run_research_job, args=(job,), daemon=True).start()
    return {"status": "started", "job": _sanitize_job(job)}


@app.post("/api/research/jobs/{job_id}/cancel")
def research_cancel(job_id: str):
    for j in state["research_jobs"]:
        if j["id"] == job_id:
            if j["status"] == "queued":
                j["status"] = "cancelled"
                j["finished_at"] = time.time()
                j["elapsed_s"] = 0
                j["log"].append("cancelled before start")
                state["research_history"].append(_sanitize_job(j))
                state["research_history"] = state["research_history"][-100:]
                state["research_jobs"] = [x for x in state["research_jobs"] if x["id"] != job_id]
                with _research_lock:
                    state["research_busy"] = False
                try:
                    JOBS_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
                    JOBS_HISTORY_FILE.write_text(json.dumps(state["research_history"], indent=2, default=str))
                except Exception:
                    pass
                return {"status": "cancelled", "job": _sanitize_job(j)}
            if j["status"] == "running":
                j["cancel_requested"] = True
                return {"status": "cancel_requested", "job": _sanitize_job(j)}
            return {"status": "finished", "job": _sanitize_job(j)}
    return {"status": "not_found"}


@app.get("/api/research/jobs/{job_id}")
def research_job_detail(job_id: str, lines: int = 500):
    lines = max(1, min(2000, int(lines)))
    for j in state["research_jobs"]:
        if j["id"] == job_id:
            return {"job": _sanitize_job(j), "log": list(j["log"])[-lines:]}
    for h in state["research_history"]:
        if h["id"] == job_id:
            return {"job": h, "log": h.get("log", [])[-lines:]}
    return {"detail": "job not found"}


def _safe_read_json(path):
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except Exception:
        return None


def _paper_board(m):
    if not m:
        return None
    agg = m.get("aggregate", {})
    methods = {}
    for name, a in agg.items():
        if name == "multiple_comparisons" or not isinstance(a, dict) or "mean_E1_proposed_tokens" not in a:
            continue
        methods[name] = {
            "mean_E1_proposed_tokens": a.get("mean_E1_proposed_tokens"),
            "mean_E1_baseline_tokens": a.get("mean_E1_baseline_tokens"),
            "E1_proposed_pooled_ci95": a.get("E1_proposed_pooled_ci95"),
            "cohens_d_paired_vs_baseline": a.get("cohens_d_paired_vs_baseline"),
            "mean_E2_proposed_recall": a.get("mean_E2_proposed_recall"),
            "mean_E2_baseline_recall": a.get("mean_E2_baseline_recall"),
            "mean_E2_proposed_forgetting_precision": a.get("mean_E2_proposed_forgetting_precision"),
            "E6": a.get("E6"),
        }
    mc = agg.get("multiple_comparisons", {})
    hard = None
    e4_hard = None
    per_cat = None
    if m.get("results"):
        r0 = m["results"][0]
        mm = r0.get("methods", {}).get("adaptive")
        if mm:
            e2 = mm.get("E2_memory_accuracy") or {}
            per_cat = e2.get("per_category_recall")
            hard = {
                "trap_recall": e2.get("trap_recall"),
                "correction_recall": e2.get("correction_recall"),
                "negation_recall": e2.get("negation_recall"),
                "wrongly_retained_after_correction": e2.get("wrongly_retained_after_correction"),
                "per_category_recall": per_cat,
            }
            e4 = mm.get("E4_needle_in_haystack") or {}
            e4_hard = e4.get("hard_case_accuracy_by_distance")
    return {
        "generated_at": m.get("generated_at"),
        "config": m.get("config"),
        "token_source": m.get("token_source"),
        "metric_source": m.get("metric_source"),
        "scoring_and_correction_fix_applied": m.get("pipeline_fix_version", 0) >= 2,
        "pipeline_fix_version": m.get("pipeline_fix_version"),
        "methods": methods,
        "hard_case": hard,
        "e4_hard_case": e4_hard,
        "multiple_comparisons": mc,
    }


def _models_board(ms):
    if not ms:
        return None
    rows = []
    for key, d in (ms.get("per_model") or {}).items():
        if not isinstance(d, dict):
            continue
        rows.append({
            "key": key, "model": d.get("model"), "seeds": d.get("seeds"),
            "mean_E1_proposed_tokens": d.get("mean_E1_proposed_tokens"),
            "mean_E2_recall": d.get("mean_E2_recall"),
            "mean_extraction_ms": d.get("mean_extraction_ms"),
            "e1_ci95": d.get("e1_ci95"),
        })
    return {"config": ms.get("config"), "note": ms.get("note"), "rows": rows}


def _e0_board(e0):
    if not e0:
        return None
    return {
        "turns_scored": e0.get("turns_scored"), "write_mode": e0.get("write_mode"),
        "model": e0.get("model"),
        "exact_or_paraphrase": e0.get("exact_or_paraphrase"),
        "plus_category_agreement": e0.get("plus_category_agreement"),
        "per_category": e0.get("per_category"), "per_property": e0.get("per_property"),
        "missed_examples": (e0.get("missed_examples") or [])[:5],
    }


def _e8_board(e8):
    if not e8:
        return None
    summary = e8.get("summary") or {}
    methods = []
    for name, d in summary.items():
        if not isinstance(d, dict):
            continue
        by_cat = d.get("by_category") or {}
        cat_rows = {}
        n_items = 0
        for k, v in by_cat.items():
            if not isinstance(v, dict):
                continue
            cat_rows[k] = {"n": v.get("n"), "recall": v.get("recall")}
            n_items += v.get("n", 0) or 0
        methods.append({
            "method": name, "overall_recall": d.get("overall_recall"),
            "mean_injected_tokens": d.get("mean_injected_tokens"),
            "n_items": n_items,
            "by_category": cat_rows,
        })
    return {
        "suite_version": e8.get("suite_version"), "run_at": e8.get("run_at"),
        "idle_turns": e8.get("idle_turns"), "tag": e8.get("tag"),
        "n_items": len(e8.get("per_item") or []), "methods": methods,
    }


def _e7_board(e7):
    if not e7:
        return None
    return {
        "baseline_point": e7.get("baseline_point"), "methods": e7.get("methods"),
        "seeds": e7.get("seeds"), "n_rows": len(e7.get("rows", [])),
        "rows": e7.get("rows", []),
    }


def _claims_board(ps):
    if not ps:
        return None
    cov = ps.get("coverage") or []
    return {
        "generated_at": ps.get("generated_at"),
        "supported": sum(1 for c in cov if c.get("supported")),
        "total": len(cov),
        "coverage": cov,
        "metric_source_sentence": ps.get("metric_source_sentence"),
    }


def _live_board(lm):
    if not lm:
        return None
    e1 = lm.get("E1_token_efficiency") or {}
    e2 = lm.get("E2_memory_accuracy") or {}
    e3 = lm.get("E3_latency") or {}
    e4 = lm.get("E4_needle_in_haystack") or {}
    e6 = lm.get("E6_power_check") or {}
    return {
        "model": lm.get("model_under_test"),
        "budget": lm.get("max_context_tokens"),
        "metric_source": lm.get("metric_source"),
        "E1": {
            "proposed_mean_tokens": e1.get("proposed_mean_tokens"),
            "baseline_mean_tokens": e1.get("baseline_mean_tokens"),
            "n": e1.get("n"), "p_value": e1.get("p_value"),
            "significant": e1.get("significant"),
            "per_turn": e1.get("per_turn", []),
        },
        "E2": {"proposed_recall": e2.get("proposed_positive_recall"),
               "baseline_recall": e2.get("baseline_positive_recall"),
               "forgetting_precision": e2.get("proposed_forgetting_precision"),
               "hard_case": e2.get("hard_case")},
        "E3": {k: e3.get(k) for k in (
            "extraction_ms", "scoring_ms", "fact_token_measurement_ms", "compression_ms",
            "decay_ms", "budget_evict_ms", "retrieval_ms", "embedding_ms",
            "window_replay_ms", "answer_ms")},
        "E4": {"distances": e4.get("distances"), "proposed": e4.get("proposed_accuracy"),
               "baseline": e4.get("baseline_accuracy")},
        "E5": lm.get("E5_ablations"),
        "E6": {"cohens_d": e6.get("observed_cohens_d"), "required_n": e6.get("required_n_per_group"),
               "n_actual": e6.get("n_actual"), "target_power": e6.get("target_power")},
    }


@app.get("/api/research/board")
def research_board():
    qual = _safe_read_json(RESEARCH_DIR / "qualitative_report.json")
    return {
        "paper": _paper_board(_safe_read_json(RESEARCH_DIR / "latest_manifest.json")),
        "live": _live_board(_safe_read_json(RESEARCH_DIR / "live_manifest.json")),
        "models": _models_board(_safe_read_json(RESEARCH_DIR / "models" / "models_summary.json")),
        "e0": _e0_board(_safe_read_json(RESEARCH_DIR / "e0_extraction_quality.json")),
        "e8": _e8_board(_safe_read_json(RESEARCH_DIR / "e8_e8-v1.json")),
        "e7": _e7_board(_safe_read_json(RESEARCH_DIR / "e7_sweep.json")),
        "claims": _claims_board(_safe_read_json(RESEARCH_DIR / "paper_state.json")),
        "qualitative": qual,
        "jobs_busy": state["research_busy"],
    }


@app.get("/api/timeline")
def get_timeline():
    tl = state["timeline"]
    return {"timeline": tl[-300:], "total": len(tl)}


@app.get("/api/timeline/full")
def get_timeline_full(offset: int = 0, limit: int = 50):
    tl = state["timeline"]
    offset = max(0, int(offset))
    limit = max(1, min(200, int(limit)))
    return {"total": len(tl), "offset": offset, "limit": limit, "entries": tl[offset:offset + limit]}


@app.get("/api/comparison/full")
def comparison_full():
    return _build_comparison(full=True)


@app.get("/api/experiments/latest")
def experiments_latest():
    p = BASE_DIR / "experiments/results/live_manifest.json"
    if p.exists():
        return json.loads(p.read_text())
    p = BASE_DIR / "experiments/results/latest_manifest.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"detail": "No experiments have been run yet.", "found": False}


@app.get("/api/state")
def get_state():
    return {
        "current_turn": state["current_turn"],
        "active_memories": state["active_memories"],
        "pruned_memories": state["pruned_memories"],
        "budget_evicted_memories": state["budget_evicted_memories"],
        "ground_truth_count": len(state["ground_truth"]),
        "ground_truth": [{
            "source_turn": g["source_turn"],
            "fact": g["fact"],
            "category": g["category"],
            "is_trap": g.get("is_trap", False),
            "is_correction_target": g.get("is_correction_target", False),
            "is_negation": g.get("is_negation", False),
            "planted": g.get("planted", False),
            "hard_case": g.get("hard_case", False),
            "supersedes_turn": g.get("supersedes_turn"),
            "superseded_by": g.get("superseded_by"),
            "obsolete": g.get("obsolete", False),
        } for g in state["ground_truth"]],
        "comparison": _build_comparison(),
        "settings": settings,
        "demo_params": state.get("last_demo", {}),
        "demo_job": state["demo_job"],
        "baseline_replay": state["baseline_replay"][-200:],
        "baseline_replay_count": len(state["baseline_replay"]),
    }


@app.get("/api/models")
def get_models():
    try:
        tags = requests.get(f"{ollama_base}/api/tags", timeout=5).json().get("models", [])
    except Exception:
        tags = []
    models = []
    for m in tags:
        name = m["name"]
        if "embed" in name.lower():
            continue
        details = m.get("details", {})
        models.append({
            "name": name,
            "params": details.get("parameter_size", ""),
            "family": (details.get("families") or [""])[0],
        })
    return {"models": models, "active": settings["active_model"]}


@app.post("/api/models/select")
def select_model(payload: dict):
    name = payload.get("model")
    if not name:
        return {"status": "error", "detail": "no model"}
    prev = settings["active_model"]
    if prev != name:
        try:
            requests.post(f"{ollama_base}/api/generate",
                          json={"model": prev, "prompt": "x", "keep_alive": 0},
                          timeout=10)
        except Exception:
            pass
    settings["active_model"] = name
    extractor.model = name
    _token_constants.update(model=None)
    _fact_token_cache.clear()
    _persist_settings()
    return {"status": "ok", "active": name, "unloaded_previous": prev if prev != name else None}


@app.get("/api/settings")
def get_settings():
    return settings


class SettingsPayload(BaseModel):
    max_context_tokens: int = None
    top_k: int = None
    similarity_threshold: float = None
    pruning_threshold: float = None
    enable_compression: bool = None
    injection_token_limit: int = None


@app.post("/api/settings")
def update_settings(payload: SettingsPayload):
    if payload.max_context_tokens is not None:
        settings["max_context_tokens"] = max(256, int(payload.max_context_tokens))
    if payload.top_k is not None:
        settings["top_k"] = max(1, int(payload.top_k))
        retriever.top_k = settings["top_k"]
    if payload.similarity_threshold is not None:
        settings["similarity_threshold"] = float(payload.similarity_threshold)
        retriever.sim_threshold = settings["similarity_threshold"]
    if payload.pruning_threshold is not None:
        settings["pruning_threshold"] = float(payload.pruning_threshold)
        decay_engine.pruning_threshold = settings["pruning_threshold"]
    if payload.enable_compression is not None:
        settings["enable_compression"] = bool(payload.enable_compression)
    if payload.injection_token_limit is not None:
        settings["injection_token_limit"] = max(0, int(payload.injection_token_limit))
    _persist_settings()
    return {"status": "ok", "settings": settings, "comparison": _build_comparison()}


@app.get("/api/gpu")
def gpu_endpoint():
    return {"vram": _vram_block(
        _build_comparison()["baseline_context_tokens"],
        _build_comparison()["active_memory_tokens"],
    )}


@app.post("/api/experiments/run")
def run_experiments():
    from experiments.live import compute_all_json

    if state["demo_job"].get("running"):
        return {"status": "error", "detail": "Demo is still running; wait for it to finish first"}
    if not state["raw_history"]:
        return {"status": "error", "detail": "No demo loaded yet. Run a demo first."}

    stream = [{
        "turn_id": h["turn_id"],
        "user": h["user"],
        "tokens": h.get("tokens"),
        "facts": [dict(f) for f in h.get("facts", [])],
    } for h in state["raw_history"]]
    ground_truth = state["ground_truth"]

    results = compute_all_json(state, settings, stream, ground_truth, scorer, decay_engine,
                               retriever, _build_comparison, fact_tokens=_lookup_or_measure,
                               embed_fn=_embed_texts, embedding_model=embed_model,
                               injection_token_limit=settings["injection_token_limit"])

    out_path = BASE_DIR / "experiments/results/live_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    with open(out_path, "w") as f:
        _json.dump(results, f, indent=2, default=str)
    return results


@app.get("/api/ethics/privacy_export")
def privacy_export():
    return {
        "user_data_retention_policy": "Zero-Knowledge Local Storage",
        "active_memories": state["active_memories"],
        "pruned_memories": state["pruned_memories"],
    }


@app.delete("/api/ethics/purge")
def privacy_purge():
    state["active_memories"].clear()
    state["pruned_memories"].clear()
    state["raw_history"].clear()
    state["active_token_trace"].clear()
    state["inject_token_trace"].clear()
    state["ground_truth"].clear()
    state["latency_stats"] = {}
    state["timeline"].clear()
    state["last_evicted_ids"] = []
    state["current_turn"] = 0
    state["last_context"] = {}
    state["baseline_replay"] = []
    state["last_budget_evictions"] = []
    state["budget_evicted_memories"] = []
    state["chat_busy"] = False
    return {"status": "All stored user memory structures successfully purged."}


@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    with open(BASE_DIR / "ui/index.html", "r") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "9002")))
