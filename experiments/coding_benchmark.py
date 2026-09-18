"""E17 common coding benchmark harness (Phase 13 / D34).

One path for every method:

    prepare(history, budget)  -> historical memory representation
    retrieve(prepared, task)  -> historical-context string (<= budget)
    run_coding_attempt(...)   -> patch + hidden-test result + metrics

The benchmark owns workspace reset, prompt construction, model invocation, patch
application, hidden tests, metrics and results writing. Methods own ONLY
historical-context construction and retrieval, so the only controlled difference
between arms is the memory mechanism.

Editing format: the coding model returns complete-file blocks
(``### FILE: path`` / ``### END FILE``) that the harness writes verbatim; the
resulting working-tree change is captured as a unified diff for provenance and
determinism checks. A unified-diff fallback is kept for robustness. This avoids
penalising a memory method for the model's unreliable multi-hunk diff formatting.

Everything is deterministic and injectable: ``embed_fn`` and ``coder`` (the
generating model wrapper) can be replaced with fakes so unit tests never touch
Ollama.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Tuple

from baselines.llm_summarization import LLMSummarizer
from baselines.raw_clipped import build_context as raw_clipped_build_context
from data.coding_task_suite import CodingTask
from memory_optimizer.tokenizer import TOKENIZER_NAME, count_tokens

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HISTORICAL_BUDGETS = [256, 512, 1024]
METHOD_NAMES = ["raw_clipped", "sliding_window", "llm_summarization",
                "vanilla_rag", "adaptive"]
MODEL_CONTEXT_TOKENS = 8192
CODING_TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 700
MAX_ATTEMPTS = 2

INSTRUCTIONS = (
    "You are modifying the current repository to implement the requested task.\n"
    "You have access to the current workspace.\n"
    "Historical engineering context is provided separately below.\n"
    "Implement the task without modifying or weakening tests.\n"
    "Return the COMPLETE final contents of every file you change.\n"
    "For each changed file emit exactly this block:\n"
    "### FILE: <relative/path>\n"
    "<complete file contents>\n"
    "### END FILE\n"
    "Do not return a diff and do not return partial snippets.\n"
)


# ---------------------------------------------------------------------------
# History bundle + prepared memory
# ---------------------------------------------------------------------------

@dataclass
class HistoryBundle:
    task_id: str
    turns: List[Dict]
    texts: List[str]


@dataclass
class PreparedMemory:
    method: str
    memory_text: str = ""
    memory_ids: set = field(default_factory=set)
    memory_tokens: int = 0
    build: Dict = field(default_factory=dict)
    payload: object = None


# ---------------------------------------------------------------------------
# Method protocol + shared method internals
# ---------------------------------------------------------------------------

class HistoricalContextMethod(Protocol):
    name: str

    def prepare(self, history: HistoryBundle, historical_budget: int,
                *, seed: int) -> PreparedMemory:
        ...

    def retrieve(self, prepared: PreparedMemory, task_prompt: str,
                 historical_budget: int) -> str:
        ...


class _BaseMethod:
    name = "base"

    def __init__(self, model: str = "llama3.1:8b",
                 endpoint: str = "http://localhost:11434",
                 embed_fn=None, embedding_model: str = "nomic-embed-text"):
        self.model = model
        self.endpoint = endpoint
        self.embed_fn = embed_fn
        self.embedding_model = embedding_model
        self.last_context_ids: set = set()
        self.last_context_latency_ms: float = 0.0

    def prepare(self, history, historical_budget, *, seed):  # pragma: no cover
        raise NotImplementedError

    def retrieve(self, prepared, task_prompt, historical_budget):  # pragma: no cover
        raise NotImplementedError


class RawClippedMethod(_BaseMethod):
    name = "raw_clipped"

    def prepare(self, history: HistoryBundle, historical_budget: int, *, seed: int):
        full = "\n".join(history.texts)
        return PreparedMemory(
            method=self.name,
            memory_text=full,
            memory_tokens=count_tokens(full),
            payload={"texts": list(history.texts)},
            build={"memory_build_calls": 0, "memory_build_input_tokens": 0,
                   "memory_build_output_tokens": 0, "memory_build_latency_ms": 0.0},
        )

    def retrieve(self, prepared, task_prompt, historical_budget):
        t0 = time.perf_counter()
        text = raw_clipped_build_context(prepared.payload["texts"], historical_budget)
        self.last_context_ids = set()
        self.last_context_latency_ms = (time.perf_counter() - t0) * 1000
        return text


class SlidingWindowMethod(_BaseMethod):
    name = "sliding_window"

    def prepare(self, history: HistoryBundle, historical_budget: int, *, seed: int):
        from baselines.sliding_window import SlidingWindowBaseline
        t0 = time.perf_counter()
        self._baseline = SlidingWindowBaseline(
            budget=historical_budget, top_k=5, embed_fn=self.embed_fn,
            fact_tokens=count_tokens, embedding_model=self.embedding_model)
        self._baseline.reset()
        for turn in history.turns:
            self._baseline.observe(turn)
        held = self._baseline.held_facts()
        memory_text = "\n".join(e.get("fact", "") for e in held)
        return PreparedMemory(
            method=self.name, memory_text=memory_text,
            memory_tokens=count_tokens(memory_text) if memory_text else 0,
            payload={"turns": history.turns},
            build={"memory_build_calls": 0, "memory_build_input_tokens": 0,
                   "memory_build_output_tokens": 0,
                   "memory_build_latency_ms": (time.perf_counter() - t0) * 1000},
        )

    def retrieve(self, prepared, task_prompt, historical_budget):
        t0 = time.perf_counter()
        entries = self._baseline.retrieve(task_prompt)
        text = "\n".join(e.get("fact", "") for e in entries)
        self.last_context_ids = set()
        self.last_context_latency_ms = (time.perf_counter() - t0) * 1000
        return text


class LLMSummarizationMethod(_BaseMethod):
    name = "llm_summarization"

    def __init__(self, *args, summarizer_generate_fn=None,
                 update_interval=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.summarizer_generate_fn = summarizer_generate_fn
        self.update_interval = update_interval

    def prepare(self, history: HistoryBundle, historical_budget: int, *, seed: int):
        t0 = time.perf_counter()
        summarizer = LLMSummarizer(
            model=self.model, endpoint=self.endpoint,
            update_interval=self.update_interval,
            generate_fn=self.summarizer_generate_fn)
        res = summarizer.build_summary(history.texts, historical_budget)
        return PreparedMemory(
            method=self.name, memory_text=res["summary"],
            memory_tokens=res["summary_tokens"],
            build={
                "memory_build_calls": res["summary_update_calls"],
                "memory_build_input_tokens": res["summary_input_tokens"],
                "memory_build_output_tokens": res["summary_output_tokens"],
                "memory_build_latency_ms": res["summary_latency_ms"],
                "summary_update_calls": res["summary_update_calls"],
            },
            payload={"summary": res["summary"]},
        )

    def retrieve(self, prepared, task_prompt, historical_budget):
        t0 = time.perf_counter()
        self.last_context_ids = set()
        self.last_context_latency_ms = (time.perf_counter() - t0) * 1000
        return prepared.payload["summary"]


class VanillaRAGMethod(_BaseMethod):
    name = "vanilla_rag"

    def prepare(self, history: HistoryBundle, historical_budget: int, *, seed: int):
        from baselines.vanilla_rag import VanillaRAGBaseline
        t0 = time.perf_counter()
        self._baseline = VanillaRAGBaseline(
            budget=historical_budget, top_k=5, embed_fn=self.embed_fn,
            fact_tokens=count_tokens, embedding_model=self.embedding_model)
        self._baseline.reset()
        for turn in history.turns:
            self._baseline.observe(turn)
        store = self._baseline.held_facts()
        ids = {f.get("fact_id") for f in store if f.get("fact_id")}
        text = "\n".join(f.get("fact", "") for f in store)
        return PreparedMemory(
            method=self.name, memory_text=text, memory_ids=ids,
            memory_tokens=count_tokens(text) if text else 0,
            payload={"store": self._baseline.store},
            build={"memory_build_calls": 0, "memory_build_input_tokens": 0,
                   "memory_build_output_tokens": 0,
                   "memory_build_latency_ms": (time.perf_counter() - t0) * 1000},
        )

    def retrieve(self, prepared, task_prompt, historical_budget):
        t0 = time.perf_counter()
        store = prepared.payload["store"]
        ranked = self._baseline.rank_by_similarity(task_prompt, store, top_k=len(store))
        selected = self._baseline.fit_entries(ranked, limit=historical_budget)
        text = "\n".join(e.get("fact", "") for e in selected)
        self.last_context_ids = {e.get("fact_id") for e in selected if e.get("fact_id")}
        self.last_context_latency_ms = (time.perf_counter() - t0) * 1000
        return text


class AdaptiveMethod(_BaseMethod):
    name = "adaptive"

    def __init__(self, *args, retention_mode: str = "dual_score", **kwargs):
        super().__init__(*args, **kwargs)
        self.retention_mode = retention_mode

    def _settings(self, historical_budget: int) -> Dict:
        from experiments.paper import load_settings
        settings = load_settings({
            "max_context_tokens": historical_budget,
            "injection_token_limit": historical_budget,
            "memory_store_token_budget": max(4096, historical_budget * 4),
            "retention_mode": self.retention_mode,
        })
        # E16/E17 run the production default with correction protection on.
        settings.setdefault("retention", {})["protect_corrections"] = True
        return settings

    def prepare(self, history: HistoryBundle, historical_budget: int, *, seed: int):
        from experiments.paper import replay_adaptive
        from memory_optimizer.compression import MemoryCompressor
        from memory_optimizer.decay import CategoryDecayEngine
        from memory_optimizer.retrieval import MemoryRetriever
        from memory_optimizer.scoring import ImportanceScorer
        settings = self._settings(historical_budget)
        t0 = time.perf_counter()
        scorer = ImportanceScorer(weights=settings["scoring_weights"])
        decay = CategoryDecayEngine(
            lambdas=settings["decay_lambdas"],
            pruning_threshold=float(settings["pruning"]["threshold"]))
        retriever = MemoryRetriever(
            top_k=int(settings["top_k"]),
            sim_threshold=float(settings.get("similarity_threshold", 0.35)),
            embed_fn=self.embed_fn, embedding_model=self.embedding_model)
        compressor = MemoryCompressor()
        rep = replay_adaptive(history.turns, settings, scorer, decay, retriever,
                              compressor, fact_tokens=count_tokens,
                              embed_fn=self.embed_fn,
                              embedding_model=self.embedding_model)
        memories = rep["memories"]
        mem_text = "\n".join(m.get("fact", "") for m in memories)
        ids = {m.get("fact_id") for m in memories if m.get("fact_id")}
        return PreparedMemory(
            method=self.name, memory_text=mem_text, memory_ids=ids,
            memory_tokens=count_tokens(mem_text) if mem_text else 0,
            payload={"memories": memories, "retriever": retriever,
                     "settings": settings},
            build={"memory_build_calls": 0, "memory_build_input_tokens": 0,
                   "memory_build_output_tokens": 0,
                   "memory_build_latency_ms": (time.perf_counter() - t0) * 1000},
        )

    def retrieve(self, prepared, task_prompt, historical_budget):
        memories = prepared.payload["memories"]
        retriever = prepared.payload["retriever"]
        settings = prepared.payload["settings"]
        t0 = time.perf_counter()
        injected = retriever.retrieve(
            task_prompt, memories, current_turn=10 ** 9,
            token_limit=historical_budget, fact_tokens=count_tokens)
        text = "\n".join(m.get("fact", "") for m in injected)
        self.last_context_ids = {m.get("fact_id") for m in injected
                                 if m.get("fact_id")}
        self.last_context_latency_ms = (time.perf_counter() - t0) * 1000
        return text


class NoHistoryMethod(_BaseMethod):
    """Diagnostic: the coding model receives NO historical context."""

    name = "no_history"

    def prepare(self, history, historical_budget, *, seed):
        return PreparedMemory(method=self.name, memory_text="", memory_ids=set(),
                              memory_tokens=0, payload={},
                              build={"memory_build_calls": 0})

    def retrieve(self, prepared, task_prompt, historical_budget):
        self.last_context_ids = set()
        self.last_context_latency_ms = 0.0
        return ""


class FullContextMethod(_BaseMethod):
    """Diagnostic upper bound: the ENTIRE raw history (ignores the budget).

    Labelled explicitly as a context-unconstrained upper bound; never compared
    as if it respected the fixed historical-context budget.
    """

    name = "full_context"

    def prepare(self, history, historical_budget, *, seed):
        full = "\n".join(history.texts)
        return PreparedMemory(method=self.name, memory_text=full,
                              memory_tokens=count_tokens(full), payload={},
                              build={"memory_build_calls": 0})

    def retrieve(self, prepared, task_prompt, historical_budget):
        self.last_context_ids = set()
        self.last_context_latency_ms = 0.0
        return prepared.memory_text


class DirectHistoryMethod(_BaseMethod):
    """Sanity/oracle diagnostic: inject the required gold facts directly.

    Labelled oracle: it uses gold fact text and is an upper bound on what any
    historical-context method could supply, not a competitor.
    """

    name = "direct_history"

    def __init__(self, *args, task: CodingTask = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._task = task

    def prepare(self, history, historical_budget, *, seed):
        if self._task is None:
            raise ValueError("DirectHistoryMethod requires task=")
        facts = [f for f in self._task.gold_facts if not f.obsolete]
        text = "\n".join(f.text for f in facts)
        return PreparedMemory(method=self.name, memory_text=text,
                              memory_ids={f.fact_id for f in facts},
                              memory_tokens=count_tokens(text) if text else 0,
                              payload={}, build={"memory_build_calls": 0})

    def retrieve(self, prepared, task_prompt, historical_budget):
        self.last_context_ids = set(prepared.memory_ids)
        self.last_context_latency_ms = 0.0
        return prepared.memory_text


def build_method(name: str, *, model: str, endpoint: str, embed_fn,
                 embedding_model: str, summarizer_generate_fn=None,
                 retention_mode: str = "dual_score",
                 task: CodingTask = None) -> HistoricalContextMethod:
    table = {
        "raw_clipped": RawClippedMethod,
        "sliding_window": SlidingWindowMethod,
        "llm_summarization": LLMSummarizationMethod,
        "vanilla_rag": VanillaRAGMethod,
        "adaptive": AdaptiveMethod,
        "no_history": NoHistoryMethod,
        "full_context": FullContextMethod,
        "direct_history": DirectHistoryMethod,
    }
    if name not in table:
        raise KeyError(f"unknown method {name!r}; options: {sorted(table)}")
    common = dict(model=model, endpoint=endpoint, embed_fn=embed_fn,
                  embedding_model=embedding_model)
    if name == "llm_summarization":
        return table[name](**common, summarizer_generate_fn=summarizer_generate_fn)
    if name == "adaptive":
        return table[name](**common, retention_mode=retention_mode)
    if name == "direct_history":
        return table[name](**common, task=task)
    return table[name](**common)


# ---------------------------------------------------------------------------
# Model caller
# ---------------------------------------------------------------------------

class OllamaCoder:
    """Thin Ollama /api/generate wrapper with injectable replacement for tests."""

    def __init__(self, model: str, endpoint: str = "http://localhost:11434",
                 max_output_tokens: int = MAX_OUTPUT_TOKENS):
        self.model = model
        self.endpoint = endpoint
        self.max_output_tokens = int(max_output_tokens)

    def generate(self, prompt: str) -> Dict:
        import requests
        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.endpoint}/api/generate",
            json={
                "model": self.model, "prompt": prompt, "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": CODING_TEMPERATURE,
                            "num_ctx": MODEL_CONTEXT_TOKENS,
                            "num_predict": self.max_output_tokens},
            },
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "text": data.get("response", "") or "",
            "prompt_tokens": int(data.get("prompt_eval_count", 0) or 0),
            "output_tokens": int(data.get("eval_count", 0) or 0),
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }


# ---------------------------------------------------------------------------
# Prompt construction + patch handling
# ---------------------------------------------------------------------------

def workspace_context(task: CodingTask) -> str:
    lines = []
    for path in sorted(task.workspace_path.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(task.workspace_path)
        if any(part in (".git", "__pycache__") for part in rel.parts):
            continue
        try:
            body = path.read_text()
        except UnicodeDecodeError:
            continue
        lines.append(f"### FILE: {rel}\n{body.rstrip()}\n")
    return "\n".join(lines)


def build_coding_prompt(task_prompt: str, workspace_ctx: str,
                        historical_context: str) -> str:
    hist = historical_context.strip() or "(none provided)"
    return (
        f"{INSTRUCTIONS}\n"
        f"CURRENT WORKSPACE FILES\n"
        f"{workspace_ctx.strip()}\n\n"
        f"CURRENT TASK\n{task_prompt.strip()}\n\n"
        f"HISTORICAL ENGINEERING CONTEXT\n{hist}\n"
    )


def build_repair_prompt(base_prompt: str, test_output: str) -> str:
    tail = (test_output or "")[-2000:]
    return (
        f"{base_prompt}\n\n"
        f"YOUR PREVIOUS EDIT FAILED THE TESTS.\n"
        f"Test failure output:\n{tail}\n\n"
        f"Return the COMPLETE corrected contents of every file you change using "
        f"the `### FILE: <path>` / `### END FILE` blocks. "
        f"Do not modify or weaken the tests.\n"
    )


_FENCE = re.compile(r"```(?:diff|patch)?[ \t]*\r?\n(.*?)```", re.DOTALL)
_FILE_HEADER = re.compile(r"^\s*#{1,6}\s*FILE:\s*(?P<path>.+?)\s*$")
_END_MARKER = re.compile(r"^\s*#{1,6}\s*END FILE\s*$")


def _extract_fenced(body: str) -> str:
    """Return the fenced content if any, else the body; drops surrounding prose."""
    lines = body.splitlines()
    start = next((k for k, ln in enumerate(lines) if ln.strip().startswith("```")), None)
    if start is None:
        return body.strip("\n")
    end = next((k for k in range(start + 1, len(lines))
                if lines[k].strip().startswith("```")), None)
    inner = lines[start + 1:end] if end is not None else lines[start + 1:]
    return "\n".join(inner).strip("\n")


def extract_file_edits(text: str) -> Dict[str, str]:
    """Parse ``### FILE: path`` complete-file blocks.

    Tolerates the formats a real model actually emits: the ``### END FILE``
    marker is optional, and the file body may be wrapped in a fenced code block
    followed by prose. The coding model reliably rewrites whole files but
    frequently emits malformed multi-hunk unified diffs (missing ``@@`` headers),
    which ``git apply`` then applies *partially* and silently. Complete-file
    blocks are deterministic to apply and keep the only controlled difference
    between arms the historical context, not diff-formatting luck.
    """
    if not text:
        return {}
    lines = str(text).splitlines()
    edits: Dict[str, str] = {}
    i, n = 0, len(lines)
    while i < n:
        m = _FILE_HEADER.match(lines[i])
        if not m:
            i += 1
            continue
        rel = m.group("path").strip().strip("`").strip()
        i += 1
        buf: List[str] = []
        while i < n and not _END_MARKER.match(lines[i]) and not _FILE_HEADER.match(lines[i]):
            buf.append(lines[i])
            i += 1
        if i < n and _END_MARKER.match(lines[i]):
            i += 1
        if not rel:
            continue
        body = _extract_fenced("\n".join(buf))
        if body:
            edits[rel] = body.rstrip("\n") + "\n"
    return edits


def apply_edits(repo: Path, edits: Dict[str, str]) -> Tuple[bool, str]:
    repo = Path(repo).resolve()
    prefix = str(repo) + os.sep
    for rel in edits:
        target = (repo / rel).resolve()
        if not str(target).startswith(prefix):
            return False, f"edit path escapes repo: {rel}"
    for rel, body in edits.items():
        target = (repo / rel).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    return True, "file_replace"


def derive_patch(repo: Path) -> str:
    """Capture the applied working-tree change as a unified diff (provenance)."""
    repo = Path(repo).resolve()
    proc = subprocess.run(["git", "--no-pager", "diff", "--no-color", "--",
                           ".", ":(exclude)_e17_attempt.patch", ":(exclude)test_hidden.py"],
                          cwd=str(repo), capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else ""


def extract_patch(text: str) -> Optional[str]:
    if not text:
        return None
    for m in _FENCE.finditer(text):
        body = m.group(1)
        if ("---" in body and "+++" in body) or body.lstrip().startswith("diff --git"):
            return _clean_patch(body)
    lines = str(text).splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("diff --git") or line.startswith("--- ") or line.startswith("--- a/"):
            start = i
            break
    if start is not None:
        body = "\n".join(lines[start:])
        if "+++" in body:
            return _clean_patch(body)
    return None


def _clean_patch(body: str) -> str:
    body = body.strip("\n")
    if body.endswith("```"):
        body = body[:-3]
    return body.rstrip() + "\n"


def apply_patch(repo: Path, patch: str) -> Tuple[bool, str]:
    repo = Path(repo).resolve()
    patch_file = (repo / "_e17_attempt.patch").resolve()
    patch_file.write_text(patch)
    check = subprocess.run(["git", "apply", "--check", str(patch_file)],
                           cwd=str(repo), capture_output=True, text=True)
    if check.returncode == 0:
        applied = subprocess.run(["git", "apply", str(patch_file)],
                                 cwd=str(repo), capture_output=True, text=True)
        if applied.returncode == 0:
            return True, "git apply"
        return False, f"git apply failed: {applied.stderr.strip()}"
    # fallback: GNU patch
    dry = subprocess.run(["patch", "-p1", "--dry-run", "-i", str(patch_file)],
                         cwd=str(repo), capture_output=True, text=True)
    if dry.returncode == 0:
        real = subprocess.run(["patch", "-p1", "-i", str(patch_file)],
                              cwd=str(repo), capture_output=True, text=True)
        if real.returncode == 0:
            return True, "patch -p1"
    return False, f"git apply --check failed: {check.stderr.strip()}"


# ---------------------------------------------------------------------------
# Workspace + hidden tests
# ---------------------------------------------------------------------------

def prepare_workspace(task: CodingTask, work_root: Path) -> Path:
    dest = work_root
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(task.workspace_path, dest)
    env = ["git", "-c", "user.email=e17@local", "-c", "user.name=e17"]
    subprocess.run(env + ["init", "-q"], cwd=str(dest), check=True,
                   capture_output=True, text=True)
    subprocess.run(env + ["add", "-A"], cwd=str(dest), check=True,
                   capture_output=True, text=True)
    subprocess.run(env + ["commit", "-q", "-m", "base"], cwd=str(dest), check=True,
                   capture_output=True, text=True)
    return dest


def run_hidden_tests(repo: Path, task: CodingTask) -> Tuple[bool, str, str]:
    test_dest = repo / "test_hidden.py"
    shutil.copy(task.hidden_test_path, test_dest)
    proc = subprocess.run(task.hidden_test_command, cwd=str(repo),
                          capture_output=True, text=True, timeout=300)
    return proc.returncode == 0, proc.stdout, proc.stderr


# ---------------------------------------------------------------------------
# Coding attempt
# ---------------------------------------------------------------------------

@dataclass
class CodingAttemptResult:
    ok: bool
    patch_valid: bool
    patch_applied: bool
    patch_method: Optional[str]
    test_pass: bool
    model_text: str
    prompt_tokens: int
    output_tokens: int
    latency_ms: float
    test_output: str
    attempt: int
    patch: Optional[str] = None


def run_coding_attempt(*, repo: Path, task: CodingTask, task_prompt: str,
                       workspace_ctx: str, historical_context: str,
                       coder, attempt: int = 1,
                       repair_output: str = None) -> CodingAttemptResult:
    base_prompt = build_coding_prompt(task_prompt, workspace_ctx, historical_context)
    prompt = base_prompt if attempt == 1 else build_repair_prompt(base_prompt, repair_output or "")
    gen = coder.generate(prompt)
    text = gen["text"]

    # Primary path: complete-file replacement blocks.
    edits = extract_file_edits(text)
    if edits:
        ok_write, method = apply_edits(repo, edits)
        if not ok_write:
            return CodingAttemptResult(False, False, False, None, False, text,
                                       gen.get("prompt_tokens", 0), gen.get("output_tokens", 0),
                                       gen.get("latency_ms", 0.0), method, attempt, None)
        patch = derive_patch(repo)
        if not patch.strip():
            return CodingAttemptResult(False, False, True, method, False, text,
                                       gen.get("prompt_tokens", 0), gen.get("output_tokens", 0),
                                       gen.get("latency_ms", 0.0),
                                       "model edits produced no working-tree change",
                                       attempt, None)
        passed, out, err = run_hidden_tests(repo, task)
        return CodingAttemptResult(passed, True, True, method, passed, text,
                                   gen.get("prompt_tokens", 0), gen.get("output_tokens", 0),
                                   gen.get("latency_ms", 0.0),
                                   (out + "\n" + err).strip(), attempt, patch)

    # Fallback: unified diff if the model ignored the format instruction.
    patch = extract_patch(text)
    if patch is None:
        return CodingAttemptResult(False, False, False, None, False, text,
                                   gen.get("prompt_tokens", 0), gen.get("output_tokens", 0),
                                   gen.get("latency_ms", 0.0),
                                   "no complete-file block or unified diff found in model output",
                                   attempt, None)
    ok_apply, method = apply_patch(repo, patch)
    if not ok_apply:
        return CodingAttemptResult(False, True, False, None, False, text,
                                   gen.get("prompt_tokens", 0), gen.get("output_tokens", 0),
                                   gen.get("latency_ms", 0.0), method, attempt, patch)
    passed, out, err = run_hidden_tests(repo, task)
    return CodingAttemptResult(passed, True, True, method, passed, text,
                               gen.get("prompt_tokens", 0), gen.get("output_tokens", 0),
                               gen.get("latency_ms", 0.0),
                               (out + "\n" + err).strip(), attempt, patch)


# ---------------------------------------------------------------------------
# Diagnostics (historical-recall; explanatory only)
# ---------------------------------------------------------------------------

def _fact_present(fact, memory_ids: set, memory_text: str,
                  context_ids: set, context_text: str) -> Tuple[bool, bool]:
    if memory_ids:
        in_mem = fact.fact_id in memory_ids
    else:
        in_mem = fact.probe.lower() in memory_text.lower()
    if context_ids:
        in_ctx = fact.fact_id in context_ids
    else:
        in_ctx = fact.probe.lower() in context_text.lower()
    return in_mem, in_ctx


def memory_diagnostics(task: CodingTask, prepared: PreparedMemory,
                       context_text: str) -> Dict:
    # Obsolete facts are *supposed* to be superseded, so they are excluded from
    # recall denominators (and handled by the obsolete-exposure metric instead).
    gold = [f for f in task.gold_facts if not f.obsolete]
    n = len(gold) or 1
    in_mem, in_ctx = [], []
    for f in gold:
        m, c = _fact_present(f, prepared.memory_ids, prepared.memory_text,
                             getattr(prepared, "context_ids", set()), context_text)
        in_mem.append(m)
        in_ctx.append(c)
    long_facts = [f for f in gold
                  if (task.history_turns - f.turn_index) > 300]
    neg_facts = [f for f in gold if f.kind == "negative"]

    def rate(sub, flags):
        if not sub:
            return None
        idx = [i for i, f in enumerate(gold) if f in sub]
        return round(sum(1 for i in idx if flags[i]) / len(idx), 3)

    corr_current, corr_ok = [], []
    obsolete_exposed = []
    for c in task.corrections:
        cur = next((f for f in gold if f.fact_id == c.current_fact_id), None)
        old = next((f for f in task.gold_facts + task.distractor_facts
                    if f.fact_id == c.obsolete_fact_id), None)
        if cur is None:
            continue
        i = gold.index(cur)
        corr_current.append(in_ctx[i])
        old_present = (old.probe.lower() in context_text.lower()) if old else False
        obsolete_exposed.append(old_present)
        corr_ok.append(in_ctx[i] and not old_present)

    return {
        "critical_fact_recall": round(sum(in_ctx) / n, 3),
        "critical_fact_in_memory": round(sum(in_mem) / n, 3),
        "correction_recall": (round(sum(corr_ok) / len(corr_ok), 3)
                              if corr_ok else None),
        "negative_constraint_recall": rate(neg_facts, in_ctx),
        "long_range_fact_recall": rate(long_facts, in_ctx),
        "obsolete_fact_exposure": (round(sum(obsolete_exposed) / len(obsolete_exposed), 3)
                                   if obsolete_exposed else None),
    }


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def leakage_check(task: CodingTask, prompt: str) -> Dict:
    """Assert the constructed prompt never contains hidden-test material.

    This is a runtime guarantee, not a post-hoc claim: if any non-trivial line
    of the hidden test file, the task's hidden-test filename, or the hidden test
    body appears in the prompt, ``leakage_detected`` is True and the pilot gate
    fails.
    """
    hidden = task.hidden_test_path.read_text() if task.hidden_test_path.exists() else ""
    # Only the test *logic* (assertions / test function names) is secret. Shared
    # import lines legitimately appear in the visible workspace and are not
    # leakage.
    distinctive = [ln.strip() for ln in hidden.splitlines()
                   if (("assert" in ln or ln.strip().startswith("def test_"))
                       and len(ln.strip()) >= 20)]
    hits = [ln for ln in distinctive if ln in prompt]
    detected = bool(hits) or ("test_hidden" in prompt)
    return {
        "hidden_test_logic_lines_in_prompt": len(hits),
        "hidden_test_filename_in_prompt": "test_hidden" in prompt,
        "leakage_detected": detected,
    }


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

FAILURE_CLASSES = [
    "MEMORY_MISS", "RETRIEVAL_MISS", "CONTEXT_OVERFLOW", "PATCH_INVALID",
    "CODING_ERROR", "HIDDEN_TEST_FAILURE", "OBSOLETE_INFORMATION_USED",
    "CORRECTION_MISSED", "OTHER",
]


def classify_failure(*, success: bool, overflow: bool, patch_valid: bool,
                     test_pass: bool, gold_in_memory: bool, gold_in_context: bool,
                     obsolete_exposed: bool, has_correction: bool,
                     correction_in_context: bool) -> Optional[str]:
    if success:
        return None
    if overflow:
        return "CONTEXT_OVERFLOW"
    if not patch_valid:
        return "PATCH_INVALID"
    if not gold_in_memory:
        return "MEMORY_MISS"
    if not gold_in_context:
        return "RETRIEVAL_MISS"
    if has_correction and obsolete_exposed and not correction_in_context:
        return "OBSOLETE_INFORMATION_USED"
    if has_correction and not correction_in_context:
        return "CORRECTION_MISSED"
    if gold_in_context and not test_pass:
        return "CODING_ERROR"
    if not test_pass:
        return "HIDDEN_TEST_FAILURE"
    return "OTHER"


# ---------------------------------------------------------------------------
# One method-run
# ---------------------------------------------------------------------------

def run_method_run(*, task: CodingTask, seed: int, historical_budget: int,
                   method: HistoricalContextMethod, coder,
                   work_root: Path, max_attempts: int = MAX_ATTEMPTS) -> Dict:
    history = HistoryBundle(task.task_id, task.history, task.history_text)
    ws_ctx = workspace_context(task)

    t_prep0 = time.perf_counter()
    prepared = method.prepare(history, historical_budget, seed=seed)
    prepare_ms = (time.perf_counter() - t_prep0) * 1000

    context = method.retrieve(prepared, task.task_prompt, historical_budget)
    context_ids = set(getattr(method, "last_context_ids", set()) or set())
    # expose context ids to diagnostics
    prepared_ctx = PreparedMemory(method=prepared.method,
                                  memory_text=prepared.memory_text,
                                  memory_ids=prepared.memory_ids,
                                  memory_tokens=prepared.memory_tokens,
                                  build=prepared.build, payload=prepared.payload)
    setattr(prepared_ctx, "context_ids", context_ids)

    historical_context_tokens = count_tokens(context) if context.strip() else 0
    workspace_context_tokens = count_tokens(ws_ctx)
    task_prompt_tokens = count_tokens(task.task_prompt)
    total_prompt_tokens = count_tokens(
        build_coding_prompt(task.task_prompt, ws_ctx, context))

    overflow = (historical_context_tokens > historical_budget
                or total_prompt_tokens >= MODEL_CONTEXT_TOKENS)

    coding_calls = 0
    coding_in = 0
    coding_out = 0
    coding_latency = 0.0
    first_pass_success = False
    final_success = False
    patch_valid = False
    patch_applied = False
    patch_method = None
    last_test_output = ""
    final_patch = None
    first_pass_patch = None

    if not overflow:
        for attempt in range(1, max_attempts + 1):
            repo = prepare_workspace(task, work_root)
            res = run_coding_attempt(
                repo=repo, task=task, task_prompt=task.task_prompt,
                workspace_ctx=ws_ctx, historical_context=context, coder=coder,
                attempt=attempt,
                repair_output=last_test_output if attempt > 1 else None)
            coding_calls += 1
            coding_in += res.prompt_tokens
            coding_out += res.output_tokens
            coding_latency += res.latency_ms
            patch_valid = patch_valid or res.patch_valid
            patch_applied = patch_applied or res.patch_applied
            patch_method = patch_method or res.patch_method
            last_test_output = res.test_output
            if res.patch is not None:
                final_patch = res.patch
            if attempt == 1:
                first_pass_success = res.ok
                first_pass_patch = res.patch
            if res.ok:
                final_success = True
                break

    diag = memory_diagnostics(task, prepared_ctx, context)

    # recovery of the "all required gold in context" signal for classification
    # (obsolete facts are expected to be absent, so they are not required)
    requirement = [f for f in task.gold_facts if not f.obsolete]
    gold_in_memory = True
    gold_in_context = True
    for f in requirement:
        m, c = _fact_present(f, prepared.memory_ids, prepared.memory_text,
                             context_ids, context)
        gold_in_memory = gold_in_memory and m
        gold_in_context = gold_in_context and c
    corr_in_ctx = diag.get("correction_recall")
    obsolete_exposed = bool(diag.get("obsolete_fact_exposure"))

    failure_class = classify_failure(
        success=final_success, overflow=overflow, patch_valid=patch_valid,
        test_pass=final_success, gold_in_memory=gold_in_memory,
        gold_in_context=gold_in_context, obsolete_exposed=obsolete_exposed,
        has_correction=bool(task.corrections),
        correction_in_context=(corr_in_ctx == 1.0) if corr_in_ctx is not None else False)

    build = prepared.build or {}
    return {
        "experiment": "E17",
        "task_id": task.task_id,
        "seed": seed,
        "method": method.name,
        "historical_budget": historical_budget,
        "historical_context_tokens": historical_context_tokens,
        "workspace_context_tokens": workspace_context_tokens,
        "task_prompt_tokens": task_prompt_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "memory_build_calls": int(build.get("memory_build_calls", 0)),
        "summary_update_calls": int(build.get("summary_update_calls", 0)),
        "coding_attempts": coding_calls,
        "first_pass_success": bool(first_pass_success),
        "final_success": bool(final_success),
        "hidden_test_pass": bool(final_success),
        "patch_valid": bool(patch_valid),
        "patch_applied": bool(patch_applied),
        "patch_method": patch_method,
        "patch": final_patch,
        "first_pass_patch": first_pass_patch,
        "failure_class": failure_class,
        "context_sha": _sha16(context),
        "uses_production_pipeline": (method.name == "adaptive"),
        "leakage": leakage_check(task, build_coding_prompt(
            task.task_prompt, ws_ctx, context)),
        "diagnostics": diag,
        "latency_ms": {
            "memory_build": round(build.get("memory_build_latency_ms", 0.0), 1),
            "prepare_total": round(prepare_ms, 1),
            "retrieval": round(getattr(method, "last_context_latency_ms", 0.0), 1),
            "coding": round(coding_latency, 1),
            "total": round(prepare_ms + getattr(method, "last_context_latency_ms", 0.0)
                           + coding_latency, 1),
        },
        "tokens": {
            "memory_build_input": int(build.get("memory_build_input_tokens", 0)),
            "memory_build_output": int(build.get("memory_build_output_tokens", 0)),
            "coding_input": coding_in,
            "coding_output": coding_out,
        },
        "cache": {},
        "provenance": {
            "tokenizer": TOKENIZER_NAME,
            "model": coder.model if hasattr(coder, "model") else None,
            "ingestion": task.metadata.get("ingestion"),
        },
    }
