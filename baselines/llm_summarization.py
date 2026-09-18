"""B: real LLM-summarization baseline for E17.

Unlike ``summarization_only`` (a concatenative placeholder), this baseline keeps a
genuine running engineering summary produced by the *same* coding model used for
the task. Every ``SUMMARY_UPDATE_INTERVAL`` turns the previous summary and the new
chunk are sent to the model with an instruction to preserve architectural
constraints, current requirements, corrections, negative constraints, interfaces,
dependencies and decisions, while dropping obsolete information and chatter.

Fairness properties enforced here and by the E17 harness:

* the summarizer sees only history available at that point (no future turns),
* it never sees the final task prompt, hidden tests or gold facts,
* it never calls once per turn (only every ``update_interval``),
* at every update ``summary_tokens <= SUMMARY_BUDGET`` holds.

The model call is injectable (``generate_fn``) so unit tests never touch Ollama.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from memory_optimizer.tokenizer import count_tokens

SUMMARY_UPDATE_INTERVAL = 50
SUMMARY_UPDATE_SYSTEM_PROMPT = (
    "Update the running engineering summary of this software project.\n\n"
    "Preserve:\n"
    "- architectural constraints\n"
    "- current requirements\n"
    "- corrections (state the current truth, drop the obsolete version)\n"
    "- negative constraints (things that must not be done)\n"
    "- interfaces\n"
    "- dependencies\n"
    "- decisions that affect future implementation\n\n"
    "Remove:\n"
    "- obsolete information\n"
    "- repeated discussion\n"
    "- irrelevant chatter\n"
    "- implementation dead ends\n\n"
    "Do not invent facts. Do not use future turns. Return only the updated summary."
)

GenerateFn = Callable[[str, int], str]


class LLMSummarizer:
    def __init__(self, model: str, endpoint: str = "http://localhost:11434",
                 update_interval: int = SUMMARY_UPDATE_INTERVAL,
                 max_output_tokens: int = 400,
                 generate_fn: GenerateFn = None):
        self.model = model
        self.endpoint = endpoint
        self.update_interval = max(1, int(update_interval))
        self.max_output_tokens = int(max_output_tokens)
        self.generate_fn = generate_fn

    # -- model call -------------------------------------------------------
    def _generate(self, prompt: str, max_output_tokens: int) -> str:
        if self.generate_fn is not None:
            return self.generate_fn(prompt, max_output_tokens)
        import requests
        resp = requests.post(
            f"{self.endpoint}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0.1, "num_ctx": 8192,
                            "num_predict": int(max_output_tokens)},
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get("response", "") or ""

    @staticmethod
    def _truncate_to_budget(text: str, budget: int) -> str:
        words = str(text).split()
        if len(words) <= budget:
            return text
        return " ".join(words[:budget])

    # -- running summary --------------------------------------------------
    def build_summary(self, history_text: List[str], budget: int,
                      *, on_update=None) -> Dict:
        """Build a running summary over ``history_text`` in chunks of ``update_interval``.

        Returns the summary plus cost telemetry. ``budget`` is the hard
        historical-context budget (the summary may never exceed it).
        """
        import time
        budget = int(budget)
        summary = ""
        calls = 0
        in_tokens = 0
        out_tokens = 0
        latency_ms = 0.0
        chunks = 0
        for start in range(0, len(history_text), self.update_interval):
            chunk = history_text[start:start + self.update_interval]
            chunk_text = "\n".join(chunk)
            prompt = (
                f"{SUMMARY_UPDATE_SYSTEM_PROMPT}\n\n"
                f"PREVIOUS SUMMARY:\n{summary or '(empty)'}\n\n"
                f"NEW HISTORY CHUNK ({start + 1}-{start + len(chunk)}):\n{chunk_text}\n\n"
                f"Keep the updated summary under {budget} words."
            )
            t0 = time.perf_counter()
            out = self._generate(prompt, self.max_output_tokens)
            latency_ms += (time.perf_counter() - t0) * 1000
            calls += 1
            chunks += 1
            in_tokens += count_tokens(prompt)
            out_tokens += count_tokens(out)
            summary = self._truncate_to_budget(out.strip(), budget)
            if on_update is not None:
                on_update(chunks, summary)
        return {
            "summary": summary,
            "summary_update_calls": calls,
            "summary_input_tokens": in_tokens,
            "summary_output_tokens": out_tokens,
            "summary_latency_ms": round(latency_ms, 1),
            "summary_tokens": count_tokens(summary) if summary else 0,
            "summary_updates": chunks,
        }
