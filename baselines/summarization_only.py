"""B4: summarization-only baseline — collapses history into monolithic text.

There is no category scoring, decay, or retrieval. Facts are concatenated into a
single rolling "summary" that grows until it hits the shared token budget, at
which point the oldest content is dropped to make room. This is deliberately the
concatenative placeholder described in brain.md D5 (a true LLM summarizer is a
separate, high-risk experiment): it measures the *cost/retention* profile of
compression-by-accrual, not the quality of real summarization.
"""

from typing import Dict, List

from baselines.baseline_runner import BaseBaseline


class SummarizationOnlyBaseline(BaseBaseline):
    name = "summarization_only"

    def __init__(self, **common):
        super().__init__(**common)
        self.summary_facts: List[Dict] = []

    def reset(self) -> None:
        super().reset()
        self.summary_facts = []

    def observe(self, turn: Dict) -> None:
        for f in turn.get("facts", []):
            item = dict(f)
            item.setdefault("source_turn_id", turn["turn_id"])
            self.summary_facts.append(item)

    def _bounded_summary(self) -> List[Dict]:
        # keep the most recent content that fits the budget
        kept, used = [], 0
        for f in reversed(self.summary_facts):
            t = self.entry_tokens(f)
            if used + t <= self.budget:
                kept.append(f)
                used += t
        kept.reverse()
        return kept

    def held_facts(self) -> List[Dict]:
        return [dict(f) for f in self._bounded_summary()]

    def retrieve(self, query: str) -> List[Dict]:
        out = []
        for i, f in enumerate(self._bounded_summary()):
            c = dict(f)
            c["retrieval_engine"] = "summary"
            c["retrieval_rank"] = i
            c["retrieval_sim"] = None
            out.append(c)
        return out

    # compatibility with the original stub API
    def summarize(self, history: List[Dict]) -> Dict:
        combined = " ".join([h["user"] for h in history])
        return {"fact": f"Global Summary: {combined[:150]}...", "category": "summary"}
