"""B2: sliding-window baseline — the model only ever sees the last N raw turns.

The window is bounded by both a turn count and the shared token budget, so it is
directly comparable to the adaptive store under D1. It performs no retrieval and
no scoring: whatever raw turns fit are what the model receives, and any fact in a
turn that has scrolled out is gone.
"""

from typing import Dict, List

from baselines.baseline_runner import BaseBaseline


class SlidingWindowBaseline(BaseBaseline):
    name = "sliding_window"

    def __init__(self, window_size: int = 10, **common):
        super().__init__(**common)
        self.window_size = int(window_size)
        self.window: List[Dict] = []

    def reset(self) -> None:
        super().reset()
        self.window = []

    def observe(self, turn: Dict) -> None:
        self.window.append(turn)
        while len(self.window) > self.window_size:
            self.window.pop(0)
        while self.window and sum((t.get("tokens") or 0) for t in self.window) > self.budget:
            self.window.pop(0)

    def held_facts(self) -> List[Dict]:
        return self._turn_entries(self.window)

    def retrieve(self, query: str) -> List[Dict]:
        return self._turn_entries(self.window)

    def _turn_entries(self, turns: List[Dict]) -> List[Dict]:
        out = []
        for i, t in enumerate(turns):
            out.append({
                "fact": t.get("user", ""),
                "source_turn_id": t.get("turn_id"),
                "injected_tokens": t.get("tokens"),
                "retrieval_rank": i,
                "retrieval_sim": None,
                "retrieval_engine": "window",
                "raw_message_tokens": t.get("tokens"),
            })
        return out

    # compatibility with the original stub API
    def process(self, history: List[Dict]) -> List[Dict]:
        return history[-self.window_size:]
