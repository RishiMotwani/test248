"""B5: MemGPT-style paged memory baseline (arXiv:2310.08560).

Two tiers: a small always-resident *working memory* and a FIFO *archival* store.
Every observation appends to working memory; the oldest working item is paged out
to archival when working memory is full. Retrieval searches the archival tier by
similarity, and the injected context is working memory + the best-fitting
archival hits, bounded by the shared token budget (D1). Unlike the adaptive
system there is no decay, no scoring, and no dedupe.
"""

from collections import deque
from typing import Dict, List

from baselines.baseline_runner import BaseBaseline


class MemGPTStyleBaseline(BaseBaseline):
    name = "memgpt_style"

    def __init__(self, max_working: int = 5, **common):
        super().__init__(**common)
        self.max_working = int(max_working)
        self.working: deque = deque()
        self.archival: List[Dict] = []

    def reset(self) -> None:
        super().reset()
        self.working = deque()
        self.archival = []

    def observe(self, turn: Dict) -> None:
        for f in turn.get("facts", []):
            item = dict(f)
            item.setdefault("source_turn_id", turn["turn_id"])
            self.working.append(item)
        while len(self.working) > self.max_working:
            self.archival.append(self.working.popleft())

    def held_facts(self) -> List[Dict]:
        return [dict(m) for m in self.working] + [dict(m) for m in self.archival]

    def retrieve(self, query: str) -> List[Dict]:
        working = []
        for m in self.working:
            c = dict(m)
            c["retrieval_engine"] = "working"
            c["retrieval_rank"] = len(working)
            c["retrieval_sim"] = None
            working.append(c)
        selected = self.fit_entries(working)
        remaining = self.budget - sum(self.entry_tokens(x) for x in selected)
        if remaining > 0 and self.archival:
            archival_ranked = self.rank_by_similarity(query, self.archival, top_k=self.top_k)
            selected += self.fit_entries(archival_ranked, limit=remaining)
        return selected

    # compatibility with the original stub API
    def add_memory(self, item: Dict) -> None:
        self.observe({"turn_id": item.get("source_turn_id", 0), "facts": [item]})
