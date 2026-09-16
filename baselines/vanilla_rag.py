"""B3: Vanilla-RAG baseline — unscored, un-decayed static vector retrieval.

Every extracted fact is embedded once and stored permanently (no decay, no
dedupe, no importance). Retrieval is pure cosine similarity top-k, bounded by the
shared token budget. This is the "naive RAG" arm of the retrieval-vs-write
comparison (arXiv:2603.02473, arXiv:2607.29104): it remembers everything and
spends its budget only on similarity.
"""

from typing import Dict, List

from baselines.baseline_runner import BaseBaseline


class VanillaRAGBaseline(BaseBaseline):
    name = "vanilla_rag"

    def __init__(self, **common):
        super().__init__(**common)
        self.store: List[Dict] = []

    def reset(self) -> None:
        super().reset()
        self.store = []

    def observe(self, turn: Dict) -> None:
        for f in turn.get("facts", []):
            item = dict(f)
            item.setdefault("source_turn_id", turn["turn_id"])
            self._ensure_embedding(item)
            self.store.append(item)

    def held_facts(self) -> List[Dict]:
        return [dict(f) for f in self.store]

    def retrieve(self, query: str) -> List[Dict]:
        ranked = self.rank_by_similarity(query, self.store, top_k=self.top_k)
        return self.fit_entries(ranked)
