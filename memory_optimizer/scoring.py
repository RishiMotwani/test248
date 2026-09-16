import numpy as np
from typing import Callable, Dict, List

from memory_optimizer.retrieval import _cosine, _token_overlap


def _write_time_salience(user_message: str, fact: str,
                         embed_fn: Callable[[List[str]], list] = None) -> float:
    """Write-time salience of a fact against the turn's user message.

    Real relevance, not the constant 0.8 the scorer formerly received: cosine
    of the user-message and fact embeddings when ``embed_fn`` is available,
    lexical token overlap otherwise (same primitives retrieval uses). Clamped
    to [0, 1] for use as the ``w1_relevance`` input.
    """
    if not user_message or not fact:
        return 0.0
    if embed_fn is not None:
        try:
            vecs = embed_fn([user_message, fact])
            vecs = vecs[0] if isinstance(vecs, (list, tuple)) and len(vecs) == 1 else vecs
            if isinstance(vecs, (list, tuple)) and len(vecs) >= 2:
                cos = _cosine(vecs[0], vecs[1])
                if cos is not None:
                    return float(np.clip(cos, 0.0, 1.0))
        except Exception:
            pass
    return float(np.clip(_token_overlap(user_message, fact), 0.0, 1.0))


class ImportanceScorer:
    """Computes multi-factor importance scores for candidate memories."""

    def __init__(self, weights: Dict[str, float] = None):
        self.w = weights or {
            "w1_relevance": 0.40,
            "w2_utility": 0.30,
            "w3_recency": 0.15,
            "w4_frequency": 0.15
        }

    def compute_score(self, memory: Dict, query_relevance: float, current_turn: int) -> float:
        recency = 1.0 / (1.0 + 0.05 * (current_turn - memory.get("source_turn_id", current_turn)))
        frequency = min(1.0, memory.get("access_count", 1) / 10.0)
        utility = memory.get("confidence", 0.8)

        score = (
            self.w["w1_relevance"] * query_relevance +
            self.w["w2_utility"] * utility +
            self.w["w3_recency"] * recency +
            self.w["w4_frequency"] * frequency
        )
        return float(np.clip(score, 0.0, 1.0))
