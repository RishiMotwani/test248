"""Future-blind task-state tracker (Phase 12 / E16).

``task_affinity`` retention candidate: a memory should survive a store squeeze
when it belongs to the *current task*, i.e. semantically close to the recent run
of user messages that actually carried extracted facts (filler turns are
excluded, so the tracker does not surface generic chit-chat).

Design:

* FIFO window of the last ``window`` *fact-carrying* turns (default 32).
* ``affinity`` of a fact = mean cosine of its embedding against the top-4 most
  similar message embeddings in the window (k = min(4, window_size)), clamped to
  [0, 1]. Top-k (rather than mean-over-window) is used because a single on-task
  burst in an otherwise heterogeneous window is a real signal.
* Causal by construction: ``observe`` is called only during ingest of the
  current turn, so the tracker can never see later (future) turns. The
  no-future-leakage assertions in E16 pin this invariant.
* No LLM calls: embeddings reuse the pipeline's cached ``embed_fn``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class TaskStateTracker:
    def __init__(self, window: int = 32):
        self.window = max(1, int(window))
        self.messages: List[Dict] = []

    def observe(self, turn_id: int, user_message: str, facts: List[Dict],
                embed_fn=None) -> None:
        """Record the turn's user message iff the turn carried extracted facts."""
        if not facts or embed_fn is None:
            return
        try:
            vec = embed_fn([user_message])
            vec = vec[0] if isinstance(vec, (list, tuple)) and len(vec) == 1 else vec
        except Exception:
            return
        if vec is None:
            return
        self.messages.append({"turn": int(turn_id), "embedding": np.asarray(vec, dtype=float)})
        if len(self.messages) > self.window:
            self.messages = self.messages[-self.window:]

    def message_matrix(self) -> Optional[np.ndarray]:
        if not self.messages:
            return None
        return np.vstack([m["embedding"] for m in self.messages])

    def batch_affinity(self, embeddings: List[Optional[np.ndarray]],
                       top_k: int = 4) -> List[float]:
        """Vectorized task affinity for a list of fact embeddings.

        Returns a value in [0, 1] aligned with ``embeddings`` (0.0 when the
        tracker is empty, the embedding is missing, or there is nothing to
        compare against).
        """
        mat = self.message_matrix()
        n = len(embeddings)
        if mat is None or n == 0:
            return [0.0] * n
        k = min(max(1, top_k), len(self.messages))
        out = []
        for emb in embeddings:
            if emb is None:
                out.append(0.0)
                continue
            f = np.asarray(emb, dtype=float)
            nf = float(np.linalg.norm(f))
            if nf == 0.0:
                out.append(0.0)
                continue
            sims = (f @ mat.T) / nf
            norms = np.linalg.norm(mat, axis=1)
            sims = np.divide(sims, norms, out=np.zeros_like(sims), where=norms > 0)
            top = np.sort(sims)[-k:]
            val = float(np.mean(top)) if k else 0.0
            out.append(max(0.0, min(1.0, val)))
        return out