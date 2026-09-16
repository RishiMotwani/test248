"""Uniform comparison baselines + batch runner (task A).

Research framing (brain.md D9): the *writer* (LLM fact extraction) is held fixed
across every method — all baselines consume the exact same pre-extracted fact
stream that the adaptive system produced. Only the *memory policy* changes:
what is retained, what is evicted under the shared token budget, and how the
context for the current turn is assembled. This isolates the contribution of the
adaptive decay/retrieval policy from the contribution of LLM extraction, matching
the retrieval-vs-write decomposition in arXiv:2603.02473.

Every number this module emits is derived from the shared extraction stream and
the shared token/embedding functions; there is no fabricated result. If token
measurement is unavailable the runner records ``estimated(word-count)`` rather
than pretending it is measured.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_optimizer.retrieval import _cosine, _token_overlap  # noqa: E402

FactTokens = Callable[[str], int]
EmbedFn = Callable[[List[str]], list]

MATCH_OVERLAP = 0.7


def fact_matches(gt: Dict, facts: List[Dict], overlap: float = MATCH_OVERLAP) -> bool:
    """A ground-truth fact is recalled if a candidate shares its source turn or
    paraphrases it (same rule as experiments.live._recall_proposed)."""
    for f in facts or []:
        if f.get("source_turn_id") == gt.get("source_turn"):
            return True
        if overlap and _token_overlap(gt.get("fact", ""), f.get("fact", "")) >= overlap:
            return True
    return False


def matched_gt_keys(ground_truth: List[Dict], facts: List[Dict], overlap: float = MATCH_OVERLAP) -> List[str]:
    keys = []
    for gt in ground_truth:
        if fact_matches(gt, facts, overlap):
            keys.append(f"{gt['source_turn']}:{gt['fact'][:48]}")
    return keys


def _dot(a, b) -> Optional[float]:
    return _cosine(a, b)


class BaseBaseline:
    """Common policy interface. Subclasses implement observe/held_facts/retrieve."""

    name = "base"

    def __init__(
        self,
        budget: int = 4096,
        top_k: int = 5,
        embed_fn: EmbedFn = None,
        fact_tokens: FactTokens = None,
        embedding_model: str = "nomic-embed-text",
        sim_threshold: float = 0.35,
    ):
        self.budget = int(budget)
        self.top_k = int(top_k)
        self.embed_fn = embed_fn
        self.fact_tokens = fact_tokens
        self.embedding_model = embedding_model
        self.sim_threshold = sim_threshold
        self.per_turn: List[Dict] = []

    # -- policy hooks -----------------------------------------------------
    def observe(self, turn: Dict) -> None:
        raise NotImplementedError

    def held_facts(self) -> List[Dict]:
        """Everything the method currently stores (unbounded view)."""
        raise NotImplementedError

    def retrieve(self, query: str) -> List[Dict]:
        """Context actually handed to the model for this query (budget-bounded)."""
        raise NotImplementedError

    # -- shared helpers ---------------------------------------------------
    def reset(self) -> None:
        self.per_turn = []

    def tokens(self, text: str) -> int:
        if self.fact_tokens is not None:
            v = self.fact_tokens(text)
            if v is not None:
                return int(v)
        return max(1, len(str(text).split()))

    def token_source(self) -> str:
        return "measured" if self.fact_tokens is not None else "estimated(word-count)"

    def entry_tokens(self, e: Dict) -> int:
        """Token cost of one injected entry: explicit override, else text length.

        Window/summary policies inject raw turns and stamp ``injected_tokens``;
        fact policies fall back to the measured size of the fact string.
        """
        if e.get("injected_tokens") is not None:
            return int(e["injected_tokens"])
        return self.tokens(e.get("fact", e.get("user", "")))

    def fit_entries(self, entries: List[Dict], limit: int = None) -> List[Dict]:
        """Greedy token-aware truncation preserving caller order (oldest first)."""
        limit = self.budget if limit is None else limit
        selected, used = [], 0
        for e in entries:
            t = self.entry_tokens(e)
            if used + t <= limit:
                selected.append(e)
                used += t
        return selected

    def _embed(self, text: str) -> Optional[list]:
        if self.embed_fn is None:
            return None
        try:
            v = self.embed_fn([text])
            return v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v
        except Exception:
            return None

    def _ensure_embedding(self, item: Dict) -> None:
        if self.embed_fn is not None and item.get("fact_embedding") is None:
            item["fact_embedding"] = self._embed(item.get("fact", ""))

    def rank_by_similarity(self, query: str, items: List[Dict], top_k: int = None) -> List[Dict]:
        """Pure similarity ranking (no importance) — the 'static vector store' path."""
        top_k = self.top_k if top_k is None else top_k
        qvec = self._embed(query)
        scored = []
        for it in items:
            self._ensure_embedding(it)
            sim, engine = None, "lexical"
            if qvec is not None and it.get("fact_embedding") is not None:
                sim = _dot(qvec, it["fact_embedding"])
                engine = "embedding"
            if sim is None:
                sim = _token_overlap(query, it.get("fact", ""))
            scored.append((sim, engine, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for rank, (sim, engine, it) in enumerate(scored):
            c = dict(it)
            c["retrieval_sim"] = round(sim, 4)
            c["retrieval_engine"] = engine
            c["retrieval_rank"] = rank
            out.append(c)
        return out[:top_k]


class BaselineRunner:
    """Runs one or more baselines over a shared extraction stream."""

    def __init__(
        self,
        budget: int = 4096,
        top_k: int = 5,
        embed_fn: EmbedFn = None,
        fact_tokens: FactTokens = None,
        embedding_model: str = "nomic-embed-text",
    ):
        self.budget = int(budget)
        self.top_k = int(top_k)
        self.embed_fn = embed_fn
        self.fact_tokens = fact_tokens
        self.embedding_model = embedding_model

    def _make(self, method: str) -> BaseBaseline:
        from baselines.memgpt_style import MemGPTStyleBaseline
        from baselines.sliding_window import SlidingWindowBaseline
        from baselines.summarization_only import SummarizationOnlyBaseline
        from baselines.vanilla_rag import VanillaRAGBaseline

        common = dict(
            budget=self.budget,
            top_k=self.top_k,
            embed_fn=self.embed_fn,
            fact_tokens=self.fact_tokens,
            embedding_model=self.embedding_model,
        )
        table = {
            "sliding_window": SlidingWindowBaseline,
            "memgpt_style": MemGPTStyleBaseline,
            "summarization_only": SummarizationOnlyBaseline,
            "vanilla_rag": VanillaRAGBaseline,
        }
        if method not in table:
            raise KeyError(f"unknown baseline '{method}'; options: {sorted(table)}")
        return table[method](**common)

    def run(self, stream: List[Dict], ground_truth: List[Dict], method: str) -> Dict:
        baseline = self._make(method)
        baseline.reset()
        per_turn = []

        for turn in stream:
            baseline.observe(turn)
            retrieved = baseline.retrieve(turn["user"])
            recalled = matched_gt_keys(ground_truth, retrieved)
            injected_tokens = sum(baseline.entry_tokens(f) for f in retrieved)
            per_turn.append({
                "turn_id": turn["turn_id"],
                "injected_tokens": injected_tokens,
                "retrieved_count": len(retrieved),
                "retrieved": [
                    {
                        "fact": f.get("fact", ""),
                        "source_turn_id": f.get("source_turn_id"),
                        "retrieval_rank": f.get("retrieval_rank"),
                        "retrieval_sim": f.get("retrieval_sim"),
                        "retrieval_engine": f.get("retrieval_engine"),
                    }
                    for f in retrieved
                ],
                "recalled_fact_ids": recalled,
                "answer_via_window": bool(recalled),
            })

        held = baseline.held_facts()
        held_recall = (
            sum(1 for gt in ground_truth if fact_matches(gt, held)) / len(ground_truth)
            if ground_truth else 0.0
        )
        inject_tokens = [p["injected_tokens"] for p in per_turn]

        return {
            "method": method,
            "budget": self.budget,
            "top_k": self.top_k,
            "per_turn": per_turn,
            "mean_injected_tokens": round(sum(inject_tokens) / len(inject_tokens), 2) if inject_tokens else 0.0,
            "max_injected_tokens": max(inject_tokens) if inject_tokens else 0,
            "held_count": len(held),
            "held_recall": round(held_recall, 3),
            "held_facts": [dict(h) for h in held],
            "turns": len(per_turn),
            "token_source": baseline.token_source(),
            "embedding_engine": "embedding" if self.embed_fn is not None else "lexical",
            "metric_source": "shared pre-extracted facts (writer held fixed); baseline policy only",
            "answer_label": "synthetic" if self.fact_tokens is None else "measured",
        }

    def run_all(self, stream: List[Dict], ground_truth: List[Dict], methods: List[str] = None) -> Dict[str, Dict]:
        methods = methods or ["sliding_window", "memgpt_style", "summarization_only", "vanilla_rag"]
        return {m: self.run(stream, ground_truth, m) for m in methods}


def _quick_selftest() -> None:
    """Offline wiring check: synthetic stream, lexical fallback, no LLM calls."""
    from data.synthetic_generator import SyntheticConversationGenerator

    turns, gt = SyntheticConversationGenerator(seed=42).generate_conversation(num_turns=40, signal_density=0.25)
    gt_by_turn = defaultdict(list)
    for g in gt:
        gt_by_turn[g["source_turn"]].append(g)
    stream = []
    for t in turns:
        facts = [dict(g, source_turn_id=g["source_turn"]) for g in gt_by_turn.get(t["turn_id"], [])]
        stream.append({
            "turn_id": t["turn_id"],
            "user": t["user"],
            "tokens": max(1, len(t["user"].split())),
            "facts": facts,
        })

    runner = BaselineRunner(budget=4096, top_k=5, embed_fn=None, fact_tokens=None)
    results = runner.run_all(stream, gt)
    print(f"synthetic stream: {len(stream)} turns, {len(gt)} ground-truth facts (lexical fallback)")
    for name, r in results.items():
        print(f"  {name:20s} held={r['held_count']:3d}  held_recall={r['held_recall']:.3f}  "
              f"mean_inj_tok={r['mean_injected_tokens']:7.2f}  src={r['token_source']}")


if __name__ == "__main__":
    _quick_selftest()
