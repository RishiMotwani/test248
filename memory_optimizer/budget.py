"""Shared token-budget logic (brain.md D1/D2/D7).

Both the live pipeline (server._ingest_turn) and the offline E5 replay must apply
the *same* budget semantics, so this module is the single source of truth:

- token_budget_evict: enforce a hard token cap on the adaptive store by evicting
  the lowest-current-importance memories (oldest source turn as tie-break),
  never overshooting the budget. Evidence: SF-AMS utility-driven survival
  (arXiv:2607.22562), MemoryBank fade curves (AAAI 2024), MemGPT
  evict-under-pressure (arXiv:2310.08560).
- fit_to_budget: greedy token-aware injection fitting (brain.md D7 fill mode) —
  ranked candidates are selected while they fit, filling the rest of the limit
  with the next-best by rank instead of stopping at top_k.
"""

from typing import Callable, Dict, List, Tuple

FactTokens = Callable[[str], int]


def _tokens(mem: Dict, fact_tokens: FactTokens) -> int:
    if fact_tokens is not None:
        v = fact_tokens(mem.get("fact", ""))
        if v is not None:
            return int(v)
    v = mem.get("measured_tokens")
    return int(v) if v is not None else 0


def _sum_tokens(memories: List[Dict], fact_tokens: FactTokens) -> int:
    return sum(_tokens(m, fact_tokens) for m in memories)


def token_budget_evict(
    memories: List[Dict],
    budget: int,
    fact_tokens: FactTokens = None,
    priority_key: str = "current_importance",
) -> Tuple[List[Dict], List[Dict], int]:
    """Evict lowest-priority memories until the active store fits `budget`.

    Returns (kept, evicted, used_tokens). Eviction keys on the memory's
    `priority_key` (default: current importance — dynamic activation), breaking
    ties toward the oldest source turn so recent, reinforced memories survive
    the squeeze.

    Phase 11/D32: under ``retention.mode = dual_score`` the caller passes
    ``priority_key="retention_priority"`` (stable, evidence-based long-term
    survival score), so store-budget eviction no longer deletes an old-but-
    valuable fact just because its activation faded. The tie-break is unchanged
    (oldest source_turn_id) for both keys.
    """
    kept = [m for m in memories]
    used = _sum_tokens(kept, fact_tokens)
    evicted: List[Dict] = []
    while used > budget and kept:
        kept.sort(
            key=lambda m: (
                m.get(priority_key, m.get("base_score", 0)),
                m.get("source_turn_id", 0),
            )
        )
        victim = kept.pop(0)
        priority = victim.get(priority_key, victim.get("base_score", 0))
        victim["eviction_reason"] = {
            "reason": "budget_squeeze",
            "detail": (f"store {used}/{budget} tokens exceeds the {budget}-token budget; "
                       f"evicted lowest {priority_key}={priority:.3f}"),
            "used_tokens": used,
            "budget": budget,
            "priority_key": priority_key,
        }
        evicted.append(victim)
        used -= _tokens(victim, fact_tokens)
    return kept, evicted, used


def fit_to_budget(
    ranked: List[Dict],
    token_limit: int,
    fact_tokens: FactTokens = None,
) -> Tuple[List[Dict], List[Dict], int]:
    """Greedy token-aware selection of injected facts (D7 fill mode).

    Selects ranked candidates while their measured sizes fit `token_limit`.
    The injection limit is a hard cap: an oversized top candidate is skipped
    rather than being injected over budget. Returns (selected, skipped, used_tokens).
    """
    selected: List[Dict] = []
    skipped: List[Dict] = []
    used = 0
    for m in ranked:
        size = _tokens(m, fact_tokens)
        if used + size <= token_limit:
            selected.append(m)
            used += size
        else:
            skipped.append(m)
    return selected, skipped, used
