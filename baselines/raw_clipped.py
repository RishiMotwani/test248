"""B: raw-clipped baseline — the real "barebone" control for E17.

Returns the newest raw historical turns that fit entirely inside a token budget.
It never summarizes, never retrieves semantically, never reorders history and
never inspects the current task to select older turns. This is the direct
raw-history-clipping control that earlier experiments only approximated as a
curve rather than as a standalone evaluated method.

The function is deliberately pure and dependency-free so the same implementation
can be reused by tests and by the E17 method adapter.
"""

from __future__ import annotations

from typing import List

from memory_optimizer.tokenizer import count_tokens


def build_context(history: List[str], token_budget: int) -> str:
    """Return the newest raw turns that fit inside ``token_budget``.

    Start from the newest turn, walk backward adding turns while the total stays
    within the budget, and return them in chronological order. If the next turn
    does not fit, stop (never partially include a turn).
    """
    budget = int(token_budget)
    if budget <= 0:
        return ""
    kept: List[str] = []
    used = 0
    for turn in reversed(list(history)):
        cost = count_tokens(turn)
        if used + cost > budget:
            break
        kept.append(turn)
        used += cost
    kept.reverse()
    return "\n".join(kept)
