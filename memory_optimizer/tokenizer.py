"""Shared token-counting utility for E17 (Phase 13 / D34).

The repository has no exact model tokenizer available offline (no tiktoken, and
Ollama exposes no tokenize endpoint), so E17 uses a single shared word-count
counter for *every* method. Using one utility everywhere — rather than one
baseline counting characters and another counting words — keeps the fixed
historical-context budget internally fair. The counter name is recorded in the
experiment manifest so reviewers know the budget unit.

This module does not change any production memory behavior; it is a measurement
helper only.
"""

from __future__ import annotations

TOKENIZER_NAME = "shared_word_count"


def count_tokens(text) -> int:
    """Deterministic shared token estimate (word count, minimum 1)."""
    return max(1, len(str(text).split()))


def count_tokens_many(texts) -> int:
    return sum(count_tokens(t) for t in texts)
