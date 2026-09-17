from typing import Dict, List


def _overlap(a: str, b: str) -> float:
    a_set = set(a.lower().split())
    b_set = set(b.lower().split())
    exact = a_set.intersection(b_set)
    if not exact:
        exact = {x for x in a_set for y in b_set if x in y or y in x}
    return len(exact) / (max(len(a_set), 1))


def _cosine(a, b):
    if a is None or b is None:
        return None
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


_SUPERSESSION_MARKERS = (
    "no longer",
    "revised requirement",
    "correction to requirement",
    "update to earlier",
    "not the case",
    " not ",
    " never ",
    "supersedes",
)


def _is_supersession(old_text: str, new_text: str) -> bool:
    """Classify a near-duplicate as a supersession when the new text
    contains a revision/negation marker and restates the old fact.

    Uses the same word-set splitting as ``_overlap`` (consistent tokenization).
    A real supersession re-states the old fact essentially verbatim (e.g.
    "Revised requirement: {old} is no longer the case" or "Correction to
    requirement: NOT {old}"), so ALL of the old fact's words must be covered by
    the new text (== 1.0). Anything short of full coverage leaves category
    template collisions indistinguishable from true corrections (e.g. a fresh
    "NOT … key_27 …" vs a stored "… key_22 …" that share only template words
    reach 0.909 coverage); replacing a stored fact with an unrelated statement
    would corrupt the memory rather than update it. Mirrors the SAT / Mem0
    update semantics: a correction replaces the stored fact rather than
    reinforcing it as a duplicate.
    """
    if not old_text or not new_text:
        return False
    lo_new = new_text.lower()
    if not any(marker in lo_new for marker in _SUPERSESSION_MARKERS):
        return False
    old_words = set(old_text.lower().split())
    if not old_words:
        return False
    new_words = set(lo_new.split())
    return len(old_words & new_words) / len(old_words) == 1.0


def _has_supersession_marker(text: str) -> bool:
    """True when the text carries any revision/negation marker (supersession
    *candidate*), regardless of whether it restates a stored fact."""
    return bool(text) and any(marker in text.lower() for marker in _SUPERSESSION_MARKERS)


class MemoryCompressor:
    """Deduplicates redundant atomic facts within similar clusters.

    Per brain.md D5 research (summarization drift in arXiv:2603.07670;
    "deduplicate, don't summarize" in arXiv:2605.08538; Mem0 add/update/noop in
    the VanillaRAG/Mem0 line of work): consolidation is done by *semantic
    dedupe*, not by LLM summarization. Deduping a fact counts as a re-mention:
    it reinforces the surviving memory (access_count++ and last_access_turn
    refresh) just like a retrieval would.
    """

    def dedupe_incremental(self, existing: List[Dict], new_items: List[Dict],
                           overlap_threshold: float = 0.85, embed_fn=None) -> List[Dict]:
        """Merge only newly ingested facts into the existing store (O(new x existing)).

        Same-category facts merge if their embeddings agree (cos >= 0.90) or,
        when embeddings are unavailable for the pair, if lexical overlap clears
        the threshold. The surviving fact is reinforced (re-mention semantics)
        and keeps embeddings on the merged item when one was computed.

        A merge that also reads as a *correction* (``_is_supersession``) replaces
        the stale fact text instead of merely reinforcing it: the corrected text
        becomes authoritative, the prior text is captured in
        ``superseded_prior_fact``, and confidence follows the new statement. The
        correction detector runs regardless of whether the merge similarity came
        from embeddings or lexical overlap.
        """
        for mem in new_items:
            merged = False
            mem_emb = None
            if embed_fn is not None:
                try:
                    mem_emb = embed_fn([mem["fact"]])
                    mem_emb = mem_emb[0] if isinstance(mem_emb, (list, tuple)) and len(mem_emb) == 1 else mem_emb
                except Exception:
                    mem_emb = None
            if mem_emb is not None:
                mem["fact_embedding"] = mem_emb

            # --- correction-specific targeting ---
            # A fact that carries ``supersedes_turn`` explicitly identifies which
            # stored memory it replaces.  Merge with that memory directly rather
            # than falling through to the generic first-overlap-match loop, which
            # may mistakenly absorb a different memory in the same category that
            # happens to share high lexical overlap (e.g. two different "Revised
            # requirement: … is no longer the case" entries).
            target_turn = mem.get("supersedes_turn")
            if target_turn is not None:
                for ex in existing:
                    if (ex.get("source_turn_id") == target_turn
                            and ex["category"] == mem["category"]):
                        ex_emb = ex.get("fact_embedding")
                        ex["access_count"] = ex.get("access_count", 1) + mem.get("access_count", 1)
                        ex["ingest_reinforcement_count"] = ex.get("ingest_reinforcement_count", 0) + 1
                        ex["last_access_turn"] = mem.get("last_access_turn", ex.get("last_access_turn",
                                                                                      ex.get("source_turn_id")))
                        ex["duplicates"] = ex.get("duplicates", 1) + 1
                        if ex_emb is None and mem_emb is not None:
                            ex["fact_embedding"] = mem_emb
                        ex["superseded_prior_fact"] = ex["fact"]
                        ex["superseded_prior_fact_id"] = ex.get("fact_id")
                        ex["fact_id"] = mem.get("fact_id", ex.get("fact_id"))
                        ex["is_current_correction"] = True
                        ex["fact"] = mem["fact"]
                        ex["confidence"] = mem.get("confidence", ex.get("confidence", 0.5))
                        merged = True
                        break
                if merged:
                    continue

            for ex in existing:
                if ex["category"] != mem["category"]:
                    continue
                ex_emb = ex.get("fact_embedding")
                sim = None
                semantic = False
                if mem_emb is not None and ex_emb is not None:
                    cos = _cosine(mem_emb, ex_emb)
                    if cos is not None:
                        sim = cos
                        semantic = True
                else:
                    sim = _overlap(ex["fact"], mem["fact"])
                threshold = 0.90 if semantic else overlap_threshold
                if sim is not None and sim >= threshold:
                    if _is_supersession(ex["fact"], mem["fact"]):
                        ex["access_count"] = ex.get("access_count", 1) + mem.get("access_count", 1)
                        ex["ingest_reinforcement_count"] = ex.get("ingest_reinforcement_count", 0) + 1
                        ex["last_access_turn"] = mem.get("last_access_turn", ex.get("last_access_turn",
                                                                                      ex.get("source_turn_id")))
                        ex["duplicates"] = ex.get("duplicates", 1) + 1
                        if ex_emb is None and mem_emb is not None:
                            ex["fact_embedding"] = mem_emb
                        ex["superseded_prior_fact"] = ex["fact"]
                        ex["superseded_prior_fact_id"] = ex.get("fact_id")
                        ex["fact_id"] = mem.get("fact_id", ex.get("fact_id"))
                        ex["is_current_correction"] = True
                        ex["fact"] = mem["fact"]
                        ex["confidence"] = mem.get("confidence", ex.get("confidence", 0.5))
                        merged = True
                        break
                    if not semantic and _has_supersession_marker(mem["fact"]):
                        continue
                    ex["access_count"] = ex.get("access_count", 1) + mem.get("access_count", 1)
                    ex["ingest_reinforcement_count"] = ex.get("ingest_reinforcement_count", 0) + 1
                    ex["last_access_turn"] = mem.get("last_access_turn", ex.get("last_access_turn",
                                                                                  ex.get("source_turn_id")))
                    ex["duplicates"] = ex.get("duplicates", 1) + 1
                    if ex_emb is None and mem_emb is not None:
                        ex["fact_embedding"] = mem_emb
                    ex["confidence"] = max(ex.get("confidence", 0.5), mem.get("confidence", 0.5))
                    merged = True
                    break
            if not merged:
                existing.append(dict(mem))
        return existing