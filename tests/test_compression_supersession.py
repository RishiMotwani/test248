"""Correction supersession tests (Phase 2, requirements CORR-01..CORR-05).

Validates the dedupe-incremental supersession heuristics added to the
MemoryCompressor:

* CORR-01: revision/negation markers + lexical overlap classify a supersession
* CORR-02: on supersession the stale fact text is replaced by the corrected text
* CORR-03: the prior text is retained as ``superseded_prior_fact``
* CORR-04: confidence follows the new (correcting) statement
* CORR-05: access_count / last_access_turn are reinforced from the correction

Also covers the two guards that keep the heuristics honest:

* a fresh negation marked statement that does NOT restate an existing fact is
  stored as its own memory (not absorbed as a duplicate), and
* a ``supersedes_turn`` fact merges directly with the memory anchored to that
  exact turn, rather than falling through to a coincidentally similar one.
"""

import pytest

from memory_optimizer.compression import (
    MemoryCompressor,
    _has_supersession_marker,
    _is_supersession,
)


@pytest.fixture
def compressor():
    return MemoryCompressor()


def _mem(fact, turn=1, category="project_context", confidence=0.5,
         access_count=1):
    return {
        "fact": fact,
        "category": category,
        "confidence": confidence,
        "source_turn_id": turn,
        "last_access_turn": turn,
        "access_count": access_count,
    }


# --------------------------------------------------------------------------- #
# _is_supersession heuristic (CORR-01)
# --------------------------------------------------------------------------- #

class TestIsSupersession:
    def test_correction_marker_and_full_restate_is_supersession(self):
        old = "Project feature flag configuration key_14 is enabled"
        new = ("Revised requirement: Project feature flag configuration key_14 "
               "is enabled is no longer the case")
        assert _is_supersession(old, new) is True

    def test_negation_marker_and_full_restate_is_supersession(self):
        old = "Project feature flag configuration key_14 is enabled"
        new = ("Correction to requirement: NOT project feature flag "
               "configuration key_14 is enabled")
        assert _is_supersession(old, new) is True

    def test_marker_but_partial_overlap_is_not_supersession(self):
        # Cross-entity template collision: only the entity token differs, so a
        # "not key_27" statement must not be read as correcting "key_23".
        old = "Project feature flag configuration key_23 is enabled"
        new = ("Correction to requirement: NOT project feature flag "
               "configuration key_27 is enabled")
        assert _is_supersession(old, new) is False

    def test_no_marker_is_not_supersession(self):
        assert _is_supersession("A is B", "A is B again") is False

    def test_empty_inputs_are_not_supersessions(self):
        assert _is_supersession("", "Revised requirement: x no longer") is False
        assert _is_supersession("x", "") is False


class TestHasSupersessionMarker:
    def test_marker_detected(self):
        assert _has_supersession_marker(
            "Revised requirement: X is no longer the case") is True

    def test_no_marker(self):
        assert _has_supersession_marker("X is enabled") is False

    def test_empty_string(self):
        assert _has_supersession_marker("") is False


# --------------------------------------------------------------------------- #
# dedupe_incremental supersession (CORR-02, CORR-03, CORR-04, CORR-05)
# --------------------------------------------------------------------------- #

class TestDedupeSupersession:
    def test_correction_replaces_stale_fact(self, compressor):
        existing = [_mem("Project feature flag configuration key_14 is enabled",
                         turn=3, confidence=0.6)]
        corrected = ("Revised requirement: Project feature flag configuration "
                     "key_14 is enabled is no longer the case")
        out = compressor.dedupe_incremental(
            existing, [_mem(corrected, turn=20, confidence=0.9)])

        assert len(out) == 1
        assert out[0]["fact"] == corrected
        assert out[0]["superseded_prior_fact"] == \
            "Project feature flag configuration key_14 is enabled"
        # CORR-04: confidence follows the new statement
        assert out[0]["confidence"] == 0.9
        # CORR-05: re-mention semantics reinforce access bookkeeping
        assert out[0]["access_count"] >= 1
        assert out[0]["last_access_turn"] == 20

    def test_negation_supersedes_stored_positive(self, compressor):
        existing = [_mem("Project feature flag configuration key_22 is enabled",
                         turn=7, confidence=0.5)]
        negated = ("Correction to requirement: NOT project feature flag "
                   "configuration key_22 is enabled")
        out = compressor.dedupe_incremental(
            existing, [_mem(negated, turn=8, confidence=0.7)])

        assert len(out) == 1
        assert out[0]["fact"] == negated

    def test_true_duplicate_keeps_original_text_and_max_confidence(self, compressor):
        fact = "Production must run PostgreSQL"
        existing = [_mem(fact, turn=1, confidence=0.6)]
        out = compressor.dedupe_incremental(
            existing, [_mem(fact, turn=9, confidence=0.8)])

        assert len(out) == 1
        # No markers -> reinforced duplicate, original text stays authoritative.
        assert out[0]["fact"] == fact
        assert out[0]["confidence"] == 0.8
        assert "superseded_prior_fact" not in out[0]

    def test_cross_entity_negation_is_stored_as_new_memory(self, compressor):
        # The guard for fresh negation statements: "NOT key_27" does not restate
        # the stored "key_23" fact, so it must not be folded into that memory.
        existing = [_mem("Project feature flag configuration key_23 is enabled",
                         turn=23, confidence=0.5)]
        fresh_negation = ("Correction to requirement: NOT project feature flag "
                          "configuration key_27 is enabled")
        out = compressor.dedupe_incremental(
            existing, [_mem(fresh_negation, turn=27, confidence=0.5)])

        assert len(out) == 2
        new_mem = next(m for m in out if m["fact"] == fresh_negation)
        assert new_mem["source_turn_id"] == 27
        # The original key_23 memory is left untouched (not superseded, not
        # conflated with the unrelated key_27 statement).
        key23 = next(m for m in out if m["fact"].startswith("Project feature"))
        assert "superseded_prior_fact" not in key23

    def test_supersedes_turn_targets_exact_turn(self, compressor):
        # Two same-category candidates; the correcting fact explicitly names the
        # turn it supersedes, so the exact turn must win over the higher-overlap
        # template twin.
        old_actual = "Project feature flag configuration key_28 is enabled"
        template_sibling = "Project feature flag configuration key_35 is enabled"
        existing = [
            _mem(template_sibling, turn=35, confidence=0.6),
            _mem(old_actual, turn=28, confidence=0.6),
        ]
        corrected = ("Revised requirement: Project feature flag configuration "
                     "key_28 is enabled is no longer the case")
        out = compressor.dedupe_incremental(
            existing, [{**_mem(corrected, turn=40, confidence=0.9),
                        "supersedes_turn": 28}])

        assert len(out) == 2
        corrected_mem = next(m for m in out if m["source_turn_id"] == 28)
        assert corrected_mem["fact"] == corrected
        assert corrected_mem["superseded_prior_fact"] == old_actual
        # The template sibling is left alone.
        sibling = next(m for m in out if m["source_turn_id"] == 35)
        assert sibling["fact"] == template_sibling