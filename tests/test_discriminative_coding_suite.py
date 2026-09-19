"""Offline tests for the Phase 21 E26 discriminative task suite.

No Ollama / no network / no embeddings. These pin the deterministic fixture
contract: 3 groups x 2 variants, 4+4+4 facts, 18-28-token sizing, obsolete
turns in 70..180 with age > 300 turns, current turns in 420..540, structured
correction metadata, byte-identical shared workspace/prompt across variants,
deterministic seed-1 history matching the committed ``history.txt``, no policy
mapping in the visible workspace, and the offline base/gold/obsolete hidden
gates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments import coding_benchmark as cb
from data import discriminative_coding_suite as ds

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_suite_self_test():
    ds.self_test(require_fixtures=True)


def test_three_groups_two_variants_each():
    assert ds.list_group_ids() == ["release_adapter", "invoice_adapter",
                                   "message_adapter"]
    for gid in ds.list_group_ids():
        assert ds.list_variants(gid) == ["A", "B"]


def test_fixture_directories_are_all_materialised():
    expected_files = 0
    for gid in ds.list_group_ids():
        assert ds.prompt_path(gid).is_file()
        for variant in ds.list_variants(gid):
            vdir = ds.variant_dir(gid, variant)
            for rel in ("history.txt", "metadata.json", "gold.patch",
                        "obsolete.patch", "hidden/test_hidden.py"):
                assert (vdir / rel).is_file(), f"{gid}/{variant}/{rel}"
            expected_files += 5
    assert expected_files == 30


# ---------------------------------------------------------------------------
# Fact sizing / placement / metadata
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gid", ds.list_group_ids())
@pytest.mark.parametrize("variant", ["A", "B"])
def test_fact_token_sizes_within_18_to_28(gid, variant):
    task = ds.build_variant(gid, variant, seed=1)
    counts = ds.fact_token_counts(task)
    assert len(counts) == 12
    for fid, n in counts.items():
        assert ds.MIN_FACT_TOKENS <= n <= ds.MAX_FACT_TOKENS, fid


@pytest.mark.parametrize("gid", ds.list_group_ids())
@pytest.mark.parametrize("variant", ["A", "B"])
def test_fact_placement_and_obsolete_age(gid, variant):
    task = ds.build_variant(gid, variant, seed=1)
    current = [f.turn_index for f in task.policy_facts]
    obsolete = [f.turn_index for f in task.obsolete_policy_facts]
    assert len(current) == 4 and len(obsolete) == 4
    assert all(420 <= t <= 540 for t in current)
    assert all(70 <= t <= 180 for t in obsolete)
    assert all((task.history_turns - t) > 300 for t in obsolete)
    assert max(current) - min(obsolete) > 250


@pytest.mark.parametrize("gid", ds.list_group_ids())
@pytest.mark.parametrize("variant", ["A", "B"])
def test_correction_metadata_structured(gid, variant):
    task = ds.build_variant(gid, variant, seed=1)
    by_id = {fact["fact_id"]: fact
             for entry in task.history for fact in entry.get("facts", [])}
    for c in task.corrections:
        cur = by_id[c.current_fact_id]
        obs = by_id[c.obsolete_fact_id]
        assert cur["is_correction_target"] is True
        assert cur["is_current_correction"] is True
        assert cur["supersedes_turn"] == obs["source_turn_id"]
        assert cur["superseded_prior_fact_id"] == c.obsolete_fact_id
        assert obs["superseded_by"] == cur["source_turn_id"]


# ---------------------------------------------------------------------------
# Identity across variants + prompt hygiene
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gid", ds.list_group_ids())
def test_workspace_and_prompt_identical_across_variants(gid):
    task_a = ds.build_variant(gid, "A", seed=1)
    task_b = ds.build_variant(gid, "B", seed=1)
    assert task_a.task_prompt == task_b.task_prompt
    assert cb.workspace_context(task_a) == cb.workspace_context(task_b)
    assert task_a.metadata["workspace_sha"] == task_b.metadata["workspace_sha"]
    assert task_a.metadata["prompt_sha"] == task_b.metadata["prompt_sha"]
    assert task_a.metadata["history_sha"] != task_b.metadata["history_sha"]


@pytest.mark.parametrize("gid", ds.list_group_ids())
def test_no_policy_mapping_in_visible_surface(gid):
    visible = ds._workspace_prompt_text(gid)
    for term in ds._FORBIDDEN_WORKSPACE_TERMS:
        assert term not in visible


@pytest.mark.parametrize("gid", ds.list_group_ids())
@pytest.mark.parametrize("variant", ["A", "B"])
def test_covert_history_terms_not_in_no_history_prompt(gid, variant):
    prompt = ds.no_history_prompt(gid, variant)
    for term in ds._FORBIDDEN_WORKSPACE_TERMS:
        assert term not in prompt
    assert "test_hidden" not in prompt


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gid", ds.list_group_ids())
@pytest.mark.parametrize("variant", ["A", "B"])
def test_history_regeneration_is_stable(gid, variant):
    task = ds.build_variant(gid, variant, seed=1)
    fresh = ds.build_history_text(gid, variant)
    assert "\n".join(fresh) + "\n" == ds.lineage_read(gid, variant)
    again = ds.build_history_text(gid, variant)
    assert fresh == again
    assert len(task.history) == ds.HISTORY_TURNS_DEFAULT


# ---------------------------------------------------------------------------
# as_coding_task mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gid", ds.list_group_ids())
@pytest.mark.parametrize("variant", ["A", "B"])
def test_as_coding_task_maps_facts_and_corrections(gid, variant):
    task = ds.build_variant(gid, variant, seed=1)
    ctask = ds.as_coding_task(task)
    obsolete_ids = {f.fact_id for f in ctask.gold_facts if f.obsolete}
    current_ids = {f.fact_id for f in ctask.gold_facts if not f.obsolete}
    assert obsolete_ids == {f.fact_id for f in task.obsolete_policy_facts}
    assert current_ids == {f.fact_id for f in task.policy_facts}
    assert {c.obsolete_fact_id for c in ctask.corrections} == obsolete_ids
    assert {c.current_fact_id for c in ctask.corrections} == current_ids
    assert len(ctask.distractor_facts) == 4


# ---------------------------------------------------------------------------
# Offline fixture gates (base fails / gold passes / obsolete fails)
# ---------------------------------------------------------------------------

def test_offline_validation_all_variants_pass():
    result = ds.validate_all_fixtures()
    assert result["passed"] is True
    for key, res in result["variants"].items():
        assert res["passed"] is True, (key, res["failing"])