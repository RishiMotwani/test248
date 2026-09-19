"""Offline tests for the Phase 22 E27 calibration task suite.

No Ollama / no network / no embeddings. These pin the deterministic fixture
contract: 3 groups x 2 variants; exactly 1 obsolete policy fact + 1 current
correction + 4 distractors per variant; obsolete facts at turns 80-150 (age
> 300) sized 50-56 shared-word tokens, current corrections at 470-530 sized
50-56 tokens and summed with the obsolete fact > 96 (never both in budget),
distractors 18-24 tokens placed near 130/250/360/560; structured correction
metadata; byte-identical shared workspace/prompt across variants; deterministic
seed-1 history matching the committed ``history.txt``; no policy mapping in the
visible surface; no hidden-test language in history; and the offline
base/gold/obsolete hidden gates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments import coding_benchmark as cb
from data import e27_calibration_suite as e27

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_suite_self_test():
    e27.self_test(require_fixtures=True)


def test_three_groups_two_variants_each():
    assert e27.list_group_ids() == ["route_contract", "serialization_contract",
                                    "retry_contract"]
    for gid in e27.list_group_ids():
        assert e27.list_variants(gid) == ["A", "B"]


def test_fixture_directories_are_all_materialised():
    expected_files = 0
    for gid in e27.list_group_ids():
        assert e27.prompt_path(gid).is_file()
        for variant in e27.list_variants(gid):
            vdir = e27.variant_dir(gid, variant)
            for rel in ("history.txt", "metadata.json", "gold.patch",
                        "obsolete.patch", "hidden/test_hidden.py"):
                assert (vdir / rel).is_file(), f"{gid}/{variant}/{rel}"
            expected_files += 5
    assert expected_files == 30


# ---------------------------------------------------------------------------
# Single-policy counts / fact sizing / placement / metadata
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gid", e27.list_group_ids())
@pytest.mark.parametrize("variant", ["A", "B"])
def test_single_policy_counts(gid, variant):
    task = e27.build_variant(gid, variant, seed=1)
    assert len(task.obsolete_policy_facts) == 1
    assert len(task.current_facts) == 1
    assert len(task.distractor_facts) == 4
    assert len(task.corrections) == 1
    assert e27.fact_token_counts(task).keys() == (
        {f.fact_id for f in task.all_policy_facts}
        | {f.fact_id for f in task.distractor_facts})


@pytest.mark.parametrize("gid", e27.list_group_ids())
@pytest.mark.parametrize("variant", ["A", "B"])
def test_fact_token_bands_and_conflict_sum(gid, variant):
    task = e27.build_variant(gid, variant, seed=1)
    obs_t, cur_t = e27.policy_token_counts(task)
    assert e27.OBSOLETE_MIN_TOKENS <= obs_t <= e27.OBSOLETE_MAX_TOKENS
    assert e27.CURRENT_MIN_TOKENS <= cur_t <= e27.CURRENT_MAX_TOKENS
    assert obs_t + cur_t > e27.MIN_OBS_CURRENT_SUM
    for f in task.distractor_facts:
        assert e27.DISTRACTOR_MIN_TOKENS <= len(f.text.split()) \
            <= e27.DISTRACTOR_MAX_TOKENS


@pytest.mark.parametrize("gid", e27.list_group_ids())
@pytest.mark.parametrize("variant", ["A", "B"])
def test_fact_placement_and_obsolete_age(gid, variant):
    task = e27.build_variant(gid, variant, seed=1)
    obs_turn = task.obsolete_policy_facts[0].turn_index
    cur_turn = task.current_facts[0].turn_index
    dist_turns = [f.turn_index for f in task.distractor_facts]
    windows = [(120, 140), (240, 260), (350, 370), (550, 570)]
    assert e27.OBSOLETE_MIN_TURN <= obs_turn <= e27.OBSOLETE_MAX_TURN
    assert (task.history_turns - obs_turn) > e27.MIN_OBSOLETE_AGE_TURNS
    assert e27.CURRENT_MIN_TURN <= cur_turn <= e27.CURRENT_MAX_TURN
    assert all(any(lo <= t <= hi for lo, hi in windows) for t in dist_turns)
    assert len(task.history) == e27.HISTORY_TURNS_DEFAULT


@pytest.mark.parametrize("gid", e27.list_group_ids())
@pytest.mark.parametrize("variant", ["A", "B"])
def test_correction_metadata_structured(gid, variant):
    task = e27.build_variant(gid, variant, seed=1)
    by_id = {fact["fact_id"]: fact
             for entry in task.history for fact in entry.get("facts", [])}
    for c in task.corrections:
        cur = by_id[c.current_fact_id]
        obs = by_id[c.obsolete_fact_id]
        assert cur["is_correction_target"] is True
        assert cur["is_current_correction"] is True
        assert cur["category"] == "project_context"
        assert cur["supersedes_turn"] == obs["source_turn_id"]
        assert cur["superseded_fact"] == obs["fact"]
        assert cur["superseded_prior_fact_id"] == c.obsolete_fact_id
        assert obs["superseded_by"] == c.current_fact_id


# ---------------------------------------------------------------------------
# Identity across variants + hygiene
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gid", e27.list_group_ids())
def test_workspace_and_prompt_identical_across_variants(gid):
    task_a = e27.build_variant(gid, "A", seed=1)
    task_b = e27.build_variant(gid, "B", seed=1)
    assert task_a.task_prompt == task_b.task_prompt
    assert cb.workspace_context(task_a) == cb.workspace_context(task_b)
    assert task_a.metadata["workspace_sha"] == task_b.metadata["workspace_sha"]
    assert task_a.metadata["prompt_sha"] == task_b.metadata["prompt_sha"]
    assert task_a.metadata["history_sha"] != task_b.metadata["history_sha"]


@pytest.mark.parametrize("gid", e27.list_group_ids())
def test_no_policy_mapping_in_visible_surface(gid):
    visible = e27._workspace_prompt_text(gid)
    for term in e27._GROUP_BY_ID[gid].forbidden_terms:
        assert term not in visible


@pytest.mark.parametrize("gid", e27.list_group_ids())
@pytest.mark.parametrize("variant", ["A", "B"])
def test_covert_history_terms_not_in_no_history_prompt(gid, variant):
    prompt = e27.no_history_prompt(gid, variant)
    for term in e27._GROUP_BY_ID[gid].forbidden_terms:
        assert term not in prompt
    assert "test_hidden" not in prompt


@pytest.mark.parametrize("gid", e27.list_group_ids())
@pytest.mark.parametrize("variant", ["A", "B"])
def test_no_hidden_test_language_in_history(gid, variant):
    task = e27.build_variant(gid, variant, seed=1)
    assert e27._check_history_text_hygiene(task) == []


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gid", e27.list_group_ids())
@pytest.mark.parametrize("variant", ["A", "B"])
def test_history_regeneration_is_stable(gid, variant):
    task = e27.build_variant(gid, variant, seed=1)
    fresh = e27.build_history_text(gid, variant)
    assert "\n".join(fresh) + "\n" == e27.lineage_read(gid, variant)
    again = e27.build_history_text(gid, variant)
    assert fresh == again
    assert len(task.history) == e27.HISTORY_TURNS_DEFAULT


# ---------------------------------------------------------------------------
# as_coding_task mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gid", e27.list_group_ids())
@pytest.mark.parametrize("variant", ["A", "B"])
def test_as_coding_task_maps_facts_and_corrections(gid, variant):
    task = e27.build_variant(gid, variant, seed=1)
    ctask = e27.as_coding_task(task)
    obsolete_ids = {f.fact_id for f in ctask.gold_facts if f.obsolete}
    current_ids = {f.fact_id for f in ctask.gold_facts if not f.obsolete}
    assert obsolete_ids == {f.fact_id for f in task.obsolete_policy_facts}
    assert current_ids == {f.fact_id for f in task.current_facts}
    assert {c.obsolete_fact_id for c in ctask.corrections} == obsolete_ids
    assert {c.current_fact_id for c in ctask.corrections} == current_ids
    assert len(ctask.distractor_facts) == 4


# ---------------------------------------------------------------------------
# Offline fixture gates (base fails / gold passes / obsolete fails)
# ---------------------------------------------------------------------------

def test_offline_validation_all_variants_pass():
    result = e27.validate_all_fixtures()
    assert result["passed"] is True
    for key, res in result["variants"].items():
        assert res["passed"] is True, (key, res["failing"])