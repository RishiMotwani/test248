"""E17 coding-task suite: real repository-editing tasks with deterministic hidden tests.

Why this exists (Phase 13 / D34)
--------------------------------
Every earlier experiment measured memory-recall or injected-context answerability.
E17 asks the actual research question: *does the adaptive memory system help a real
coding model complete long-running coding tasks better than simpler ways of managing
historical context when the amount of historical context given to the coding model is
fixed?* The dependent variable is therefore **whether the produced patch passes
deterministic hidden tests**, not whether a fact was recalled.

Each task has:

* a small real workspace under ``data/coding_tasks/<task_id>/workspace`` that the
  coding model edits,
* deterministic hidden tests under ``.../<task_id>/hidden`` that the model never
  sees and that fail if a buried historical constraint is violated,
* a ~600-turn historical transcript that buries critical facts among distractors,
* machine-readable gold fact ids, obsolete fact ids and correction relationships.

Historical dependence is genuine: each hidden test exercises a constraint that the
current workspace alone does not make obvious (see the per-task prompt's neutral
wording). The ``no_history`` diagnostic in E17 measures this directly.

Ingestion mode is **policy-controlled / oracle-pre-extracted**: the historical facts
are generated from the task spec, not extracted live by an LLM. This keeps E17 a
test of retention + retrieval + context allocation, not of extraction cost. The
transcript text is still fully generated so raw/summary methods operate on realistic
history.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HISTORY_TURNS_DEFAULT = 600

_TASKS_DIR = Path(__file__).resolve().parent / "coding_tasks"


# ---------------------------------------------------------------------------
# Data model (mission section 41)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HistoricalFact:
    fact_id: str
    turn_index: int
    text: str
    kind: str  # constraint | correction | negative | decision | implementation | distractor
    topic: str
    probe: str
    obsolete: bool = False


@dataclass(frozen=True)
class Correction:
    obsolete_fact_id: str
    current_fact_id: str


@dataclass(frozen=True)
class CodingTask:
    task_id: str
    title: str
    task_prompt: str
    workspace_path: Path
    hidden_test_path: Path
    hidden_test_command: List[str]
    history: List[Dict]          # [{"turn_id", "user", "tokens", "facts": [...]}]
    history_text: List[str]      # raw turn strings (for raw/summary methods)
    gold_facts: List[HistoricalFact]
    distractor_facts: List[HistoricalFact]
    corrections: List[Correction]
    obsolete_fact_ids: List[str]
    distractor_topics: List[str]
    seed: int
    history_turns: int
    metadata: Dict = field(default_factory=dict)

    @property
    def gold_fact_ids(self) -> List[str]:
        return [f.fact_id for f in self.gold_facts]


# ---------------------------------------------------------------------------
# Filler pools — unrelated work + broad distractor chatter (3+ topic families)
# ---------------------------------------------------------------------------

_FILLER: Dict[str, List[str]] = {
    "auth": [
        "Spent the morning rotating the staging service account keys.",
        "Debugged a flaky login test that turned out to be a clock skew issue.",
        "Reviewed the session revocation PR; left comments about token naming.",
        "The OAuth callback handler swallowed an error code last week; fixed now.",
        "Talked through MFA enrollment edge cases with the platform team.",
        "Benchmarked password hashing cost and left the parameters as they are.",
        "Cleaned up the permissions matrix doc after the access review.",
        "Investigated a logout race; added it to the backlog for later.",
        "Renamed a few role constants for readability, no behavior change.",
        "Wrote a small script to dump active sessions for support.",
        "The identity provider had a brief outage; nothing we can do about it.",
        "Refactored the auth middleware test to use fixtures instead of mocks.",
    ],
    "database": [
        "Vacuumed the analytics replica and watched the query plans settle.",
        "One slow join showed up in the query log; added it to the tuning list.",
        "Discussed the migration ordering for the next release; no change yet.",
        "The schema diff tool produced noise again, ignored it.",
        "Talked about read replicas for the reporting workload.",
        "A deadlock showed up under synthetic load; noted the transaction shape.",
        "Added an index on a lookup column in dev to see if it helps.",
        "Reviewed backup retention with the ops team; policy unchanged.",
        "A batch job ran long overnight; scheduling it later solved nothing.",
        "Cleaned up unused columns from a scratch table.",
        "The ORM generated a nested select again; replaced it in one spot.",
        "Discussed partitioning options for the events table, deferred.",
    ],
    "caching": [
        "Compared two serialization formats for the cache values, no decision.",
        "A stale entry caused a confusing test failure; cleared it and moved on.",
        "Talked about cache stampede mitigation for a shared key.",
        "The cache metrics dashboard is missing a panel; filed a ticket.",
        "Discussed key naming conventions for the cache namespace.",
        "Saw a memory spike in the cache process during the load test.",
        "Reviewed TTL settings informally; left them alone.",
        "Someone asked about cache warming on deploy; deferred the discussion.",
        "Fixed a typo in the cache client docstring.",
        "The cache eviction counter looked off by one; traced it to labels.",
        "Talked about whether to persist cache entries across restarts.",
        "Cleared a stale namespace on staging to reproduce a bug.",
    ],
    "generic": [
        "Standup ran long today; mostly status updates.",
        "The CI runner was slow after lunch; eventually recovered.",
        "Cleaned up a pile of stale branches in the repo.",
        "Updated the on-call runbook with a couple of small clarifications.",
        "Discussed the offsite logistics for a few minutes.",
        "The lint config got reordered; no rule changes.",
        "Reopened and re-closed an old issue that was already fixed.",
        "Pair-reviewed a trivial formatting change.",
        "The build cache got cleared to fix a flaky stage.",
        "Talked about the release train schedule, no action items.",
        "Filed a docs ticket for the onboarding guide.",
        "Coffee machine broke again; work continued regardless.",
    ],
}

_FACT_TAG = "Team note: {text}"
_CORRECTION_TAG = "Correction to an earlier note: {text}"


def _filler_line(topic: str, rng: random.Random) -> str:
    return rng.choice(_FILLER[topic])


# ---------------------------------------------------------------------------
# Per-task specifications
# ---------------------------------------------------------------------------
# Each spec carries the workspace/hidden paths, the neutral task prompt, the
# critical gold facts, additive distractor facts, and corrections. Turn indices
# are fixed so facts are spread across the whole session (never clustered at the
# end) and never carry IMPORTANT/CRITICAL markers.

def _f(fid, turn, text, kind, topic, probe, obsolete=False) -> Dict:
    return dict(fact_id=fid, turn_index=turn, text=text, kind=kind, topic=topic,
                probe=probe, obsolete=obsolete)


_TASK_SPECS: List[Dict] = [
    {
        "task_id": "cache_readonly",
        "title": "Implement a cache write helper (early architectural constraint)",
        "dir": "cache_readonly",
        "prompt": (
            "Implement the function `set_cached(store, key, value)` in "
            "`cacheapp/handlers.py`. It must store `value` under `key` in the "
            "supplied `CacheStore` and return the stored value. Keep `handle_get` "
            "and all existing public signatures unchanged. Do not modify the "
            "hidden tests."
        ),
        "critical": [
            _f("task_a.old.001", 37,
               "For initial velocity, request handlers may write directly to the "
               "cache store while the audit path is being built.",
               "decision", "caching",
               "request handlers may write directly to the cache store",
               obsolete=True),
            _f("task_a.arch.001", 260,
               "Correction to an earlier note: for initial velocity, request "
               "handlers may no longer write directly to the cache store while the "
               "audit path is being built. All cache mutation must go through "
               "CacheCoordinator; a handler must never write to the store's "
               "internal data dict.",
               "negative", "caching",
               "all cache mutation must go through CacheCoordinator"),
        ],
        "distractors": [
            _f("task_a.d1", 120, "The cache value serializer defaults to JSON for "
               "compatibility with the older clients.", "implementation", "caching",
               "serializer defaults to JSON"),
            _f("task_a.d2", 330, "The cache namespace is prefixed per environment to "
               "avoid cross-talk on staging.", "decision", "caching",
               "prefixed per environment"),
            _f("task_a.d3", 430, "Auth middleware caches the verified key material for "
               "the lifetime of a request.", "implementation", "auth",
               "caches the verified key material"),
            _f("task_a.d4", 520, "The database connection pool is warmed on startup.",
               "implementation", "database", "connection pool is warmed"),
        ],
        "corrections": [
            ("task_a.old.001", "task_a.arch.001"),
        ],
    },
    {
        "task_id": "user_ids",
        "title": "Implement a user storage key (correction / supersession)",
        "dir": "user_ids",
        "prompt": (
            "Implement `user_key(uid)` in `idapp/users.py` and make "
            "`normalize_user_id` consistent with it: both must return the same "
            "canonical storage key for a user id. Keep the function names and "
            "signatures. Do not modify the hidden tests."
        ),
        "critical": [
            _f("task_b.old.001", 80,
               "User IDs are stored as integers in the users table, so callers cast "
               "them on the way in.",
               "decision", "database",
               "callers cast them on the way in", obsolete=True),
            _f("task_b.correct.001", 260,
               "Correction to an earlier note: user IDs are no longer stored as "
               "integers in the users table, so callers do not cast them on the way "
               "in. User IDs are opaque strings; preserve the client-provided string "
               "exactly.",
               "correction", "database", "opaque strings"),
        ],
        "distractors": [
            _f("task_b.d1", 150, "The users table gained a nullable display_name "
               "column in the last migration.", "implementation", "database",
               "nullable display_name"),
            _f("task_b.d2", 330, "Session lookup uses the account email as a secondary "
               "index.", "implementation", "auth", "account email as a secondary index"),
            _f("task_b.d3", 470, "Audit rows reference the actor by external id.",
               "decision", "database", "actor by external id"),
        ],
        "corrections": [
            ("task_b.old.001", "task_b.correct.001"),
        ],
    },
    {
        "task_id": "write_retry",
        "title": "Implement a record submitter (negative constraint)",
        "dir": "write_retry",
        "prompt": (
            "Implement `submit(client, payload)` in `writeapp/writer.py`. It must "
            "send the payload through `write_record` and make the writer resilient "
            "to transient upstream failures. On `UpstreamError` from the upstream, "
            "raise `WriteFailed`; on success, return the upstream result. Keep "
            "existing signatures. Do not modify the hidden tests."
        ),
        "critical": [
            _f("task_c.neg.001", 55,
               "Do not retry writes automatically because the upstream write is not "
               "idempotent. A failed write must be surfaced to the caller after a "
               "single attempt.",
               "negative", "database", "not idempotent"),
            _f("task_c.neg.002", 300,
               "The upstream client must be called exactly once per submit; sending "
               "the same payload twice creates a duplicate record.",
               "negative", "database", "exactly once"),
        ],
        "distractors": [
            _f("task_c.d1", 140, "The metrics uploader retries up to five times with "
               "exponential backoff because it is idempotent.", "implementation",
               "auth", "retries up to five times"),
            _f("task_c.d2", 210, "The webhook fan-out worker retries failed deliveries "
               "on a schedule.", "implementation", "auth", "retries failed deliveries"),
            _f("task_c.d3", 380, "The config loader retries the remote fetch on a "
               "network timeout.", "implementation", "caching", "retries the remote fetch"),
            _f("task_c.d4", 520, "The billing reconciler retries idempotent ledger "
               "writes.", "implementation", "database", "retries idempotent ledger writes"),
        ],
        "corrections": [],
    },
    {
        "task_id": "validation_pure",
        "title": "Add strict config validation (long-range purity constraint)",
        "dir": "validation_pure",
        "prompt": (
            "Add strict validation for the config path. Implement "
            "`validate_config(config)` in `parseapp/validator.py`: it raises "
            "`ValueError` if any value is an empty string and returns `True` "
            "otherwise. Wire it into `parseapp/service.py` so `load_config` runs it "
            "after `validate_required`. Keep existing signatures. Do not modify the "
            "hidden tests."
        ),
        "critical": [
            _f("task_d.arch.001", 35,
               "The repository deliberately separates parsing from validation. "
               "Validation must remain pure and side-effect free: no audit writes, "
               "no database access, and no mutation of shared state during "
               "validation.",
               "negative", "database", "side-effect free"),
            _f("task_d.old.001", 170,
               "Validation records every check in the AUDIT log for observability.",
               "decision", "database", "validation records every check",
               obsolete=True),
            _f("task_d.correct.001", 480,
               "Correction to an earlier note: validation no longer records every "
               "check in the AUDIT log for observability. Validation must not write "
               "to AUDIT or any shared state and must remain side-effect free; "
               "instrumentation belongs in the service layer, not in validators.",
               "correction", "database", "must not write to AUDIT"),
        ],
        "distractors": [
            _f("task_d.d1", 240, "The parser ignores unknown keys to stay "
               "forward-compatible.", "decision", "database",
               "ignores unknown keys"),
            _f("task_d.d2", 400, "The service layer logs a structured event for every "
               "load_config call.", "implementation", "database",
               "structured event for every load_config"),
            _f("task_d.d3", 540, "The config cache validates checksums before reuse.",
               "implementation", "caching", "validates checksums"),
        ],
        "corrections": [
            ("task_d.old.001", "task_d.correct.001"),
        ],
    },
]


# ---------------------------------------------------------------------------
# Transcript construction
# ---------------------------------------------------------------------------

_CATEGORY_BY_KIND = {
    "constraint": "project_context",
    "negative": "project_context",
    "correction": "project_context",
    "decision": "project_context",
    "implementation": "technical_preference",
    "distractor": "transient",
}


def _word_tokens(text: str) -> int:
    return max(1, len(str(text).split()))


def _fact_turn_message(fact: Dict) -> str:
    template = _CORRECTION_TAG if fact["kind"] == "correction" else _FACT_TAG
    return template.format(text=fact["text"])


def _build_history(spec: Dict, seed: int, turns: int) -> Tuple[List[Dict], List[str]]:
    """Deterministically interleave facts and filler across ``turns`` turns."""
    rng = random.Random(seed)
    facts_by_turn: Dict[int, List[Dict]] = {}
    for f in list(spec["critical"]) + list(spec["distractors"]):
        turn = int(f["turn_index"])
        if turn >= turns:
            raise ValueError(f"{spec['task_id']}: fact {f['fact_id']} turn {turn} >= {turns}")
        facts_by_turn.setdefault(turn, []).append(f)

    topics = ["auth", "database", "caching", "generic"]
    history: List[Dict] = []
    history_text: List[str] = []
    for turn_id in range(1, turns + 1):
        here = facts_by_turn.get(turn_id, [])
        if here:
            user = " ".join(_fact_turn_message(f) for f in here)
            facts = [{
                "fact": f["text"],
                "category": _CATEGORY_BY_KIND.get(f["kind"], "transient"),
                "source_turn_id": turn_id,
                "fact_id": f["fact_id"],
            } for f in here]
        else:
            topic = topics[turn_id % len(topics)]
            user = f"Turn {turn_id}: {_filler_line(topic, rng)}"
            facts = []
        history.append({"turn_id": turn_id, "user": user,
                        "tokens": _word_tokens(user), "facts": facts})
        history_text.append(user)
    return history, history_text


def _facts_from_dicts(raws: List[Dict]) -> List[HistoricalFact]:
    return [HistoricalFact(**r) for r in raws]


def build_task(task_id: str, seed: int = 42,
               history_turns: int = HISTORY_TURNS_DEFAULT) -> CodingTask:
    spec = next((s for s in _TASK_SPECS if s["task_id"] == task_id), None)
    if spec is None:
        raise KeyError(f"unknown task_id {task_id!r}; options: {list_task_ids()}")
    history, history_text = _build_history(spec, seed, history_turns)
    base = _TASKS_DIR / spec["dir"]
    workspace = base / "workspace"
    hidden = base / "hidden" / "test_hidden.py"
    if not workspace.is_dir():
        raise FileNotFoundError(f"missing workspace for {task_id}: {workspace}")
    if not hidden.is_file():
        raise FileNotFoundError(f"missing hidden test for {task_id}: {hidden}")
    import sys
    return CodingTask(
        task_id=task_id,
        title=spec["title"],
        task_prompt=spec["prompt"],
        workspace_path=workspace,
        hidden_test_path=hidden,
        hidden_test_command=[sys.executable, "-m", "pytest", "test_hidden.py", "-q"],
        history=history,
        history_text=history_text,
        gold_facts=_facts_from_dicts(spec["critical"]),
        distractor_facts=_facts_from_dicts(spec["distractors"]),
        corrections=[Correction(o, c) for o, c in spec["corrections"]],
        obsolete_fact_ids=[f["fact_id"] for f in spec["critical"] if f.get("obsolete")]
        + [f["fact_id"] for f in spec["distractors"] if f.get("obsolete")],
        distractor_topics=["auth", "database", "caching"],
        seed=seed,
        history_turns=history_turns,
        metadata={
            "ingestion": "oracle_pre_extracted",
            "ingestion_note": ("historical facts are generated from the task spec, "
                               "not extracted live; E17 measures retention + "
                               "retrieval + context allocation"),
            "visible_workspace_files": sorted(
                str(p.relative_to(workspace)) for p in workspace.rglob("*")
                if p.is_file()),
        },
    )


def list_task_ids() -> List[str]:
    return [s["task_id"] for s in _TASK_SPECS]


def self_test() -> None:
    """Determinism + task-shape sanity, no pipeline or LLM calls."""
    assert len(list_task_ids()) >= 4, "expected at least 4 tasks"
    turns = HISTORY_TURNS_DEFAULT
    n_corr = n_neg = n_long = 0
    for tid in list_task_ids():
        t1 = build_task(tid, seed=42, history_turns=turns)
        t2 = build_task(tid, seed=42, history_turns=turns)
        assert t1.history == t2.history, f"{tid} not deterministic"
        assert len(t1.history) == turns
        assert len(t1.gold_facts) >= 2, f"{tid} needs >=2 critical facts"
        assert len(set(t1.gold_fact_ids)) == len(t1.gold_fact_ids), f"{tid} dup gold ids"
        for f in t1.gold_facts:
            assert 0 < f.turn_index < turns, f"{tid}:{f.fact_id} turn out of range"
            assert f.probe.lower() in f.text.lower(), (
                f"{tid}:{f.fact_id} probe not in text")
        if t1.corrections:
            n_corr += 1
            ids = {f.fact_id for f in t1.gold_facts}
            for c in t1.corrections:
                assert c.obsolete_fact_id in t1.obsolete_fact_ids
                assert c.current_fact_id in ids
        if any(f.kind == "negative" for f in t1.gold_facts):
            n_neg += 1
        if any((turns - f.turn_index) > 300 for f in t1.gold_facts):
            n_long += 1
    assert n_corr >= 2, "need >=2 tasks with corrections"
    assert n_neg >= 2, "need >=2 tasks with negative constraints"
    assert n_long >= 2, "need >=2 tasks with >300-turn-old critical facts"
    print("coding_task_suite self-test OK")


if __name__ == "__main__":
    self_test()
