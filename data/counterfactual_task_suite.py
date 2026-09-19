"""E20 counterfactual-history task suite (Phase 16 / benchmark calibration).

Purpose
-------
Phase 15 (E19) failed gate C: the pilot could not establish that its coding
tasks were *genuinely* dependent on historical information — some fixtures were
solvable from the workspace alone. E20 therefore calibrates the benchmark with
**counterfactual history pairs**:

    same workspace
    same task prompt
    different historical engineering decision
    different hidden test / expected behavior
    different gold patch

If a model with access to the correct variant's historical record can solve the
variant while a model with *no* history cannot, the benchmark has demonstrated
that historical information changes task solvability. This suite only models
the fixtures; the experiment lives in ``experiments/e20_counterfactual_history.py``.

Three groups, each with two byte-identical workspaces+prompts:

* ``routing_policy`` — opaque operations routed to lanes per the historical
  allocation table (finalized table A) vs the inverse allocation (B)
* ``retry_policy``       — no auto-retry, non-idempotent (A) vs retry exactly once (B)
* ``serialization_policy`` — preserve unknown keys (A) vs drop unknown keys (B)

Every variant reuses the E17 ``CodingTask`` data model and the deterministic
transcript conventions from ``data/coding_task_suite.py`` (facts placed at fixed
turns among multi-topic filler; corrections marked ``Correction to an earlier
note:``; final decision >300 turns old). Histories are committed as readable
``history.txt`` files; ``build_variant`` verifies a freshly regenerated transcript
is byte-identical to the committed history so the fixtures cannot silently drift.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from data.coding_task_suite import (
    _build_history,
    _facts_from_dicts,
    _word_tokens,
    Correction,
    CodingTask,
    HistoricalFact,
)

_CF_DIR = Path(__file__).resolve().parent / "counterfactual_tasks"
HISTORY_TURNS_DEFAULT = 600


# ---------------------------------------------------------------------------
# Specs (single source of truth for the fixtures)
# ---------------------------------------------------------------------------

def _f(fid, turn, text, kind, topic, probe, obsolete=False) -> Dict:
    return dict(fact_id=fid, turn_index=turn, text=text, kind=kind, topic=topic,
                probe=probe, obsolete=obsolete)


@dataclass(frozen=True)
class CounterfactualVariant:
    variant: str            # "A" or "B"
    dir: str                # "variant_a" / "variant_b"
    decision_summary: str   # one-line human summary of the decision
    critical: List[Dict]    # gold facts (+ obsolete intermediates)
    distractors: List[Dict]
    corrections: List[Tuple[str, str]]
    gold: Dict[str, str]    # relative path -> complete final file contents


@dataclass(frozen=True)
class CounterfactualGroup:
    group_id: str
    dir: str
    prompt: str
    workspace: Dict[str, str]
    variants: List[CounterfactualVariant]


_GROUPS: List[CounterfactualGroup] = [
    CounterfactualGroup(
        group_id="routing_policy",
        dir="routing_policy",
        prompt=(
            "Implement `select_route(operation)` in `routeapp/router.py` "
            "according to the repository's established contract. Keep the "
            "existing public name and signature. Do not modify hidden tests."
        ),
        workspace={
            "routeapp/__init__.py": (
                '"""routeapp - routing helpers used as an E20 counterfactual '
                'workspace."""\n'
            ),
            "routeapp/router.py": (
                '"""Routing policy helper."""\n'
                'OPS = ("op_17", "op_23", "op_41", "op_52")\n'
                'LANES = ("lane_a", "lane_b")\n'
                "\n"
                "def select_route(operation):\n"
                '    """Return the configured lane for one opaque operation code."""\n'
                "    raise NotImplementedError\n"
            ),
            "README.md": (
                "# routeapp\n\n"
                "Routing policy helpers.\n\n"
                "`routeapp.router.select_route` is the public routing entry point.\n\n"
                "Operations are opaque repository codes and lanes are deployment targets.\n"
            ),
        },
        variants=[
            CounterfactualVariant(
                variant="A",
                dir="variant_a",
                decision_summary=(
                    "the current routing contract assigns opaque operations to "
                    "deployment lanes according to the historical allocation "
                    "table"
                ),
                critical=[
                    _f("rp.a.inter.001", 40, "The first routing plan assigned "
                       "operation codes using the opposite lane table during an "
                       "early deployment rehearsal.",
                       "decision", "routing", "opposite lane table",
                       obsolete=True),
                    _f("rp.a.sup.001", 420, "The finalized routing allocation "
                       "table for this release is: op_17 goes to lane_b; op_23 "
                       "goes to lane_a; op_41 goes to lane_a; op_52 goes to "
                       "lane_b. The operation codes are intentionally opaque "
                       "and the mapping is an external engineering contract.",
                       "decision", "routing",
                       "finalized routing allocation table"),
                    _f("rp.a.corr.001", 470, "Correction to an earlier routing "
                       "note: use the finalized allocation table, not the "
                       "rehearsal table. The current contract is op_17 -> "
                       "lane_b, op_23 -> lane_a, op_41 -> lane_a, and op_52 -> "
                       "lane_b.",
                       "correction", "routing", "current contract"),
                ],
                distractors=[
                    _f("rp.a.d1", 130, "The health checker reports deployment-"
                       "lane readiness once per minute.",
                       "implementation", "routing", "deployment-lane readiness"),
                    _f("rp.a.d2", 310, "The release dashboard groups incidents "
                       "by service owner.",
                       "implementation", "auth", "groups incidents by service owner"),
                    _f("rp.a.d3", 520, "The deployment CLI records the selected "
                       "lane in its audit event.",
                       "implementation", "database", "records the selected lane"),
                ],
                corrections=[("rp.a.inter.001", "rp.a.corr.001")],
                gold={
                    "routeapp/router.py": (
                        '"""Routing policy helper."""\n'
                        'OPS = ("op_17", "op_23", "op_41", "op_52")\n'
                        'LANES = ("lane_a", "lane_b")\n'
                        "\n"
                        "def select_route(operation):\n"
                        '    """Return the configured lane for one opaque operation code."""\n'
                        '    routes = {\n'
                        '        "op_17": "lane_b",\n'
                        '        "op_23": "lane_a",\n'
                        '        "op_41": "lane_a",\n'
                        '        "op_52": "lane_b",\n'
                        "    }\n"
                        "    return routes[operation]\n"
                    ),
                },
            ),
            CounterfactualVariant(
                variant="B",
                dir="variant_b",
                decision_summary=(
                    "the current routing contract assigns the same opaque "
                    "operation codes to the inverse deployment-lane allocation"
                ),
                critical=[
                    _f("rp.b.inter.001", 42, "The first routing plan assigned "
                       "operation codes using the opposite lane table during an "
                       "early deployment rehearsal.",
                       "decision", "routing", "opposite lane table",
                       obsolete=True),
                    _f("rp.b.sup.001", 425, "The finalized routing allocation "
                       "table for this release is: op_17 goes to lane_a; op_23 "
                       "goes to lane_b; op_41 goes to lane_b; op_52 goes to "
                       "lane_a. The operation codes are intentionally opaque "
                       "and the mapping is an external engineering contract.",
                       "decision", "routing",
                       "finalized routing allocation table"),
                    _f("rp.b.corr.001", 475, "Correction to an earlier routing "
                       "note: use the finalized allocation table, not the "
                       "rehearsal table. The current contract is op_17 -> "
                       "lane_a, op_23 -> lane_b, op_41 -> lane_b, and op_52 -> "
                       "lane_a.",
                       "correction", "routing", "current contract"),
                ],
                distractors=[
                    _f("rp.b.d1", 135, "The readiness probe confirms that the "
                       "deployment agents are alive.",
                       "implementation", "routing", "deployment agents are alive"),
                    _f("rp.b.d2", 315, "The release dashboard groups incidents "
                       "by service owner.",
                       "implementation", "auth", "groups incidents by service owner"),
                    _f("rp.b.d3", 525, "The deployment CLI writes the selected "
                       "lane into the audit record.",
                       "implementation", "database", "writes the selected lane"),
                ],
                corrections=[("rp.b.inter.001", "rp.b.corr.001")],
                gold={
                    "routeapp/router.py": (
                        '"""Routing policy helper."""\n'
                        'OPS = ("op_17", "op_23", "op_41", "op_52")\n'
                        'LANES = ("lane_a", "lane_b")\n'
                        "\n"
                        "def select_route(operation):\n"
                        '    """Return the configured lane for one opaque operation code."""\n'
                        '    routes = {\n'
                        '        "op_17": "lane_a",\n'
                        '        "op_23": "lane_b",\n'
                        '        "op_41": "lane_b",\n'
                        '        "op_52": "lane_a",\n'
                        "    }\n"
                        "    return routes[operation]\n"
                    ),
                },
            ),
        ],
    ),
    CounterfactualGroup(
        group_id="retry_policy",
        dir="retry_policy",
        prompt=(
            "Implement `submit(client, payload)` according to the repository's "
            "established contract. Keep the existing public name and signature. "
            "Do not modify hidden tests."
        ),
        workspace={
            "writeapp/__init__.py": (
                '"""writeapp - a record-submission package used as an E20 '
                'counterfactual workspace."""\n'
            ),
            "writeapp/client.py": (
                'id="qy70hb"\n'
                "class UpstreamError(RuntimeError):\n"
                "    pass\n"
                "\n"
                "\n"
                "class Client:\n"
                "    def __init__(self, failures=0):\n"
                "        self.failures = int(failures)\n"
                "        self.calls = 0\n"
                "        self.records = []\n"
                "\n"
                "    def write(self, payload):\n"
                "        self.calls += 1\n"
                "\n"
                "        if self.failures > 0:\n"
                "            self.failures -= 1\n"
                "            raise UpstreamError(\"transient failure\")\n"
                "\n"
                "        self.records.append(payload)\n"
                '        return {"status": "ok", "payload": payload}\n'
            ),
            "writeapp/writer.py": (
                'id="s50e6n"\n'
                "from writeapp.client import UpstreamError\n"
                "\n"
                "\n"
                "def submit(client, payload):\n"
                '    """Submit one payload according to repository contract."""\n'
                "    raise NotImplementedError\n"
            ),
            "README.md": (
                "# writeapp\n\n"
                "Record-submission client.\n\n"
                "`writeapp.writer.submit` is the public submission helper.\n"
            ),
        },
        variants=[
            CounterfactualVariant(
                variant="A",
                dir="variant_a",
                decision_summary=(
                    "the write is non-idempotent: submit must NOT retry a failed "
                    "attempt and must surface the error after a single call"),
                critical=[
                    _f("rp.a.inter.001", 30, "The write endpoint used to tolerate "
                       "repeated submits because earlier clients were retried by "
                       "a shared queue.",
                       "decision", "database",
                       "tolerate repeated submits", obsolete=True),
                    _f("rp.a.sup.001", 160, "The records table is append-only with "
                       "no deduplication key, so writing the same payload twice "
                       "creates two distinct rows.",
                       "decision", "database",
                       "append-only with no deduplication key"),
                    _f("rp.a.corr.001", 264, "Correction to an earlier note: the "
                       "write is not idempotent and the client must not retry "
                       "automatically. The contract is exactly one attempt per "
                       "submit and a transient failure is surfaced to the caller "
                       "unchanged.",
                       "correction", "database", "must not retry automatically"),
                ],
                distractors=[
                    _f("rp.a.d1", 118, "The metrics uploader retries up to five "
                       "times with exponential backoff because its endpoint is "
                       "idempotent.",
                       "implementation", "caching", "exponential backoff"),
                    _f("rp.a.d2", 245, "The webhook fan-out worker retries failed "
                       "deliveries on a fixed schedule.",
                       "implementation", "auth", "fixed schedule"),
                    _f("rp.a.d3", 405, "The config fetcher retries the remote "
                       "fetch once on a network timeout.",
                       "implementation", "caching", "remote fetch once"),
                ],
                corrections=[("rp.a.inter.001", "rp.a.corr.001")],
                gold={
                    "writeapp/writer.py": (
                        'id="s50e6n"\n'
                        "from writeapp.client import UpstreamError\n"
                        "\n"
                        "\n"
                        "def submit(client, payload):\n"
                        '    """Submit one payload according to repository contract."""\n'
                        "    return client.write(payload)\n"
                    ),
                },
            ),
            CounterfactualVariant(
                variant="B",
                dir="variant_b",
                decision_summary=(
                    "the write is transient/retry-safe: submit retries exactly "
                    "once after an UpstreamError"),
                critical=[
                    _f("rp.b.inter.001", 34, "The write endpoint used to record "
                       "requests under a per-request dedup key so repeated "
                       "submits of the same payload were dropped as duplicates.",
                       "decision", "database",
                       "per-request dedup key", obsolete=True),
                    _f("rp.b.sup.001", 164, "Edge gateway timeouts are the most "
                       "common failure signature and the endpoint treats "
                       "identical re-submits idempotently.",
                       "decision", "database",
                       "treats identical re-submits idempotently"),
                    _f("rp.b.corr.001", 271, "Correction to an earlier note: the "
                       "write is transient and retry-safe. The submit helper "
                       "retries exactly once after a transient failure before "
                       "surfacing the error to the caller.",
                       "correction", "database", "retries exactly once"),
                ],
                distractors=[
                    _f("rp.b.d1", 115, "The log buffer flushes entries on the "
                       "next interval after a failed upload.",
                       "implementation", "database", "flushes entries"),
                    _f("rp.b.d2", 250, "The HTTP client retries GET requests up "
                       "to three times on transport errors.",
                       "implementation", "caching", "transport errors"),
                    _f("rp.b.d3", 410, "The cache warmer refreshes entries after "
                       "every deploy, separately from the write path.",
                       "implementation", "caching", "cache warmer"),
                ],
                corrections=[("rp.b.inter.001", "rp.b.corr.001")],
                gold={
                    "writeapp/writer.py": (
                        'id="s50e6n"\n'
                        "from writeapp.client import UpstreamError\n"
                        "\n"
                        "\n"
                        "def submit(client, payload):\n"
                        '    """Submit one payload according to repository contract."""\n'
                        "    try:\n"
                        "        return client.write(payload)\n"
                        "    except UpstreamError:\n"
                        "        return client.write(payload)\n"
                    ),
                },
            ),
        ],
    ),
    CounterfactualGroup(
        group_id="serialization_policy",
        dir="serialization_policy",
        prompt=(
            "Implement `normalize(config)` according to the repository's "
            "established contract. Keep the existing public name and signature. "
            "Do not modify hidden tests."
        ),
        workspace={
            "configapp/__init__.py": (
                '"""configapp - configuration normalization helpers used as an '
                'E20 counterfactual workspace."""\n'
            ),
            "configapp/codec.py": (
                'id="2jqwd0"\n'
                '"""Configuration normalization helper."""\n'
                "def normalize(config):\n"
                '    """Normalize a parsed config according to repository contract."""\n'
                "    raise NotImplementedError\n"
            ),
            "README.md": (
                "# configapp\n\n"
                "Configuration normalization helpers.\n\n"
                "`configapp.codec.normalize` is the public normalization entry point.\n"
            ),
        },
        variants=[
            CounterfactualVariant(
                variant="A",
                dir="variant_a",
                decision_summary=(
                    "unknown/future config keys must be preserved through "
                    "normalization"),
                critical=[
                    _f("sp.a.inter.001", 31, "The config loader used to project "
                       "onto a fixed schema and silently dropped keys it did not "
                       "recognise.",
                       "decision", "database", "silently dropped keys",
                       obsolete=True),
                    _f("sp.a.sup.001", 162, "Normalized config is written back "
                       "into the next deploy artifact, so losing keys during "
                       "normalization would silently corrupt later config files.",
                       "decision", "caching",
                       "written back into the next deploy artifact"),
                    _f("sp.a.corr.001", 266, "Correction to an earlier note: the "
                       "config normalizer preserves unknown and future keys. A "
                       "config passed through normalization keeps every key it "
                       "carried and recognised keys are never dropped.",
                       "correction", "database",
                       "preserves unknown and future keys"),
                ],
                distractors=[
                    _f("sp.a.d1", 110, "The settings dashboard groups keys by "
                       "their declared environment scope.",
                       "implementation", "caching", "environment scope"),
                    _f("sp.a.d2", 300, "The config cache validates checksums "
                       "before reusing a cached file.",
                       "implementation", "caching", "validates checksums"),
                    _f("sp.a.d3", 480, "The deploy pipeline rewrites the config "
                       "file on every release with the pinned values.",
                       "implementation", "auth", "rewrites the config file"),
                ],
                corrections=[("sp.a.inter.001", "sp.a.corr.001")],
                gold={
                    "configapp/codec.py": (
                        'id="2jqwd0"\n'
                        '"""Configuration normalization helper."""\n'
                        "def normalize(config):\n"
                        '    """Normalize a parsed config according to repository contract."""\n'
                        "    return dict(config)\n"
                    ),
                },
            ),
            CounterfactualVariant(
                variant="B",
                dir="variant_b",
                decision_summary=(
                    "unknown/future config keys must be dropped during "
                    "normalization"),
                critical=[
                    _f("sp.b.inter.001", 33, "The config loader used to carry "
                       "every key it read, including keys written by future "
                       "versions of the tool.",
                       "decision", "caching", "carry every key it read",
                       obsolete=True),
                    _f("sp.b.sup.001", 166, "Validators and dashboards enumerate "
                       "an explicit allowlist of keys, so unexpected keys are "
                       "ignored by design and never rendered.",
                       "decision", "database",
                       "explicit allowlist of keys"),
                    _f("sp.b.corr.001", 270, "Correction to an earlier note: the "
                       "config normalizer drops unknown and future keys; only "
                       "recognised keys survive normalization and extra keys are "
                       "discarded.",
                       "correction", "database",
                       "drops unknown and future keys"),
                ],
                distractors=[
                    _f("sp.b.d1", 140, "The config files are laid out under "
                       "/etc/app with one file per component.",
                       "implementation", "database", "one file per component"),
                    _f("sp.b.d2", 310, "The config fetcher keeps a checksummed "
                       "cache so unchanged files are not re-read.",
                       "implementation", "caching", "checksummed cache"),
                    _f("sp.b.d3", 495, "The service layer logs a structured event "
                       "for every config load.",
                       "implementation", "database", "structured event"),
                ],
                corrections=[("sp.b.inter.001", "sp.b.corr.001")],
                gold={
                    "configapp/codec.py": (
                        'id="2jqwd0"\n'
                        '"""Configuration normalization helper."""\n'
                        "def normalize(config):\n"
                        '    """Normalize a parsed config according to repository contract."""\n'
                        '    return {"known": config["known"]} if "known" in config else {}\n'
                    ),
                },
            ),
        ],
    ),
]

_GROUP_BY_ID: Dict[str, CounterfactualGroup] = {g.group_id: g for g in _GROUPS}


# ---------------------------------------------------------------------------
# Hidden-test templates (variant-specific; never leak into history or prompts)
# ---------------------------------------------------------------------------

_HIDDEN_TESTS: Dict[Tuple[str, str], str] = {
    ("routing_policy", "A"): (
        "from routeapp.router import select_route\n"
        "\n"
        "\n"
        "def test_op_17_route():\n"
        '    assert select_route("op_17") == "lane_b"\n'
        "\n"
        "\n"
        "def test_op_23_route():\n"
        '    assert select_route("op_23") == "lane_a"\n'
        "\n"
        "\n"
        "def test_op_41_route():\n"
        '    assert select_route("op_41") == "lane_a"\n'
        "\n"
        "\n"
        "def test_op_52_route():\n"
        '    assert select_route("op_52") == "lane_b"\n'
    ),
    ("routing_policy", "B"): (
        "from routeapp.router import select_route\n"
        "\n"
        "\n"
        "def test_op_17_route():\n"
        '    assert select_route("op_17") == "lane_a"\n'
        "\n"
        "\n"
        "def test_op_23_route():\n"
        '    assert select_route("op_23") == "lane_b"\n'
        "\n"
        "\n"
        "def test_op_41_route():\n"
        '    assert select_route("op_41") == "lane_b"\n'
        "\n"
        "\n"
        "def test_op_52_route():\n"
        '    assert select_route("op_52") == "lane_a"\n'
    ),
    ("retry_policy", "A"): (
        "from writeapp.client import Client, UpstreamError\n"
        "from writeapp.writer import submit\n"
        "\n"
        "\n"
        "def test_single_attempt_no_retry():\n"
        "    client = Client(failures=1)\n"
        "    try:\n"
        "        submit(client, {\"id\": 1})\n"
        "    except UpstreamError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError(\"expected UpstreamError\")\n"
        "    assert client.calls == 1\n"
        "    assert client.records == []\n"
        "\n"
        "\n"
        "def test_success_records_payload():\n"
        "    client = Client(failures=0)\n"
        "    result = submit(client, {\"id\": 2})\n"
        '    assert result == {"status": "ok", "payload": {"id": 2}}\n'
        "    assert client.calls == 1\n"
        "    assert client.records == [{\"id\": 2}]\n"
    ),
    ("retry_policy", "B"): (
        "from writeapp.client import Client, UpstreamError\n"
        "from writeapp.writer import submit\n"
        "\n"
        "\n"
        "def test_retries_exactly_once():\n"
        "    client = Client(failures=1)\n"
        "    result = submit(client, {\"id\": 1})\n"
        '    assert result == {"status": "ok", "payload": {"id": 1}}\n'
        "    assert client.calls == 2\n"
        "    assert client.records == [{\"id\": 1}]\n"
        "\n"
        "\n"
        "def test_surfaces_error_after_two_failures():\n"
        "    client = Client(failures=2)\n"
        "    try:\n"
        "        submit(client, {\"id\": 3})\n"
        "    except UpstreamError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError(\"expected UpstreamError\")\n"
        "    assert client.calls == 2\n"
        "    assert client.records == []\n"
    ),
    ("serialization_policy", "A"): (
        "from configapp.codec import normalize\n"
        "\n"
        "\n"
        "def test_preserves_unknown_keys():\n"
        '    result = normalize({"known": "1", "future_key": "x"})\n'
        '    assert result == {"known": "1", "future_key": "x"}\n'
        "\n"
        "\n"
        "def test_known_keys_kept():\n"
        '    assert normalize({"known": "1"}) == {"known": "1"}\n'
    ),
    ("serialization_policy", "B"): (
        "from configapp.codec import normalize\n"
        "\n"
        "\n"
        "def test_drops_unknown_keys():\n"
        '    result = normalize({"known": "1", "future_key": "x"})\n'
        '    assert result == {"known": "1"}\n'
        "\n"
        "\n"
        "def test_unknown_only_is_empty():\n"
        '    assert normalize({"future_key": "x"}) == {}\n'
    ),
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def group_dir(group_id: str) -> Path:
    return _CF_DIR / _GROUP_BY_ID[group_id].dir


def variant_dir(group_id: str, variant: str) -> Path:
    vspec = _variant_spec(group_id, variant)
    return group_dir(group_id) / vspec.dir


def hidden_test_path(group_id: str, variant: str) -> Path:
    return variant_dir(group_id, variant) / "hidden" / "test_hidden.py"


def gold_path(group_id: str, variant: str) -> Path:
    return variant_dir(group_id, variant) / "gold.patch"


def history_path(group_id: str, variant: str) -> Path:
    return variant_dir(group_id, variant) / "history.txt"


def prompt_path(group_id: str) -> Path:
    return group_dir(group_id) / "prompt.txt"


def workspace_dir(group_id: str) -> Path:
    return group_dir(group_id) / "workspace"


def _variant_spec(group_id: str, variant: str) -> CounterfactualVariant:
    for vs in _GROUP_BY_ID[group_id].variants:
        if vs.variant == variant:
            return vs
    raise KeyError(f"{group_id}: unknown variant {variant!r}; "
                   f"options={list_variants(group_id)}")


def list_group_ids() -> List[str]:
    return [g.group_id for g in _GROUPS]


def list_variants(group_id: str) -> List[str]:
    return [v.variant for v in _GROUP_BY_ID[group_id].variants]


# ---------------------------------------------------------------------------
# Transcript generation (deterministic, mirrors coding_task_suite conventions)
# ---------------------------------------------------------------------------

def build_history_text(group_id: str, variant: str,
                       turns: int = HISTORY_TURNS_DEFAULT) -> List[str]:
    """Regenerate the variant transcript from the spec (for integrity checks)."""
    vspec = _variant_spec(group_id, variant)
    _, hist_text = _build_history(
        {"task_id": f"{group_id}_{variant}", "critical": vspec.critical,
         "distractors": vspec.distractors},
        seed=1, turns=turns)
    return hist_text


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha(path.read_text(encoding="utf-8"))


def history_sha(group_id: str, variant: str) -> str:
    return _file_sha(history_path(group_id, variant))


def hidden_test_sha(group_id: str, variant: str) -> str:
    return _file_sha(hidden_test_path(group_id, variant))


def gold_patch_sha(group_id: str, variant: str) -> str:
    return _file_sha(gold_path(group_id, variant))


def prompt_sha(group_id: str) -> str:
    return _file_sha(prompt_path(group_id))


def workspace_sha(group_id: str) -> str:
    """Hash of the effective workspace *as the harness delivers it*.

    Mirrors ``experiments.coding_benchmark.workspace_context`` byte-for-byte
    so this matches what ``build_coding_prompt`` embeds, without recursing
    through ``build_variant``.
    """
    root = workspace_dir(group_id)
    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in (".git", "__pycache__") for part in rel.parts):
            continue
        try:
            body = path.read_text()
        except UnicodeDecodeError:
            continue
        lines.append(f"### FILE: {rel}\n{body.rstrip()}\n")
    return _sha("\n".join(lines))


def prompt_bytes(group_id: str) -> str:
    return prompt_path(group_id).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixture materialisation
# ---------------------------------------------------------------------------

def write_fixtures(overwrite: bool = False) -> List[Path]:
    """Write workspace files, prompt, histories, hidden tests and gold patches
    from the in-code specs. Never overwrites committed fixtures unless
    ``overwrite=True``; returns the list of written paths."""
    written: List[Path] = []

    def _write(path: Path, text: str) -> None:
        if path.exists() and not overwrite:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(path)

    for group in _GROUPS:
        _write(prompt_path(group.group_id), group.prompt)
        for rel, content in group.workspace.items():
            _write(workspace_dir(group.group_id) / rel, content)
        for vs in group.variants:
            hist_text = build_history_text(group.group_id, vs.variant)
            _write(history_path(group.group_id, vs.variant),
                   "\n".join(hist_text) + "\n")
            _write(hidden_test_path(group.group_id, vs.variant),
                   _HIDDEN_TESTS[(group.group_id, vs.variant)])
            _write(gold_path(group.group_id, vs.variant),
                   _build_gold_patch(group, vs))
    return written


def _build_gold_patch(group: CounterfactualGroup,
                      vs: CounterfactualVariant) -> str:
    """Build a unified diff for the variant's gold implementation in a throwaway
    git repo (deterministic: written files carry no timestamps)."""
    import os
    import shutil
    import subprocess
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    try:
        repo = tmp / "repo"
        shutil.copytree(workspace_dir(group.group_id), repo)
        env = {"PATH": os.environ.get("PATH", "")}
        subprocess.run(["git", "-c", "user.email=e20@local",
                        "-c", "user.name=e20", "init", "-q"],
                       cwd=str(repo), check=True, capture_output=True,
                       env=env)
        subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True,
                       capture_output=True, env=env)
        subprocess.run(["git", "-c", "user.email=e20@local",
                        "-c", "user.name=e20", "commit", "-q", "-m", "base"],
                       cwd=str(repo), check=True, capture_output=True, env=env)
        for rel, body in vs.gold.items():
            target = repo / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        proc = subprocess.run(["git", "--no-pager", "diff", "--no-color",
                               "--", "."], cwd=str(repo), capture_output=True,
                              text=True, env=env)
        if proc.returncode != 0 or not proc.stdout.strip():
            raise RuntimeError(f"{group.group_id}/{vs.variant}: "
                               "gold diff generation failed")
        return proc.stdout
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# build_variant: assemble a CodingTask for the harness
# ---------------------------------------------------------------------------

def build_variant(group_id: str, variant: str, seed: int = 1,
                  history_turns: int = HISTORY_TURNS_DEFAULT) -> CodingTask:
    """Build a fully-populated E17 ``CodingTask`` for one counterfactual variant.

    The workspace and task prompt are the *shared* group fixtures (byte-identical
    across variants); only history, hidden test and gold patch are variant
    specific. The committed ``history.txt`` must match a fresh regeneration.
    """
    group = _GROUP_BY_ID[group_id]
    vspec = _variant_spec(group_id, variant)
    hpath = history_path(group_id, variant)
    if not hpath.is_file():
        raise FileNotFoundError(
            f"missing committed history for {group_id}/{variant}: {hpath}; "
            "run `python -m data.counterfactual_task_suite` to materialise "
            "fixtures first")
    committed = hpath.read_text(encoding="utf-8").splitlines()
    hist, hist_text = _build_history(
        {"task_id": f"{group_id}_{variant}", "critical": vspec.critical,
         "distractors": vspec.distractors},
        seed=1, turns=history_turns)

    # Counterfactual fixtures are oracle-pre-extracted structured histories.
    # Preserve the explicit correction relation used by the existing coding
    # workloads so the production consolidation path can target the obsolete
    # source turn directly. This enriches only structured fact metadata; the
    # committed history text remains byte-identical.
    facts_by_id = {
        fact["fact_id"]: (entry["turn_id"], fact)
        for entry in hist
        for fact in entry.get("facts", [])
    }

    for obsolete_fact_id, current_fact_id in vspec.corrections:
        if obsolete_fact_id not in facts_by_id:
            raise RuntimeError(
                f"{group_id}/{variant}: obsolete correction fact "
                f"{obsolete_fact_id!r} missing from generated history"
            )
        if current_fact_id not in facts_by_id:
            raise RuntimeError(
                f"{group_id}/{variant}: current correction fact "
                f"{current_fact_id!r} missing from generated history"
            )

        obsolete_turn, obsolete_fact = facts_by_id[obsolete_fact_id]
        current_turn, current_fact = facts_by_id[current_fact_id]

        current_fact["is_correction_target"] = True
        current_fact["is_current_correction"] = True
        current_fact["supersedes_turn"] = obsolete_turn
        current_fact["superseded_fact"] = obsolete_fact["fact"]
        current_fact["superseded_prior_fact_id"] = obsolete_fact_id
        obsolete_fact["superseded_by"] = current_turn

    if hist_text != committed:
        raise RuntimeError(
            f"{group_id}/{variant}: committed history.txt is out of sync with "
            "the in-code spec; rerun fixture materialisation to refresh")
    import sys
    return CodingTask(
        task_id=f"{group_id}_{variant}",
        title=(f"counterfactual {group_id} variant {variant} "
               f"({vspec.decision_summary})"),
        task_prompt=prompt_bytes(group_id),
        workspace_path=workspace_dir(group_id),
        hidden_test_path=hidden_test_path(group_id, variant),
        hidden_test_command=[sys.executable, "-m", "pytest",
                             "test_hidden.py", "-q"],
        history=hist,
        history_text=hist_text,
        gold_facts=_facts_from_dicts(vspec.critical),
        distractor_facts=_facts_from_dicts(vspec.distractors),
        corrections=[Correction(o, c) for o, c in vspec.corrections],
        obsolete_fact_ids=[f["fact_id"] for f in vspec.critical
                           if f.get("obsolete")]
        + [f["fact_id"] for f in vspec.distractors if f.get("obsolete")],
        distractor_topics=["auth", "database", "caching"],
        seed=seed,
        history_turns=history_turns,
        metadata={
            "ingestion": "oracle_pre_extracted",
            "ingestion_note": ("counterfactual fixtures; history generated from "
                               "the spec, hidden tests and gold patches are "
                               "variant-specific and per-group shared "
                               "workspace/prompt are byte-identical"),
            "group": group_id,
            "variant": variant,
            "decision_summary": vspec.decision_summary,
            "counterfactual": True,
            "history_sha": history_sha(group_id, variant),
            "prompt_sha": prompt_sha(group_id),
            "workspace_sha": workspace_sha(group_id),
            "visible_workspace_files": sorted(
                str(p.relative_to(workspace_dir(group_id)))
                for p in workspace_dir(group_id).rglob("*") if p.is_file()),
        },
        gold_patch_path=gold_path(group_id, variant),
    )


def no_history_prompt(group_id: str, variant: str) -> str:
    """The exact prompt the model receives under the ``no_history`` condition."""
    from experiments.coding_benchmark import build_coding_prompt, workspace_context
    task = build_variant(group_id, variant, seed=1)
    return build_coding_prompt(task.task_prompt, workspace_context(task), "")


def no_history_prompt_sha(group_id: str, variant: str) -> str:
    return _sha(no_history_prompt(group_id, variant))


def self_test() -> None:
    import sys
    assert len(_GROUPS) == 3, "expected exactly 3 counterfactual groups"
    for group_id in list_group_ids():
        assert _GROUP_BY_ID[group_id].prompt.strip(), "prompt required"
        assert workspace_dir(group_id).is_dir(), f"workspace missing: {group_id}"
        for variant in list_variants(group_id):
            vs = _variant_spec(group_id, variant)
            assert variant in ("A", "B")
            assert vs.critical and vs.corrections, \
                f"{group_id}/{variant} needs critical facts + correction"
            assert len({f["fact_id"] for f in vs.critical}) == len(vs.critical)
            task = build_variant(group_id, variant, seed=1)
            assert len(task.history) == HISTORY_TURNS_DEFAULT
            assert any((task.history_turns - f.turn_index) > 300
                       for f in task.gold_facts), \
                 f"{group_id}/{variant}: no critical fact older than 300 turns"
            assert task.gold_patch_path.is_file(), f"gold missing: {group_id}/{variant}"
            assert task.hidden_test_path.is_file(), f"hidden missing: {group_id}/{variant}"
            for f in task.gold_facts:
                assert f.probe.lower() in f.text.lower()

            # Verify counterfactual correction metadata enrichment
            by_id = {
                fact["fact_id"]: fact
                for entry in task.history
                for fact in entry.get("facts", [])
            }
            for correction in task.corrections:
                obsolete = by_id[correction.obsolete_fact_id]
                current = by_id[correction.current_fact_id]

                assert current["is_correction_target"] is True
                assert current["is_current_correction"] is True
                assert current["supersedes_turn"] == obsolete["source_turn_id"]
                assert current["superseded_fact"] == obsolete["fact"]
                assert current["superseded_prior_fact_id"] == (
                    correction.obsolete_fact_id
                )
                assert obsolete["superseded_by"] == current["source_turn_id"]
    a, b = list_variants("routing_policy")
    assert prompt_bytes("routing_policy") == prompt_bytes("routing_policy")
    assert no_history_prompt("routing_policy", a) == \
        no_history_prompt("routing_policy", b)
    print("counterfactual_task_suite self-test OK")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Materialise/verify E20 counterfactual fixtures")
    parser.add_argument("--write", action="store_true",
                        help="write missing fixture files from the specs")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing fixture files")
    args = parser.parse_args()
    if args.write or args.force:
        written = write_fixtures(overwrite=args.force)
        print(f"wrote {len(written)} fixture files")
    self_test()