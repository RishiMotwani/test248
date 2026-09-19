"""E27 counterfactual calibration repair (Phase 22 / D44).

Purpose
-------
E26 (Phase 21) could not separate adaptive memory from retrieval baselines on
the discriminative policy fixtures: the pilot was stopped at the failed gates
(history_dependence and contradiction_state both FAIL) before any full grid
was legal. E27 does NOT re-run E26 and does NOT compare adaptive to vanilla
RAG; its only job is to *calibrate* a cleaner measurement surface for a future
head-to-head benchmark. It isolates memory retrieval from coding complexity by
reducing every task to exactly ONE decisive historical policy:

* ``route_contract``  — ``choose_lane(code)`` maps four operation codes to two
  lanes (variant A: op-17->lane-b, op-23->lane-a, op-41->lane-a, op-52->lane-b;
  variant B is the exact opposite mapping).
* ``serialization_contract`` — ``normalize(config)`` either preserves unknown
  keys (variant A) or drops them (variant B).
* ``retry_contract``  — ``submit(client, payload)`` either never retries and
  propagates ``UpstreamError`` (variant A) or retries exactly once (variant B).

Calibration design (mirrors E26's counterfactual shape at a much lower
complexity): each variant is a byte-identical workspace+prompt pair whose sole
difference is the *hidden* genetic history. Every variant carries exactly one
superseded (obsolete) policy fact and one current correction, plus four
transient distractors. The base workspace FAILS the hidden test, the gold patch
PASSES, and the obsolete-policy patch FAILS — so obsolete-only historical
evidence is genuinely unsafe for the hidden test.

Retrieval geometry: the obsolete fact and the current correction are both 50-56
shared-word tokens and together exceed the 96-token pilot budget (each pair sums
> 100), so no retrieval method can inject both sides of the conflict. The four
distractors are 18-24 tokens each so either policy fact plus distractors fills
the ~96-token budget. The obsolete fact is placed at turn 80-150 (its strongest
lexical alignment with the neutral prompt), the current correction at 470-530,
and the distractors near 130/250/360/560 inside a 600-turn deterministic
transcript (``_build_history`` with ``random.Random(seed)`` — identical to the
E19/E20/E26 conventions). Current corrections are paraphrases of the revised
contract, never duplicate the obsolete wording, and never mention hidden tests,
patches, or test syntax.

The suite drives the eight offline fixture gates (A-H); ``passed`` must be True
before any LLM may run against the fixtures. Nothing here touches E26 — the
E26 suite, runner and results are byte-identical and never re-run.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from data.coding_task_suite import (
    _build_history,
    _facts_from_dicts,
    _word_tokens,
    Correction,
    CodingTask,
    HistoricalFact,
)

_CAL_DIR = Path(__file__).resolve().parent / "e27_calibration_tasks"
HISTORY_TURNS_DEFAULT = 600

PILOT_BUDGET = 96                  # predeclared; the pilot uses no other budget
GRID_SEEDS = (1, 2)
METHODS = ("no_history", "direct_history", "vanilla_rag", "adaptive")

# Token geometry (shared-word tokens, same unit as the budget and the harness)
OBSOLETE_MIN_TOKENS = 50
OBSOLETE_MAX_TOKENS = 56
CURRENT_MIN_TOKENS = 50
CURRENT_MAX_TOKENS = 56
DISTRACTOR_MIN_TOKENS = 18
DISTRACTOR_MAX_TOKENS = 24
MIN_OBS_CURRENT_SUM = 96          # obsolete + current must NOT both fit

# Placement rules (turn indices inside the 600-turn transcript)
OBSOLETE_MIN_TURN = 80
OBSOLETE_MAX_TURN = 150
CURRENT_MIN_TURN = 470
CURRENT_MAX_TURN = 530
MIN_OBSOLETE_AGE_TURNS = 300      # (600 - t) > 300


# ---------------------------------------------------------------------------
# Specs (single source of truth for the fixtures)
# ---------------------------------------------------------------------------

def _f(fid, turn, text, kind, topic, probe, obsolete=False) -> Dict:
    return dict(fact_id=fid, turn_index=turn, text=text, kind=kind, topic=topic,
                probe=probe, obsolete=obsolete)


@dataclass(frozen=True)
class CalibrationVariant:
    variant: str                          # "A" or "B"
    dir: str                              # "variant_a" / "variant_b"
    decision_summary: str                 # one-line human summary of the contract
    obsolete: List[Dict]                  # exactly one superseded policy fact
    current: List[Dict]                   # exactly one current correction
    distractors: List[Dict]               # exactly four transient facts
    corrections: List[Tuple[str, str]]    # [(obsolete_fact_id, current_fact_id)]
    gold: Dict[str, str]                  # relative path -> complete final contents
    obsolete_gold: Dict[str, str]         # implementation of the *superseded* contract


@dataclass(frozen=True)
class CalibrationGroup:
    group_id: str
    dir: str
    prompt: str
    workspace: Dict[str, str]
    forbidden_terms: List[str]            # policy-bearing surface strings
    variants: List[CalibrationVariant]


_GROUPS: List[CalibrationGroup] = [
    # =======================================================================
    # route_contract
    # =======================================================================
    CalibrationGroup(
        group_id="route_contract",
        dir="route_contract",
        prompt=(
            "Implement `choose_lane(code)` in `routeapp/router.py` according to "
            "the repository's established routing contract. Keep the existing "
            "public name and signature. Do not modify the hidden tests."
        ),
        workspace={
            "routeapp/__init__.py": (
                '"""routeapp - routing helpers used in the E27 calibration '
                'suite."""\n'
            ),
            "routeapp/router.py": (
                '"""Operation-to-lane routing helper."""\n'
                "\n"
                "def choose_lane(code):\n"
                '    """Return the destination lane for an operation code."""\n'
                "    raise NotImplementedError\n"
            ),
            "README.md": (
                "# routeapp\n\n"
                "Operation-to-lane routing helper.\n\n"
                "`routeapp.router.choose_lane` is the public routing entry "
                "point.\n"
            ),
        },
        forbidden_terms=["lane-a", "lane-b"],
        variants=[
            CalibrationVariant(
                variant="A",
                dir="variant_a",
                decision_summary=(
                    "the routing contract sends op-17 to lane-b, op-23 to "
                    "lane-a, op-41 to lane-a and op-52 to lane-b"
                ),
                obsolete=[
                    _f("route.a.old.001", 118,
                       "The routing contract for choose_lane was that op-17 "
                       "lands in lane-a, op-23 lands in lane-b, op-41 lands in "
                       "lane-b, op-52 lands in lane-a, and the operation codes "
                       "are never reassigned once the table is published. Each "
                       "release for a code is batched onto that lane by the "
                       "router before the delivery pipeline forwards it onward.",
                       "decision", "caching", "op-17 lands in lane-a",
                       obsolete=True),
                ],
                current=[
                    _f("route.a.cur.001", 508,
                       "Correction to the routing note: choose_lane now maps "
                       "op-17 to lane-b, op-23 to lane-a, op-41 to lane-a, "
                       "op-52 to lane-b, and the revised operation table is the "
                       "sole lane authority. The updated binding is read "
                       "directly by route lookups and no stale lane assignment "
                       "survives a dispatch under the current contract.",
                       "correction", "caching", "maps op-17 to lane-b"),
                ],
                distractors=[
                    _f("route.d1", 132,
                       "The parser splits incoming operation payloads into "
                       "headers and body before the routing stage inspects the "
                       "code field.",
                       "distractor", "caching", "splits incoming operation payloads"),
                    _f("route.d2", 251,
                       "A nightly soak test replays a week of operation traffic "
                       "against the staging lane cluster before the morning "
                       "release.",
                       "distractor", "caching", "soak test replays"),
                    _f("route.d3", 363,
                       "Operators page the on-call engineer when route lookup "
                       "latency exceeds the configured alert threshold for five "
                       "consecutive minutes.",
                       "distractor", "caching", "on-call engineer"),
                    _f("route.d4", 559,
                       "The audit pipeline stores every lane decision alongside "
                       "its originating trace id for later review by the "
                       "delivery team.",
                       "distractor", "caching", "audit pipeline stores every lane decision"),
                ],
                corrections=[("route.a.old.001", "route.a.cur.001")],
                gold={
                    "routeapp/router.py": (
                        '"""Operation-to-lane routing helper."""\n'
                        "\n"
                        "OP_TO_LANE = {\n"
                        '    "op-17": "lane-b",\n'
                        '    "op-23": "lane-a",\n'
                        '    "op-41": "lane-a",\n'
                        '    "op-52": "lane-b",\n'
                        "}\n"
                        "\n"
                        "\n"
                        "def choose_lane(code):\n"
                        '    """Return the destination lane for an operation '
                        'code."""\n'
                        "    return OP_TO_LANE[code]\n"
                    ),
                },
                obsolete_gold={
                    "routeapp/router.py": (
                        '"""Operation-to-lane routing helper."""\n'
                        "\n"
                        "OP_TO_LANE = {\n"
                        '    "op-17": "lane-a",\n'
                        '    "op-23": "lane-b",\n'
                        '    "op-41": "lane-b",\n'
                        '    "op-52": "lane-a",\n'
                        "}\n"
                        "\n"
                        "\n"
                        "def choose_lane(code):\n"
                        '    """Return the destination lane for an operation '
                        'code."""\n'
                        "    return OP_TO_LANE[code]\n"
                    ),
                },
            ),
            CalibrationVariant(
                variant="B",
                dir="variant_b",
                decision_summary=(
                    "the routing contract sends op-17 to lane-a, op-23 to "
                    "lane-b, op-41 to lane-b and op-52 to lane-a"
                ),
                obsolete=[
                    _f("route.b.old.001", 118,
                       "The routing contract for choose_lane was that op-17 "
                       "lands in lane-b, op-23 lands in lane-a, op-41 lands in "
                       "lane-a, op-52 lands in lane-b, and the operation codes "
                       "are never reassigned once the table is published. Each "
                       "release for a code is batched onto that lane by the "
                       "router before the delivery pipeline forwards it onward.",
                       "decision", "caching", "op-17 lands in lane-b",
                       obsolete=True),
                ],
                current=[
                    _f("route.b.cur.001", 508,
                       "Correction to the routing note: choose_lane now maps "
                       "op-17 to lane-a, op-23 to lane-b, op-41 to lane-b, "
                       "op-52 to lane-a, and the revised operation table is the "
                       "sole lane authority. The updated binding is read "
                       "directly by route lookups and no stale lane assignment "
                       "survives a dispatch under the current contract.",
                       "correction", "caching", "maps op-17 to lane-a"),
                ],
                distractors=[
                    _f("route.d1", 132,
                       "The parser splits incoming operation payloads into "
                       "headers and body before the routing stage inspects the "
                       "code field.",
                       "distractor", "caching", "splits incoming operation payloads"),
                    _f("route.d2", 251,
                       "A nightly soak test replays a week of operation traffic "
                       "against the staging lane cluster before the morning "
                       "release.",
                       "distractor", "caching", "soak test replays"),
                    _f("route.d3", 363,
                       "Operators page the on-call engineer when route lookup "
                       "latency exceeds the configured alert threshold for five "
                       "consecutive minutes.",
                       "distractor", "caching", "on-call engineer"),
                    _f("route.d4", 559,
                       "The audit pipeline stores every lane decision alongside "
                       "its originating trace id for later review by the "
                       "delivery team.",
                       "distractor", "caching", "audit pipeline stores every lane decision"),
                ],
                corrections=[("route.b.old.001", "route.b.cur.001")],
                gold={
                    "routeapp/router.py": (
                        '"""Operation-to-lane routing helper."""\n'
                        "\n"
                        "OP_TO_LANE = {\n"
                        '    "op-17": "lane-a",\n'
                        '    "op-23": "lane-b",\n'
                        '    "op-41": "lane-b",\n'
                        '    "op-52": "lane-a",\n'
                        "}\n"
                        "\n"
                        "\n"
                        "def choose_lane(code):\n"
                        '    """Return the destination lane for an operation '
                        'code."""\n'
                        "    return OP_TO_LANE[code]\n"
                    ),
                },
                obsolete_gold={
                    "routeapp/router.py": (
                        '"""Operation-to-lane routing helper."""\n'
                        "\n"
                        "OP_TO_LANE = {\n"
                        '    "op-17": "lane-b",\n'
                        '    "op-23": "lane-a",\n'
                        '    "op-41": "lane-a",\n'
                        '    "op-52": "lane-b",\n'
                        "}\n"
                        "\n"
                        "\n"
                        "def choose_lane(code):\n"
                        '    """Return the destination lane for an operation '
                        'code."""\n'
                        "    return OP_TO_LANE[code]\n"
                    ),
                },
            ),
        ],
    ),
    # =======================================================================
    # serialization_contract
    # =======================================================================
    CalibrationGroup(
        group_id="serialization_contract",
        dir="serialization_contract",
        prompt=(
            "Implement `normalize(config)` in `configapp/codec.py` according to "
            "the repository's established configuration contract. Keep the "
            "existing public name and signature. Do not modify the hidden tests."
        ),
        workspace={
            "configapp/__init__.py": (
                '"""configapp - configuration codec helpers used in the E27 '
                'calibration suite."""\n'
            ),
            "configapp/codec.py": (
                '"""Configuration normalization codec."""\n'
                "\n"
                "def normalize(config):\n"
                '    """Normalize one configuration dict."""\n'
                "    raise NotImplementedError\n"
            ),
            "README.md": (
                "# configapp\n\n"
                "Configuration normalization codec.\n\n"
                "`configapp.codec.normalize` is the public normalization entry "
                "point.\n"
            ),
        },
        forbidden_terms=["future_key", "preserved", "dropped"],
        variants=[
            CalibrationVariant(
                variant="A",
                dir="variant_a",
                decision_summary=(
                    "normalization preserves unrecognized keys exactly as "
                    "supplied"
                ),
                obsolete=[
                    _f("serial.a.old.001", 121,
                       "Under the older configuration contract normalize dropped "
                       "every key not present in the known list, so an "
                       "unrecognized future key carried by a newer caller was "
                       "silently removed from the config before the service "
                       "consumed it. The output dict therefore never contained "
                       "any entry the current schema did not declare by name at "
                       "that time.",
                       "decision", "database", "dropped every key not present in the known list",
                       obsolete=True),
                ],
                current=[
                    _f("serial.a.cur.001", 514,
                       "Correction to the config note: normalize now preserves "
                       "every key found in the input config, including "
                       "unrecognized future keys, and the normalized output "
                       "always contains exactly the keys the caller supplied. "
                       "No key is dropped during normalization because the "
                       "accepted schema deliberately tolerates forward-"
                       "compatible entries and keeps those entries unchanged "
                       "for later consumers.",
                       "correction", "database", "now preserves every key found"),
                ],
                distractors=[
                    _f("serial.d1", 134,
                       "The bootstrap loader reads the config from the mounted "
                       "volume before the running service applies its built-in "
                       "defaults.",
                       "distractor", "database", "bootstrap loader reads the config"),
                    _f("serial.d2", 253,
                       "Secret files are mounted read-only and never appear in "
                       "the serialized config output that is logged at startup.",
                       "distractor", "database", "mounted read-only"),
                    _f("serial.d3", 358,
                       "The validation pass rejects encrypted values until the "
                       "decryptor has been fully initialized by the runtime "
                       "with the correct keys.",
                       "distractor", "database", "validation pass rejects"),
                    _f("serial.d4", 561,
                       "Environment overrides are flattened into dotted keys "
                       "before the config object is assembled for each service "
                       "boot by default.",
                       "distractor", "database", "flattened into dotted keys"),
                ],
                corrections=[("serial.a.old.001", "serial.a.cur.001")],
                gold={
                    "configapp/codec.py": (
                        '"""Configuration normalization codec."""\n'
                        "\n"
                        "\n"
                        "def normalize(config):\n"
                        '    """Normalize one configuration dict, preserving '
                        'every key."""\n'
                        "    return dict(config)\n"
                    ),
                },
                obsolete_gold={
                    "configapp/codec.py": (
                        '"""Configuration normalization codec."""\n'
                        "\n"
                        "_KNOWN_KEYS = frozenset({\"known\"})\n"
                        "\n"
                        "\n"
                        "def normalize(config):\n"
                        '    """Normalize one configuration dict, keeping only '
                        'known keys."""\n'
                        "    return {key: value for key, value in "
                        "config.items() if key in _KNOWN_KEYS}\n"
                    ),
                },
            ),
            CalibrationVariant(
                variant="B",
                dir="variant_b",
                decision_summary=(
                    "normalization drops unrecognized keys from the output dict"
                ),
                obsolete=[
                    _f("serial.b.old.001", 121,
                       "Under the older configuration contract normalize "
                       "preserved every key found in the input config, "
                       "including unrecognized future keys, and the normalized "
                       "output contained exactly the keys that the caller "
                       "supplied at that time. A caller relying on a newer key "
                       "kept receiving it back intact from the codec as well.",
                       "decision", "database", "preserved every key found",
                       obsolete=True),
                ],
                current=[
                    _f("serial.b.cur.001", 514,
                       "Correction to the config note: normalize now drops "
                       "every key not present in the known list, so an "
                       "unrecognized future key carried by a newer caller is "
                       "silently removed from the config before the service "
                       "consumes it. The output dict therefore never contains "
                       "any entry the current schema does not declare by name.",
                       "correction", "database", "now drops every key not present in the known list"),
                ],
                distractors=[
                    _f("serial.d1", 134,
                       "The bootstrap loader reads the config from the mounted "
                       "volume before the running service applies its built-in "
                       "defaults.",
                       "distractor", "database", "bootstrap loader reads the config"),
                    _f("serial.d2", 253,
                       "Secret files are mounted read-only and never appear in "
                       "the serialized config output that is logged at startup.",
                       "distractor", "database", "mounted read-only"),
                    _f("serial.d3", 358,
                       "The validation pass rejects encrypted values until the "
                       "decryptor has been fully initialized by the runtime "
                       "with the correct keys.",
                       "distractor", "database", "validation pass rejects"),
                    _f("serial.d4", 561,
                       "Environment overrides are flattened into dotted keys "
                       "before the config object is assembled for each service "
                       "boot by default.",
                       "distractor", "database", "flattened into dotted keys"),
                ],
                corrections=[("serial.b.old.001", "serial.b.cur.001")],
                gold={
                    "configapp/codec.py": (
                        '"""Configuration normalization codec."""\n'
                        "\n"
                        "_KNOWN_KEYS = frozenset({\"known\"})\n"
                        "\n"
                        "\n"
                        "def normalize(config):\n"
                        '    """Normalize one configuration dict, keeping only '
                        'known keys."""\n'
                        "    return {key: value for key, value in "
                        "config.items() if key in _KNOWN_KEYS}\n"
                    ),
                },
                obsolete_gold={
                    "configapp/codec.py": (
                        '"""Configuration normalization codec."""\n'
                        "\n"
                        "\n"
                        "def normalize(config):\n"
                        '    """Normalize one configuration dict, preserving '
                        'every key."""\n'
                        "    return dict(config)\n"
                    ),
                },
            ),
        ],
    ),
    # =======================================================================
    # retry_contract
    # =======================================================================
    CalibrationGroup(
        group_id="retry_contract",
        dir="retry_contract",
        prompt=(
            "Implement `submit(client, payload)` in `writeapp/writer.py` "
            "according to the repository's established submission contract. "
            "Keep the existing public name and signature. Do not modify the "
            "hidden tests."
        ),
        workspace={
            "writeapp/__init__.py": (
                '"""writeapp - submission helpers used in the E27 calibration '
                'suite."""\n'
            ),
            "writeapp/client.py": (
                '"""Low-level write client shared by the submission contract."""\n'
                "\n"
                "\n"
                "class UpstreamError(Exception):\n"
                '    """Raised when the upstream write endpoint rejects a '
                'payload."""\n'
                "    pass\n"
                "\n"
                "\n"
                "class Client:\n"
                '    """Minimal write client; the first ``failures`` writes '
                'raise."""\n'
                "\n"
                "    def __init__(self, failures=0):\n"
                "        self.failures = failures\n"
                "        self.calls = 0\n"
                "        self.records = []\n"
                "\n"
                "    def write(self, payload):\n"
                "        self.calls += 1\n"
                "        if self.calls <= self.failures:\n"
                '            raise UpstreamError("upstream unavailable")\n'
                "        self.records.append(payload)\n"
            ),
            "writeapp/writer.py": (
                '"""Submission entry point."""\n'
                "\n"
                "\n"
                "def submit(client, payload):\n"
                '    """Submit a payload through the client according to the '
                'repository contract."""\n'
                "    raise NotImplementedError\n"
            ),
            "README.md": (
                "# writeapp\n\n"
                "Submission helper.\n\n"
                "`writeapp.writer.submit` is the public submission entry point.\n"
            ),
        },
        forbidden_terms=["retry", "retries", "exactly once"],
        variants=[
            CalibrationVariant(
                variant="A",
                dir="variant_a",
                decision_summary=(
                    "submit makes a single write call and propagates any "
                    "UpstreamError to the caller without retrying"
                ),
                obsolete=[
                    _f("retry.a.old.001", 116,
                       "The earlier submission contract had submit retry an "
                       "upstream failure exactly once: after the first "
                       "UpstreamError the client accepted a second write "
                       "attempt for the same payload, and only a second failure "
                       "was allowed to surface to the caller. The retry touched "
                       "only transient write errors and the payload was never "
                       "mutated between the attempts.",
                       "decision", "auth", "retry an upstream failure exactly once",
                       obsolete=True),
                ],
                current=[
                    _f("retry.a.cur.001", 519,
                       "Correction to the submission note: submit never retries "
                       "an upstream failure, a single UpstreamError propagates "
                       "directly to the caller, and exactly one write call is "
                       "made per payload. The client records the payload only "
                       "when the very first write succeeds. No retry attempt is "
                       "made and no record is written when the first call "
                       "fails.",
                       "correction", "auth", "never retries an upstream failure"),
                ],
                distractors=[
                    _f("retry.d1", 130,
                       "The nightly rollup writes delivery summaries into the "
                       "audit bucket and purges entries older than the "
                       "ninety-day retention window.",
                       "distractor", "auth", "nightly rollup writes delivery summaries"),
                    _f("retry.d2", 248,
                       "The writer logs a correlation id for every accepted "
                       "payload so operators can trace a record back to its "
                       "request.",
                       "distractor", "auth", "correlation id for every accepted payload"),
                    _f("retry.d3", 356,
                       "Timeouts are configured per endpoint with a ten-second "
                       "socket deadline on the shared connection pool for write "
                       "requests.",
                       "distractor", "auth", "ten-second socket deadline"),
                    _f("retry.d4", 557,
                       "A weekly report lists write latency percentiles across "
                       "the fleet for the on-call review meeting and flags "
                       "degraded regions.",
                       "distractor", "auth", "write latency percentiles"),
                ],
                corrections=[("retry.a.old.001", "retry.a.cur.001")],
                gold={
                    "writeapp/writer.py": (
                        '"""Submission entry point."""\n'
                        "\n"
                        "\n"
                        "def submit(client, payload):\n"
                        '    """Submit the payload once and propagate any '
                        'upstream error."""\n'
                        "    client.write(payload)\n"
                    ),
                },
                obsolete_gold={
                    "writeapp/writer.py": (
                        '"""Submission entry point."""\n'
                        "\n"
                        "from writeapp.client import UpstreamError\n"
                        "\n"
                        "\n"
                        "def submit(client, payload):\n"
                        '    """Submit the payload and retry exactly once after '
                        'an upstream error."""\n'
                        "    try:\n"
                        "        client.write(payload)\n"
                        "    except UpstreamError:\n"
                        "        client.write(payload)\n"
                    ),
                },
            ),
            CalibrationVariant(
                variant="B",
                dir="variant_b",
                decision_summary=(
                    "submit retries a transient upstream failure exactly once "
                    "before surfacing any second failure"
                ),
                obsolete=[
                    _f("retry.b.old.001", 116,
                       "The earlier submission contract had submit propagate "
                       "the first UpstreamError directly to the caller with "
                       "exactly one write call per payload, and the client "
                       "recorded the payload only when the very first write "
                       "succeeded. No write was ever attempted a second time "
                       "under that submission helper contract and the caller "
                       "dealt with the failure immediately.",
                       "decision", "auth", "propagate the first UpstreamError directly",
                       obsolete=True),
                ],
                current=[
                    _f("retry.b.cur.001", 519,
                       "Correction to the submission note: submit now retries "
                       "an upstream failure exactly once, the client accepts a "
                       "second write attempt for the same payload after the "
                       "first UpstreamError, and only a second failure is "
                       "allowed to surface to the caller. The retry covers only "
                       "transient write errors under the revised submission "
                       "helper contract.",
                       "correction", "auth", "now retries an upstream failure exactly once"),
                ],
                distractors=[
                    _f("retry.d1", 130,
                       "The nightly rollup writes delivery summaries into the "
                       "audit bucket and purges entries older than the "
                       "ninety-day retention window.",
                       "distractor", "auth", "nightly rollup writes delivery summaries"),
                    _f("retry.d2", 248,
                       "The writer logs a correlation id for every accepted "
                       "payload so operators can trace a record back to its "
                       "request.",
                       "distractor", "auth", "correlation id for every accepted payload"),
                    _f("retry.d3", 356,
                       "Timeouts are configured per endpoint with a ten-second "
                       "socket deadline on the shared connection pool for write "
                       "requests.",
                       "distractor", "auth", "ten-second socket deadline"),
                    _f("retry.d4", 557,
                       "A weekly report lists write latency percentiles across "
                       "the fleet for the on-call review meeting and flags "
                       "degraded regions.",
                       "distractor", "auth", "write latency percentiles"),
                ],
                corrections=[("retry.b.old.001", "retry.b.cur.001")],
                gold={
                    "writeapp/writer.py": (
                        '"""Submission entry point."""\n'
                        "\n"
                        "from writeapp.client import UpstreamError\n"
                        "\n"
                        "\n"
                        "def submit(client, payload):\n"
                        '    """Submit the payload and retry exactly once after '
                        'an upstream error."""\n'
                        "    try:\n"
                        "        client.write(payload)\n"
                        "    except UpstreamError:\n"
                        "        client.write(payload)\n"
                    ),
                },
                obsolete_gold={
                    "writeapp/writer.py": (
                        '"""Submission entry point."""\n'
                        "\n"
                        "\n"
                        "def submit(client, payload):\n"
                        '    """Submit the payload once and propagate any '
                        'upstream error."""\n'
                        "    client.write(payload)\n"
                    ),
                },
            ),
        ],
    ),
]

_GROUP_BY_ID: Dict[str, CalibrationGroup] = {g.group_id: g for g in _GROUPS}


def list_group_ids() -> List[str]:
    return [g.group_id for g in _GROUPS]


def list_variants(group_id: str) -> List[str]:
    return [v.variant for v in _GROUP_BY_ID[group_id].variants]


def _variant_spec(group_id: str, variant: str) -> CalibrationVariant:
    for vs in _GROUP_BY_ID[group_id].variants:
        if vs.variant == variant:
            return vs
    raise KeyError(f"{group_id}: unknown variant {variant!r}; "
                   f"options={list_variants(group_id)}")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def group_dir(group_id: str) -> Path:
    return _CAL_DIR / _GROUP_BY_ID[group_id].dir


def variant_dir(group_id: str, variant: str) -> Path:
    vspec = _variant_spec(group_id, variant)
    return group_dir(group_id) / vspec.dir


def hidden_test_path(group_id: str, variant: str) -> Path:
    return variant_dir(group_id, variant) / "hidden" / "test_hidden.py"


def gold_path(group_id: str, variant: str) -> Path:
    return variant_dir(group_id, variant) / "gold.patch"


def obsolete_gold_path(group_id: str, variant: str) -> Path:
    return variant_dir(group_id, variant) / "obsolete.patch"


def history_path(group_id: str, variant: str) -> Path:
    return variant_dir(group_id, variant) / "history.txt"


def metadata_path(group_id: str, variant: str) -> Path:
    return variant_dir(group_id, variant) / "metadata.json"


def prompt_path(group_id: str) -> Path:
    return group_dir(group_id) / "prompt.txt"


def workspace_dir(group_id: str) -> Path:
    return group_dir(group_id) / "workspace"


# ---------------------------------------------------------------------------
# Transcript generation (deterministic; mirrors coding_task_suite conventions)
# ---------------------------------------------------------------------------

def build_history_text(group_id: str, variant: str,
                       turns: int = HISTORY_TURNS_DEFAULT) -> List[str]:
    """Regenerate the variant transcript from the spec (for integrity checks)."""
    vspec = _variant_spec(group_id, variant)
    _, hist_text = _build_history(
        {"task_id": f"{group_id}_{variant}",
         "critical": vspec.obsolete + vspec.current,
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


def obsolete_gold_patch_sha(group_id: str, variant: str) -> str:
    return _file_sha(obsolete_gold_path(group_id, variant))


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
# Structured fact types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyFact:
    fact_id: str
    turn_index: int
    text: str
    kind: str
    topic: str
    probe: str
    old: bool


@dataclass(frozen=True)
class CalibrationCorrection:
    obsolete_fact_id: str
    current_fact_id: str


@dataclass(frozen=True)
class CalibrationTask:
    task_id: str
    title: str
    task_prompt: str
    workspace_path: Path
    hidden_test_path: Path
    hidden_test_command: List[str]
    history: List[Dict]          # [{"turn_id", "user", "tokens", "facts": [...]}]
    history_text: List[str]
    group_id: str
    variant: str
    current_facts: List[PolicyFact]
    obsolete_policy_facts: List[PolicyFact]
    distractor_facts: List[HistoricalFact]
    corrections: List[CalibrationCorrection]
    gold_patch_path: Path
    obsolete_gold_patch_path: Path
    seed: int
    history_turns: int
    metadata: Dict = field(default_factory=dict)

    @property
    def all_policy_facts(self) -> List[PolicyFact]:
        return list(self.obsolete_policy_facts) + list(self.current_facts)


def _policy_from_dict(raw: Dict) -> PolicyFact:
    return PolicyFact(
        fact_id=raw["fact_id"], turn_index=raw["turn_index"], text=raw["text"],
        kind=raw["kind"], topic=raw["topic"], probe=raw["probe"],
        old=bool(raw.get("obsolete", False)),
    )


# ---------------------------------------------------------------------------
# Fixture materialisation
# ---------------------------------------------------------------------------

def _build_patch(group: CalibrationGroup, workspace: Path,
                 files: Dict[str, str]) -> str:
    """Build a unified diff for the given final files in a throwaway git repo
    (deterministic: written files carry no timestamps)."""
    import os
    import shutil
    import subprocess
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    try:
        repo = tmp / "repo"
        shutil.copytree(workspace, repo)
        env = {"PATH": os.environ.get("PATH", "")}
        subprocess.run(["git", "-c", "user.email=e27@local",
                        "-c", "user.name=e27", "init", "-q"],
                       cwd=str(repo), check=True, capture_output=True, env=env)
        subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True,
                       capture_output=True, env=env)
        subprocess.run(["git", "-c", "user.email=e27@local",
                        "-c", "user.name=e27", "commit", "-q", "-m", "base"],
                       cwd=str(repo), check=True, capture_output=True, env=env)
        for rel, body in files.items():
            target = repo / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        proc = subprocess.run(["git", "--no-pager", "diff", "--no-color", "--", "."],
                              cwd=str(repo), capture_output=True, text=True, env=env)
        if proc.returncode != 0 or not proc.stdout.strip():
            raise RuntimeError(f"{group.group_id}: gold diff generation failed")
        return proc.stdout
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _variant_metadata(group: CalibrationGroup,
                      vspec: CalibrationVariant) -> Dict:
    fact_rows = []
    for raw in vspec.obsolete + vspec.current + vspec.distractors:
        fact_rows.append({
            "fact_id": raw["fact_id"], "turn_index": raw["turn_index"],
            "kind": raw["kind"], "topic": raw["topic"], "old": raw["obsolete"],
        })
    return {
        "group_id": group.group_id,
        "variant": vspec.variant,
        "decision_summary": vspec.decision_summary,
        "facts": fact_rows,
        "corrections": [{"obsolete": o, "current": c}
                        for o, c in vspec.corrections],
        "current_fact_ids": [f["fact_id"] for f in vspec.current],
        "obsolete_fact_ids": [f["fact_id"] for f in vspec.obsolete],
        "distractor_fact_ids": [f["fact_id"] for f in vspec.distractors],
    }


def write_fixtures(overwrite: bool = False) -> List[Path]:
    """Write workspace files, prompt, histories, hidden tests and gold patches
    from the in-code specs. Never overwrites committed fixtures unless
    ``overwrite=True``; returns the list of written paths."""
    import json
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
                   _build_patch(group, workspace_dir(group.group_id), vs.gold))
            _write(obsolete_gold_path(group.group_id, vs.variant),
                   _build_patch(group, workspace_dir(group.group_id),
                                vs.obsolete_gold))
            _write(metadata_path(group.group_id, vs.variant),
                   json.dumps(_variant_metadata(group, vs), indent=2) + "\n")
    return written


# ---------------------------------------------------------------------------
# Hidden-test templates (variant-specific; never leak into history or prompts)
# ---------------------------------------------------------------------------

_HIDDEN_TESTS: Dict[Tuple[str, str], str] = {
    ("route_contract", "A"): (
        "from routeapp.router import choose_lane\n"
        "\n"
        "\n"
        "def test_op_17():\n"
        '    assert choose_lane("op-17") == "lane-b"\n'
        "\n"
        "\n"
        "def test_op_23():\n"
        '    assert choose_lane("op-23") == "lane-a"\n'
        "\n"
        "\n"
        "def test_op_41():\n"
        '    assert choose_lane("op-41") == "lane-a"\n'
        "\n"
        "\n"
        "def test_op_52():\n"
        '    assert choose_lane("op-52") == "lane-b"\n'
    ),
    ("route_contract", "B"): (
        "from routeapp.router import choose_lane\n"
        "\n"
        "\n"
        "def test_op_17():\n"
        '    assert choose_lane("op-17") == "lane-a"\n'
        "\n"
        "\n"
        "def test_op_23():\n"
        '    assert choose_lane("op-23") == "lane-b"\n'
        "\n"
        "\n"
        "def test_op_41():\n"
        '    assert choose_lane("op-41") == "lane-b"\n'
        "\n"
        "\n"
        "def test_op_52():\n"
        '    assert choose_lane("op-52") == "lane-a"\n'
    ),
    ("serialization_contract", "A"): (
        "from configapp.codec import normalize\n"
        "\n"
        "\n"
        "def test_unknown_keys_preserved():\n"
        '    assert normalize({"known": "1", "future_key": "x"}) == {\n'
        '        "known": "1", "future_key": "x"}\n'
    ),
    ("serialization_contract", "B"): (
        "from configapp.codec import normalize\n"
        "\n"
        "\n"
        "def test_unknown_keys_dropped():\n"
        '    assert normalize({"known": "1", "future_key": "x"}) == '
        '{"known": "1"}\n'
    ),
    ("retry_contract", "A"): (
        "import pytest\n"
        "\n"
        "from writeapp.client import Client, UpstreamError\n"
        "from writeapp.writer import submit\n"
        "\n"
        "\n"
        "def test_no_retry_propagates_and_calls_once():\n"
        "    client = Client(failures=1)\n"
        "    with pytest.raises(UpstreamError):\n"
        '        submit(client, {"id": 1})\n'
        "    assert client.calls == 1\n"
        "    assert client.records == []\n"
    ),
    ("retry_contract", "B"): (
        "from writeapp.client import Client\n"
        "from writeapp.writer import submit\n"
        "\n"
        "\n"
        "def test_retry_once_records_one_payload():\n"
        "    client = Client(failures=1)\n"
        '    submit(client, {"id": 1})\n'
        "    assert client.calls == 2\n"
        '    assert client.records == [{"id": 1}]\n'
    ),
}


# ---------------------------------------------------------------------------
# build_variant: assemble a CalibrationTask for the benchmark
# ---------------------------------------------------------------------------

def build_variant(group_id: str, variant: str, seed: int = 1,
                  history_turns: int = HISTORY_TURNS_DEFAULT) -> CalibrationTask:
    """Build a fully-populated ``CalibrationTask`` for one variant.

    The workspace and task prompt are the *shared* group fixtures
    (byte-identical across variants); only history, hidden test and
    gold/obsolete patches are variant specific. The committed ``history.txt``
    must match a fresh regeneration for ``seed=1``.
    """
    group = _GROUP_BY_ID[group_id]
    vspec = _variant_spec(group_id, variant)
    hpath = history_path(group_id, variant)
    if not hpath.is_file():
        raise FileNotFoundError(
            f"missing committed history for {group_id}/{variant}: {hpath}; "
            "run `python -m data.e27_calibration_suite --write` to "
            "materialise fixtures first")
    committed = hpath.read_text(encoding="utf-8").splitlines()
    hist, hist_text = _build_history(
        {"task_id": f"{group_id}_{variant}",
         "critical": vspec.obsolete + vspec.current,
         "distractors": vspec.distractors},
        seed=seed, turns=history_turns)

    # Propagate the explicit correction metadata into the structured history
    # so the production consolidation path can target the superseded turn
    # directly (oracle-pre-extracted, mirroring E20's repair and E26).
    facts_by_id = {
        fact["fact_id"]: (entry["turn_id"], fact)
        for entry in hist
        for fact in entry.get("facts", [])
    }
    for obsolete_fact_id, current_fact_id in vspec.corrections:
        if obsolete_fact_id not in facts_by_id:
            raise RuntimeError(
                f"{group_id}/{variant}: obsolete correction fact "
                f"{obsolete_fact_id!r} missing from generated history")
        if current_fact_id not in facts_by_id:
            raise RuntimeError(
                f"{group_id}/{variant}: current correction fact "
                f"{current_fact_id!r} missing from generated history")
        obsolete_turn, obsolete_fact = facts_by_id[obsolete_fact_id]
        current_turn, current_fact = facts_by_id[current_fact_id]
        current_fact["is_correction_target"] = True
        current_fact["is_current_correction"] = True
        current_fact["supersedes_turn"] = obsolete_turn
        current_fact["superseded_fact"] = obsolete_fact["fact"]
        current_fact["superseded_prior_fact_id"] = obsolete_fact_id
        obsolete_fact["superseded_by"] = current_fact_id

    if seed == 1 and hist_text != committed:
        raise RuntimeError(
            f"{group_id}/{variant}: committed history.txt is out of sync with "
            "the in-code spec; rerun fixture materialisation to refresh")

    return CalibrationTask(
        task_id=f"{group_id}_{variant}",
        title=(f"calibration {group_id} variant {variant} "
               f"({vspec.decision_summary})"),
        task_prompt=prompt_bytes(group_id),
        workspace_path=workspace_dir(group_id),
        hidden_test_path=hidden_test_path(group_id, variant),
        hidden_test_command=[sys.executable, "-m", "pytest",
                             "test_hidden.py", "-q"],
        history=hist,
        history_text=hist_text,
        group_id=group_id,
        variant=variant,
        current_facts=[_policy_from_dict(r) for r in vspec.current],
        obsolete_policy_facts=[_policy_from_dict(r) for r in vspec.obsolete],
        distractor_facts=_facts_from_dicts(vspec.distractors),
        corrections=[CalibrationCorrection(o, c) for o, c in vspec.corrections],
        gold_patch_path=gold_path(group_id, variant),
        obsolete_gold_patch_path=obsolete_gold_path(group_id, variant),
        seed=seed,
        history_turns=history_turns,
        metadata={
            "ingestion": "oracle_pre_extracted",
            "ingestion_note": (
                "calibration fixtures; one obsolete policy fact, one current "
                "correction, four distractors; hidden tests and gold/obsolete "
                "patches are variant-specific and the per-group shared "
                "workspace/prompt are byte-identical"),
            "group": group_id,
            "variant": variant,
            "decision_summary": vspec.decision_summary,
            "single_policy_calibration": True,
            "history_sha": history_sha(group_id, variant),
            "prompt_sha": prompt_sha(group_id),
            "workspace_sha": workspace_sha(group_id),
            "visible_workspace_files": sorted(
                str(p.relative_to(workspace_dir(group_id)))
                for p in workspace_dir(group_id).rglob("*") if p.is_file()),
        },
    )


def as_coding_task(task: CalibrationTask) -> CodingTask:
    """Convert a ``CalibrationTask`` into the harness ``CodingTask`` type."""
    obsolete_ids = [f.fact_id for f in task.obsolete_policy_facts]
    return CodingTask(
        task_id=task.task_id,
        title=task.title,
        task_prompt=task.task_prompt,
        workspace_path=task.workspace_path,
        hidden_test_path=task.hidden_test_path,
        hidden_test_command=task.hidden_test_command,
        history=task.history,
        history_text=task.history_text,
        gold_facts=(
            [HistoricalFact(fact_id=f.fact_id, turn_index=f.turn_index,
                            text=f.text, kind=f.kind, topic=f.topic,
                            probe=f.probe, obsolete=True)
             for f in task.obsolete_policy_facts]
            + [HistoricalFact(fact_id=f.fact_id, turn_index=f.turn_index,
                              text=f.text, kind=f.kind, topic=f.topic,
                              probe=f.probe, obsolete=False)
               for f in task.current_facts]
        ),
        distractor_facts=task.distractor_facts,
        corrections=[Correction(c.obsolete_fact_id, c.current_fact_id)
                     for c in task.corrections],
        obsolete_fact_ids=obsolete_ids,
        distractor_topics=["auth", "database", "caching"],
        seed=task.seed,
        history_turns=task.history_turns,
        metadata=dict(task.metadata),
        gold_patch_path=task.gold_patch_path,
    )


def no_history_prompt(group_id: str, variant: str) -> str:
    """The exact prompt the model receives under the ``no_history`` condition."""
    from experiments.coding_benchmark import build_coding_prompt, workspace_context
    task = as_coding_task(build_variant(group_id, variant, seed=1))
    return build_coding_prompt(task.task_prompt, workspace_context(task), "")


def no_history_prompt_sha(group_id: str, variant: str) -> str:
    return _sha(no_history_prompt(group_id, variant))


# ---------------------------------------------------------------------------
# Offline validation (fixture gates; no LLM, no embeddings)
# ---------------------------------------------------------------------------

# Decision-bearing surface strings that must never appear in the visible
# workspace or the task prompt (the policy is history-only).
_FORBIDDEN_GLOBAL_TERMS = {"hidden tests", "hidden test", "pytest",
                           "```", "assert"}


def _workspace_prompt_text(group_id: str) -> str:
    parts = [_GROUP_BY_ID[group_id].prompt]
    root = workspace_dir(group_id)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in (".git", "__pycache__") for part in path.relative_to(root).parts):
            continue
        try:
            parts.append(path.read_text())
        except UnicodeDecodeError:
            continue
    return "\n".join(parts)


def fact_token_counts(task: CalibrationTask) -> Dict[str, int]:
    counts = {}
    for f in task.all_policy_facts:
        counts[f.fact_id] = _word_tokens(f.text)
    for f in task.distractor_facts:
        counts[f.fact_id] = _word_tokens(f.text)
    return counts


def policy_token_counts(task: CalibrationTask) -> Tuple[int, int]:
    """(obsolete_tokens, current_tokens) for the geometry gate."""
    obs = _word_tokens(task.obsolete_policy_facts[0].text)
    cur = _word_tokens(task.current_facts[0].text)
    return obs, cur


def _check_history_text_hygiene(task: CalibrationTask) -> List[str]:
    """Return decision-bearing or hidden-test-like strings found in history."""
    joined = "\n".join(task.history_text)
    hits = sorted(t for t in _FORBIDDEN_GLOBAL_TERMS if t in joined.lower())
    return hits


def validate_group_variant(task: CalibrationTask,
                           other_variant_task: CalibrationTask = None) -> Dict:
    """Run every offline fixture gate for one variant.

    ``other_variant_task`` is the sibling A/B variant used for identity and
    difference checks; when omitted it is built automatically (seed=1).

    Returns a dict with per-check booleans plus the aggregate ``passed``.
    ``passed`` must be True before any LLM runs against the fixture.
    """
    checks: List[Dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    gid = task.group_id
    grp = _GROUP_BY_ID[gid]

    if other_variant_task is None:
        other = "B" if task.variant == "A" else "A"
        other_variant_task = build_variant(gid, other, seed=1)

    # --- A: base workspace fails the hidden test (NotImplementedError) ---
    # --- B: gold patch passes; C: obsolete patch fails ---
    base_fail, gold_pass, obsolete_fail = _run_offline_hidden_checks(task)
    check("base_workspace_fails_hidden", base_fail)
    check("gold_patch_passes_hidden", gold_pass)
    check("obsolete_patch_fails_hidden", obsolete_fail)

    # --- D/E: workspace and prompt byte-identical across variants ---
    check("workspace_identical_across_variants",
          task.metadata["workspace_sha"]
          == other_variant_task.metadata["workspace_sha"])
    check("prompt_identical_across_variants",
          task.metadata["prompt_sha"] == other_variant_task.metadata["prompt_sha"])

    # --- F: history differs across variants ---
    check("history_differs_across_variants",
          task.metadata["history_sha"]
          != other_variant_task.metadata["history_sha"])

    # --- G: fact token geometry + placement + depth ---
    obs_tokens, cur_tokens = policy_token_counts(task)
    dist_tokens = [_word_tokens(f.text) for f in task.distractor_facts]
    geo_ok = (
        OBSOLETE_MIN_TOKENS <= obs_tokens <= OBSOLETE_MAX_TOKENS
        and CURRENT_MIN_TOKENS <= cur_tokens <= CURRENT_MAX_TOKENS
        and all(DISTRACTOR_MIN_TOKENS <= t <= DISTRACTOR_MAX_TOKENS
                for t in dist_tokens)
        and (obs_tokens + cur_tokens) > MIN_OBS_CURRENT_SUM
    )
    turns = task.history_turns
    obs_turn = task.obsolete_policy_facts[0].turn_index
    cur_turn = task.current_facts[0].turn_index
    dist_turns = sorted(f.turn_index for f in task.distractor_facts)
    windows = [(120, 140), (240, 260), (350, 370), (550, 570)]
    place_ok = (
        OBSOLETE_MIN_TURN <= obs_turn <= OBSOLETE_MAX_TURN
        and (turns - obs_turn) > MIN_OBSOLETE_AGE_TURNS
        and CURRENT_MIN_TURN <= cur_turn <= CURRENT_MAX_TURN
        and len(task.history) == HISTORY_TURNS_DEFAULT
        and all(any(lo <= t <= hi for lo, hi in windows) for t in dist_turns)
    )
    counts_ok = (
        len(task.obsolete_policy_facts) == 1
        and len(task.current_facts) == 1
        and len(task.distractor_facts) == 4
        and len(task.corrections) == 1
    )
    check("fact_token_geometry",
          geo_ok,
          f"obs={obs_tokens} cur={cur_tokens} dist={dist_tokens} "
          f"sum={obs_tokens + cur_tokens} (>96: {obs_tokens + cur_tokens > 96})")
    check("fact_placement_and_history_depth",
          place_ok,
          f"obs_turn={obs_turn} cur_turn={cur_turn} dist_turns={dist_turns}")
    check("single_policy_counts", counts_ok)

    # --- H: structured correction metadata reaches history[*]["facts"] ---
    by_id = {fact["fact_id"]: fact
             for entry in task.history
             for fact in entry.get("facts", [])}
    cur = by_id.get(task.corrections[0].current_fact_id)
    obs = by_id.get(task.corrections[0].obsolete_fact_id)
    meta_ok = bool(
        cur and obs
        and cur.get("category") == "project_context"
        and cur.get("is_correction_target") is True
        and cur.get("is_current_correction") is True
        and cur.get("supersedes_turn") == obs.get("source_turn_id")
        and cur.get("superseded_fact") == obs.get("fact")
        and cur.get("superseded_prior_fact_id") == task.corrections[0].obsolete_fact_id
        and obs.get("superseded_by") == task.corrections[0].current_fact_id
    )
    check("correction_metadata_structured", meta_ok)

    # --- retrieval-shaping hygiene ---
    visible = _workspace_prompt_text(gid)
    group_hits = sorted(t for t in grp.forbidden_terms if t in visible)
    check("no_policy_mapping_in_visible_workspace", not group_hits,
          f"hits={group_hits}")
    hist_hits = _check_history_text_hygiene(task)
    check("no_hidden_test_language_in_history", not hist_hits,
          f"hits={hist_hits}")
    check("probe_terms_in_fact_text", all(
        f.probe.lower() in f.text.lower() for f in task.all_policy_facts))

    # correction must be a paraphrase, not a copy of the obsolete wording
    copy = (task.current_facts[0].text.lower()
            == task.obsolete_policy_facts[0].text.lower())
    check("current_not_duplicate_of_obsolete", not copy)

    return {
        "group_id": gid,
        "variant": task.variant,
        "checks": checks,
        "failing": [c["check"] for c in checks if not c["passed"]],
        "passed": all(c["passed"] for c in checks),
    }


def _run_offline_hidden_checks(task: CalibrationTask) -> Tuple[bool, bool, bool]:
    """Deterministic base/gold/obsolete hidden-test outcomes.

    Applies the empty patch (base), the gold patch, and the obsolete patch in
    three throwaway git repos and runs each hidden test. No LLM/embedding call.
    """
    from experiments import coding_benchmark as cb

    def run_one(patch: Optional[str]) -> bool:
        import shutil
        import subprocess
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        try:
            repo = cb.prepare_workspace(task, tmp)
            if patch:
                ok, _ = cb.apply_patch(repo, patch)
                if not ok:
                    return False
            test_dest = repo / "test_hidden.py"
            test_dest.write_text(task.hidden_test_path.read_text(encoding="utf-8"))
            proc = subprocess.run(task.hidden_test_command, cwd=str(repo),
                                  capture_output=True, text=True, timeout=300)
            return proc.returncode == 0
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    base = run_one(None)
    gold = run_one(task.gold_patch_path.read_text(encoding="utf-8"))
    obsolete = run_one(task.obsolete_gold_patch_path.read_text(encoding="utf-8"))
    return (not base), gold, (not obsolete)


def validate_all_fixtures() -> Dict:
    """Validate all 6 variants against their sibling. Returns a scope dict."""
    out = {}
    all_ok = True
    for gid in list_group_ids():
        for variant in list_variants(gid):
            task = build_variant(gid, variant, seed=1)
            res = validate_group_variant(task)
            out[f"{gid}:{variant}"] = res
            all_ok = all_ok and res["passed"]
    return {"variants": out, "passed": all_ok}


# ---------------------------------------------------------------------------
# self-test (python -m data.e27_calibration_suite)
# ---------------------------------------------------------------------------

def self_test(require_fixtures: bool = False) -> None:
    assert len(_GROUPS) == 3, "expected exactly 3 calibration groups"
    for gid in list_group_ids():
        assert _GROUP_BY_ID[gid].prompt.strip(), "prompt required"
        assert len(_GROUP_BY_ID[gid].forbidden_terms) > 0, "forbidden terms"
        for variant in list_variants(gid):
            vs = _variant_spec(gid, variant)
            assert variant in ("A", "B")
            assert len(vs.obsolete) == 1, f"{gid}/{variant} needs 1 obsolete fact"
            assert len(vs.current) == 1, f"{gid}/{variant} needs 1 current fact"
            assert len(vs.distractors) == 4, f"{gid}/{variant} needs 4 distractors"
            assert len(vs.corrections) == 1, f"{gid}/{variant} needs 1 correction"
            task = build_variant(gid, variant, seed=1)
            assert len(task.history) == HISTORY_TURNS_DEFAULT
            assert task.obsolete_policy_facts[0].turn_index in range(
                OBSOLETE_MIN_TURN, OBSOLETE_MAX_TURN + 1)
            assert len(task.current_facts) == 1
            fresh = build_history_text(gid, variant)
            if require_fixtures:
                assert "\n".join(fresh) + "\n" == lineage_read(gid, variant), \
                    f"{gid}/{variant} history drifted"
                assert task.gold_patch_path.is_file()
                assert task.obsolete_gold_patch_path.is_file()
                assert task.hidden_test_path.is_file()
            obs_t, cur_t = policy_token_counts(task)
            assert OBSOLETE_MIN_TOKENS <= obs_t <= OBSOLETE_MAX_TOKENS
            assert CURRENT_MIN_TOKENS <= cur_t <= CURRENT_MAX_TOKENS
            assert obs_t + cur_t > MIN_OBS_CURRENT_SUM
            for f in task.all_policy_facts + task.distractor_facts:
                assert f.probe.lower() in f.text.lower(), \
                    f"{gid}/{variant}:{f.fact_id} probe not in text"
                if getattr(f, "old", False):
                    assert OBSOLETE_MIN_TOKENS <= _word_tokens(f.text) <= OBSOLETE_MAX_TOKENS
                elif getattr(f, "obsolete", False):
                    assert OBSOLETE_MIN_TOKENS <= _word_tokens(f.text) <= OBSOLETE_MAX_TOKENS
                elif f.kind == "correction":
                    assert CURRENT_MIN_TOKENS <= _word_tokens(f.text) <= CURRENT_MAX_TOKENS
                else:
                    assert DISTRACTOR_MIN_TOKENS <= _word_tokens(f.text) <= DISTRACTOR_MAX_TOKENS
    # byte-identical workspace/prompt across variants
    for gid in list_group_ids():
        a = build_variant(gid, "A", seed=1)
        b = build_variant(gid, "B", seed=1)
        assert a.metadata["workspace_sha"] == b.metadata["workspace_sha"]
        assert a.metadata["prompt_sha"] == b.metadata["prompt_sha"]
        assert no_history_prompt(gid, "A") == no_history_prompt(gid, "B")
        assert a.metadata["history_sha"] != b.metadata["history_sha"]
    print("e27_calibration_suite self-test OK")


def lineage_read(gid: str, variant: str) -> str:
    return history_path(gid, variant).read_text(encoding="utf-8")


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        description="Materialise/verify E27 calibration fixtures")
    parser.add_argument("--write", action="store_true",
                        help="write missing fixture files from the specs")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing fixture files")
    args = parser.parse_args()
    if args.write or args.force:
        written = write_fixtures(overwrite=args.force)
        print(f"wrote {len(written)} fixture files")
    require = (args.write or args.force)
    self_test(require_fixtures=require)