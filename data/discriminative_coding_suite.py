"""E26 discriminative adaptive-vs-RAG coding suite (Phase 21 / D43).

Purpose
-------
E25 (Phase 20) established that the E19 benchmark could not separate adaptive
memory from retrieval baselines: the 256/512/1024 budget range never bound
either recall method, and vanilla_rag tolerated obsolete-only correction
evidence >= 0.80 of the time (succeeding 26/27 primary cells at the ceiling).
E26 therefore builds a *discriminative* suite whose workload makes obsolete-only
historical evidence materially unsafe for the hidden test and whose budget range
(64/96/128) actually binds both recall methods at the low end.

Three groups, each with two byte-identical workspaces+prompts (variants A/B),
exactly like E20's counterfactual suite:

* ``release_adapter``  — whether the adapter journals raw release payloads to a
  local ``pending_releases.json`` file before export (A: no journal / direct
  publish; B: journal before export)
* ``invoice_adapter``  — whether unknown line-item codes survive normalization
  verbatim (A: preserved) or are remapped to ``MISCELLANEOUS`` (B: remapped)
* ``message_adapter``  — whether a message counts as delivered only after the
  consumer confirms it (A: confirmation required) or as soon as the relay
  accepts it (B: acceptance is delivery)

Every variant exists for its own hidden-test success rule: the base workspace
FAILS the hidden test, the gold patch PASSES, and the *obsolete*-policy patch
FAILS (an implementation following the superseded historical contract is
trapped by the hidden test). This makes obsolete-only historical evidence
unsafe, which is E25's requirement (a).

Each variant has 12 structured facts: 4 superseded (obsolete) policy facts at
turns ~70-180 (age >300 turns at the 600-turn end, matching the E19/E20
long-range convention), 4 current policy facts at turns ~420-540 that explicitly
supersede the obsolete ones, and 4 transient distractors. Every fact is 18-28
shared-word tokens. Current facts carry the full correction metadata
(``is_correction_target``, ``is_current_correction``, ``supersedes_turn``,
``superseded_fact``, ``superseded_prior_fact_id``); obsolete facts carry
``superseded_by``. This is exactly the structured metadata the production
consolidation path consumes, so the adaptive arm resolves each correction by
explicit supersession while vanilla_rag must rank both sides of the conflict.
"""

from __future__ import annotations

import hashlib
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

_DISC_DIR = Path(__file__).resolve().parent / "discriminative_coding_tasks"
HISTORY_TURNS_DEFAULT = 600

FULL_BUDGETS = (64, 96, 128)          # predeclared (E25-constrained), never reselected
PILOT_BUDGET = 96                     # the middle budget for the 48-cell pilot
GRID_SEEDS = (1, 2, 3)
PILOT_SEEDS = (1, 2)
METHODS = ("vanilla_rag", "adaptive")
PILOT_METHODS = ("no_history", "direct_history", "vanilla_rag", "adaptive")

MIN_FACT_TOKENS = 18
MAX_FACT_TOKENS = 28


# ---------------------------------------------------------------------------
# Specs (single source of truth for the fixtures)
# ---------------------------------------------------------------------------

def _f(fid, turn, text, kind, topic, probe, obsolete=False) -> Dict:
    return dict(fact_id=fid, turn_index=turn, text=text, kind=kind, topic=topic,
                probe=probe, obsolete=obsolete)


@dataclass(frozen=True)
class DiscriminativeVariant:
    variant: str                # "A" or "B"
    dir: str                    # "variant_a" / "variant_b"
    decision_summary: str       # one-line human summary of the current contract
    obsolete: List[Dict]        # superseded policy facts (kind decision, obsolete=True)
    current: List[Dict]         # superseding policy facts (kind correction)
    distractors: List[Dict]     # transient facts
    corrections: List[Tuple[str, str]]   # (obsolete_fact_id, current_fact_id)
    gold: Dict[str, str]        # relative path -> complete final file contents
    obsolete_gold: Dict[str, str]  # implementation following the *obsolete* contract


@dataclass(frozen=True)
class DiscriminativeGroup:
    group_id: str
    dir: str
    prompt: str
    workspace: Dict[str, str]
    variants: List[DiscriminativeVariant]


_GROUPS: List[DiscriminativeGroup] = [
    # =======================================================================
    # release_adapter
    # =======================================================================
    DiscriminativeGroup(
        group_id="release_adapter",
        dir="release_adapter",
        prompt=(
            "Implement `publish_release(store, exporter, payload)` in "
            "`releaseapp/adapter.py` according to the repository's established "
            "contract. Keep the existing public name and signature. Do not "
            "modify the hidden tests."
        ),
        workspace={
            "releaseapp/__init__.py": (
                '"""releaseapp - release publishing adapter used in the E26 '
                'discriminative suite."""\n'
            ),
            "releaseapp/adapter.py": (
                '"""Release publishing adapter."""\n'
                "\n"
                "def publish_release(store, exporter, payload):\n"
                '    """Publish one release payload according to repository '
                'contract."""\n'
                "    raise NotImplementedError\n"
            ),
            "README.md": (
                "# releaseapp\n\n"
                "Release publishing adapter.\n\n"
                "`releaseapp.adapter.publish_release` is the public publishing "
                "entry point.\n"
            ),
        },
        variants=[
            DiscriminativeVariant(
                variant="A",
                dir="variant_a",
                decision_summary=(
                    "the adapter publishes directly through the exporter and "
                    "records only id+timestamp audit rows, never journaling raw "
                    "payloads to a local file"
                ),
                obsolete=[
                    _f("ra.a.old.001", 74,
                       "The release adapter journals every published payload "
                       "verbatim into the local pending_releases.json file "
                       "before the exporter ships the release to consumers.",
                       "decision", "caching",
                       "journals every published payload verbatim",
                       obsolete=True),
                    _f("ra.a.old.002", 105,
                       "The journal file pending_releases.json keeps one JSON "
                       "line per published release and is never truncated by "
                       "the adapter during normal operation.",
                       "decision", "caching",
                       "never truncated by the adapter",
                       obsolete=True),
                    _f("ra.a.old.003", 142,
                       "A published release payload must remain recoverable "
                       "from the local journal, so the adapter appends the "
                       "raw payload line before any export completes.",
                       "decision", "caching",
                       "appends the raw payload line",
                       obsolete=True),
                    _f("ra.a.old.004", 178,
                       "The adapter treats the local journal as the system of "
                       "record for what was published, reading it back when "
                       "reconciling consumer acknowledgements.",
                       "decision", "caching",
                       "local journal as the system of record",
                       obsolete=True),
                ],
                current=[
                    _f("ra.a.cur.001", 424,
                       "Correction to an earlier note: the release adapter "
                       "does not journal published payloads into any local "
                       "pending_releases.json file and nothing is written to "
                       "disk during a publish.",
                       "correction", "caching",
                       "does not journal published payloads",
                       obsolete=False),
                    _f("ra.a.cur.002", 452,
                       "The adapter publishes directly through the exporter "
                       "and records only a release id and published timestamp "
                       "as one audit row in the store.",
                       "correction", "caching",
                       "records only a release id and published timestamp",
                       obsolete=False),
                    _f("ra.a.cur.003", 496,
                       "Correction to an earlier note: raw release payloads "
                       "are never written to any local file or journal and "
                       "the audit stream is the only record of a publish.",
                       "correction", "caching",
                       "never written to any local file",
                       obsolete=False),
                    _f("ra.a.cur.004", 538,
                       "The adapter must not append payload lines to "
                       "pending_releases.json because the local journal was "
                       "removed from the contract entirely.",
                       "correction", "caching",
                       "local journal was removed",
                       obsolete=False),
                ],
                distractors=[
                    _f("ra.a.d1", 129,
                       "The exporter computes a delivery digest per release id "
                       "and deduplicates retried deliveries before forwarding "
                       "the release to consumers.",
                       "distractor", "caching",
                       "delivery digest per release id"),
                    _f("ra.a.d2", 254,
                       "The nightly release schedule writes a build timestamp "
                       "into every container image tag that the pipeline "
                       "publishes to the registry.",
                       "distractor", "caching",
                       "writes a build timestamp"),
                    _f("ra.a.d3", 352,
                       "The storage tier runs a weekend compaction over the "
                       "audit table so reclaimed space remains available for "
                       "the newest entries.",
                       "distractor", "database",
                       "weekend compaction over the audit table"),
                    _f("ra.a.d4", 558,
                       "The monitoring dashboard renders exporter latency for "
                       "every release so operators can spot slow shippings in "
                       "real time.",
                       "distractor", "caching",
                       "renders exporter latency"),
                ],
                corrections=[
                    ("ra.a.old.001", "ra.a.cur.001"),
                    ("ra.a.old.002", "ra.a.cur.002"),
                    ("ra.a.old.003", "ra.a.cur.003"),
                    ("ra.a.old.004", "ra.a.cur.004"),
                ],
                gold={
                    "releaseapp/adapter.py": (
                        '"""Release publishing adapter."""\n'
                        "\n"
                        "def publish_release(store, exporter, payload):\n"
                        '    """Publish one release payload according to '
                        'repository contract."""\n'
                        "    exporter.export(payload)\n"
                        '    store.write_audit({"release_id": '
                        'payload["release_id"], "published_ts": '
                        'payload.get("published_ts")})\n'
                    ),
                },
                obsolete_gold={
                    "releaseapp/adapter.py": (
                        '"""Release publishing adapter."""\n'
                        "\n"
                        "import json\n"
                        "from pathlib import Path\n"
                        "\n"
                        "def publish_release(store, exporter, payload):\n"
                        '    """Publish one release payload according to '
                        'repository contract."""\n'
                        '    with Path("pending_releases.json").open("a") as fh:\n'
                        '        fh.write(json.dumps(payload) + "\\n")\n'
                        "    exporter.export(payload)\n"
                        '    store.write_audit({"release_id": '
                        'payload["release_id"], "published_ts": '
                        'payload.get("published_ts")})\n'
                    ),
                },
            ),
            DiscriminativeVariant(
                variant="B",
                dir="variant_b",
                decision_summary=(
                    "the adapter must journal every published release payload "
                    "to the local pending_releases.json file before export, "
                    "recording the journal as the delivery source"
                ),
                obsolete=[
                    _f("ra.b.old.001", 78,
                       "The release adapter previously trusted an external "
                       "outbox queue and never wrote release payloads to any "
                       "local file during a publish.",
                       "decision", "caching",
                       "never wrote release payloads",
                       obsolete=True),
                    _f("ra.b.old.002", 109,
                       "Delivery was complete once the outbox queue accepted "
                       "the release, so local disk storage was not part of "
                       "the publishing path.",
                       "decision", "caching",
                       "local disk storage was not part",
                       obsolete=True),
                    _f("ra.b.old.003", 146,
                       "The outbox acknowledged a release immediately on "
                       "acceptance and the adapter surfaced that "
                       "acknowledgement as the audit record.",
                       "decision", "caching",
                       "outbox acknowledged a release",
                       obsolete=True),
                    _f("ra.b.old.004", 178,
                       "Reconciliation read from the external outbox service "
                       "rather than from any local journal file because none "
                       "existed on the host during that period.",
                       "decision", "caching",
                       "rather than from any local journal",
                       obsolete=True),
                ],
                current=[
                    _f("ra.b.cur.001", 427,
                       "Correction to an earlier note: the release adapter "
                       "now journals every published release payload to "
                       "pending_releases.json before the export happens.",
                       "correction", "caching",
                       "journals every published release payload",
                       obsolete=False),
                    _f("ra.b.cur.002", 458,
                       "The local pending_releases.json journal is the system "
                       "of record, appended one JSON line per published "
                       "release before export.",
                       "correction", "caching",
                       "system of record",
                       obsolete=False),
                    _f("ra.b.cur.003", 502,
                       "Correction to an earlier note: a release payload is "
                       "not considered published until its raw payload line "
                       "has been appended to the local journal.",
                       "correction", "caching",
                       "not considered published",
                       obsolete=False),
                    _f("ra.b.cur.004", 538,
                       "The adapter must write the raw payload to "
                       "pending_releases.json and record the journal as the "
                       "delivery source in the audit row.",
                       "correction", "caching",
                       "record the journal as the delivery source",
                       obsolete=False),
                ],
                distractors=[
                    _f("ra.b.d1", 133,
                       "The exporter computes a delivery digest per release id "
                       "and deduplicates retried deliveries before forwarding "
                       "the release to consumers.",
                       "distractor", "caching",
                       "delivery digest per release id"),
                    _f("ra.b.d2", 258,
                       "The nightly release schedule writes a build timestamp "
                       "into every container image tag that the pipeline "
                       "publishes to the registry.",
                       "distractor", "caching",
                       "writes a build timestamp"),
                    _f("ra.b.d3", 356,
                       "The storage tier runs a weekend compaction over the "
                       "audit table so reclaimed space remains available for "
                       "the newest entries.",
                       "distractor", "database",
                       "weekend compaction over the audit table"),
                    _f("ra.b.d4", 564,
                       "The monitoring dashboard renders exporter latency for "
                       "every release so operators can spot slow shippings in "
                       "real time.",
                       "distractor", "caching",
                       "renders exporter latency"),
                ],
                corrections=[
                    ("ra.b.old.001", "ra.b.cur.001"),
                    ("ra.b.old.002", "ra.b.cur.002"),
                    ("ra.b.old.003", "ra.b.cur.003"),
                    ("ra.b.old.004", "ra.b.cur.004"),
                ],
                gold={
                    "releaseapp/adapter.py": (
                        '"""Release publishing adapter."""\n'
                        "\n"
                        "import json\n"
                        "from pathlib import Path\n"
                        "\n"
                        "def publish_release(store, exporter, payload):\n"
                        '    """Publish one release payload according to '
                        'repository contract."""\n'
                        '    with Path("pending_releases.json").open("a") as fh:\n'
                        '        fh.write(json.dumps(payload) + "\\n")\n'
                        "    exporter.export(payload)\n"
                        '    store.write_audit({"release_id": '
                        'payload["release_id"], "published_ts": '
                        'payload.get("published_ts"), "journal": '
                        '"pending_releases.json"})\n'
                    ),
                },
                obsolete_gold={
                    "releaseapp/adapter.py": (
                        '"""Release publishing adapter."""\n'
                        "\n"
                        "def publish_release(store, exporter, payload):\n"
                        '    """Publish one release payload according to '
                        'repository contract."""\n'
                        "    exporter.export(payload)\n"
                        '    store.write_audit({"release_id": '
                        'payload["release_id"], "published_ts": '
                        'payload.get("published_ts")})\n'
                    ),
                },
            ),
        ],
    ),
    # =======================================================================
    # invoice_adapter
    # =======================================================================
    DiscriminativeGroup(
        group_id="invoice_adapter",
        dir="invoice_adapter",
        prompt=(
            "Implement `normalize_invoice(row, registry)` in "
            "`invoiceapp/adapter.py` according to the repository's established "
            "contract. Keep the existing public name and signature. Do not "
            "modify the hidden tests."
        ),
        workspace={
            "invoiceapp/__init__.py": (
                '"""invoiceapp - invoice normalization adapter used in the E26 '
                'discriminative suite."""\n'
            ),
            "invoiceapp/adapter.py": (
                '"""Invoice normalization adapter."""\n'
                "\n"
                "def normalize_invoice(row, registry):\n"
                '    """Normalize one invoice row according to repository '
                'contract."""\n'
                "    raise NotImplementedError\n"
            ),
            "README.md": (
                "# invoiceapp\n\n"
                "Invoice normalization adapter.\n\n"
                "`invoiceapp.adapter.normalize_invoice` is the public "
                "normalization entry point.\n"
            ),
        },
        variants=[
            DiscriminativeVariant(
                variant="A",
                dir="variant_a",
                decision_summary=(
                    "unknown line-item codes are preserved verbatim through "
                    "normalization; no fallback display name is applied"
                ),
                obsolete=[
                    _f("ia.a.old.001", 76,
                       "The invoice normalizer maps every unrecognised "
                       "line-item code to the generic MISCELLANEOUS code so "
                       "downstream reports stay grouped.",
                       "decision", "database",
                       "maps every unrecognised line-item code",
                       obsolete=True),
                    _f("ia.a.old.002", 107,
                       "An unknown line-item code is replaced by "
                       "MISCELLANEOUS and its original code is discarded "
                       "before the row leaves the adapter.",
                       "decision", "database",
                       "original code is discarded",
                       obsolete=True),
                    _f("ia.a.old.003", 144,
                       "The registry lookup fallback maps unknowns to "
                       "MISCELLANEOUS with the display name Miscellaneous for "
                       "every report that consumes invoices.",
                       "decision", "database",
                       "maps unknowns to MISCELLANEOUS",
                       obsolete=True),
                    _f("ia.a.old.004", 180,
                       "Unknown line-item codes never survive normalization "
                       "because forward consumers expect only recognised codes "
                       "to appear in their downstream input stream.",
                       "decision", "database",
                       "never survive normalization",
                       obsolete=True),
                ],
                current=[
                    _f("ia.a.cur.001", 426,
                       "Correction to an earlier note: the invoice normalizer "
                       "preserves unrecognised line-item codes verbatim and "
                       "does not map them to MISCELLANEOUS.",
                       "correction", "database",
                       "preserves unrecognised line-item codes",
                       obsolete=False),
                    _f("ia.a.cur.002", 454,
                       "An unknown line-item code keeps its original code "
                       "string and no fallback display name is applied to it "
                       "during normalization.",
                       "correction", "database",
                       "no fallback display name",
                       obsolete=False),
                    _f("ia.a.cur.003", 498,
                       "Unknown line-item codes survive normalization "
                       "unchanged so forward consumers can see exactly the "
                       "original code that the vendor sent on the invoice.",
                       "correction", "database",
                       "survive normalization unchanged",
                       obsolete=False),
                    _f("ia.a.cur.004", 540,
                       "The MISCELLANEOUS mapping was removed from the "
                       "contract and the adapter never rewrites an "
                       "unrecognised code to it.",
                       "correction", "database",
                       "never rewrites an unrecognised code",
                       obsolete=False),
                ],
                distractors=[
                    _f("ia.a.d1", 131,
                       "The approval queue holds invoices over a configured "
                       "threshold amount until manual review before processing "
                       "can continue downstream in the pipeline.",
                       "distractor", "auth",
                       "manual review before processing"),
                    _f("ia.a.d2", 256,
                       "Currency rounding is applied only at report time and "
                       "is never stored on the normalized invoice row by the "
                       "adapter.",
                       "distractor", "database",
                       "applied only at report time"),
                    _f("ia.a.d3", 354,
                       "Batch exports page through normalized invoices in "
                       "fixed chunks of five hundred rows each for the "
                       "downstream warehouse load job.",
                       "distractor", "database",
                       "chunks of five hundred rows"),
                    _f("ia.a.d4", 561,
                       "The vendor code registry is refreshed from a vendor "
                       "feed each night and the cached display names are held "
                       "in memory for fast lookups.",
                       "distractor", "caching",
                       "refreshed from a vendor feed"),
                ],
                corrections=[
                    ("ia.a.old.001", "ia.a.cur.001"),
                    ("ia.a.old.002", "ia.a.cur.002"),
                    ("ia.a.old.003", "ia.a.cur.003"),
                    ("ia.a.old.004", "ia.a.cur.004"),
                ],
                gold={
                    "invoiceapp/adapter.py": (
                        '"""Invoice normalization adapter."""\n'
                        "\n"
                        "def normalize_invoice(row, registry):\n"
                        '    """Normalize one invoice row according to '
                        'repository contract."""\n'
                        '    return {"invoice_id": row["invoice_id"], '
                        '"line_items": [{"id": i["id"], "code": i["code"]} '
                        "for i in row[\"line_items\"]]}\n"
                    ),
                },
                obsolete_gold={
                    "invoiceapp/adapter.py": (
                        '"""Invoice normalization adapter."""\n'
                        "\n"
                        "def normalize_invoice(row, registry):\n"
                        '    """Normalize one invoice row according to '
                        'repository contract."""\n'
                        '    return {"invoice_id": row["invoice_id"], '
                        '"line_items": [{"id": i["id"], "code": '
                        '"MISCELLANEOUS" if i["code"] not in registry else '
                        'i["code"]} for i in row["line_items"]]}\n'
                    ),
                },
            ),
            DiscriminativeVariant(
                variant="B",
                dir="variant_b",
                decision_summary=(
                    "unknown line-item codes must be remapped to the generic "
                    "MISCELLANEOUS code during normalization"
                ),
                obsolete=[
                    _f("ia.b.old.001", 80,
                       "The invoice normalizer previously carried every "
                       "line-item code it read, including codes written by "
                       "future versions of the vendor software.",
                       "decision", "database",
                       "carried every line-item code",
                       obsolete=True),
                    _f("ia.b.old.002", 111,
                       "Downstream reports enumerated an explicit allowlist of "
                       "codes so unexpected codes were ignored by design and "
                       "never rendered.",
                       "decision", "database",
                       "explicit allowlist of codes",
                       obsolete=True),
                    _f("ia.b.old.003", 148,
                       "A missing registry entry left the original code "
                       "untouched so the auditor could trace the vendor "
                       "number exactly.",
                       "decision", "database",
                       "left the original code untouched",
                       obsolete=True),
                    _f("ia.b.old.004", 178,
                       "Preserving unknown codes was deliberate so that no "
                       "report ever grouped distinct vendor items under a "
                       "shared fallback.",
                       "decision", "database",
                       "no report ever grouped distinct vendor items",
                       obsolete=True),
                ],
                current=[
                    _f("ia.b.cur.001", 429,
                       "Correction to an earlier note: the invoice normalizer "
                       "now maps every unrecognised line-item code to the "
                       "generic MISCELLANEOUS code.",
                       "correction", "database",
                       "maps every unrecognised line-item code",
                       obsolete=False),
                    _f("ia.b.cur.002", 460,
                       "An unknown line-item code is replaced by "
                       "MISCELLANEOUS and its original code is discarded "
                       "before the row leaves the adapter.",
                       "correction", "database",
                       "original code is discarded",
                       obsolete=False),
                    _f("ia.b.cur.003", 504,
                       "Correction to an earlier note: a missing registry "
                       "entry must fall back to MISCELLANEOUS so new vendor "
                       "items group under one code.",
                       "correction", "database",
                       "must fall back to MISCELLANEOUS",
                       obsolete=False),
                    _f("ia.b.cur.004", 538,
                       "The normalizer rewrites every unrecognised line-item "
                       "code to the MISCELLANEOUS code and no original code "
                       "outside the registry survives normalization at all.",
                       "correction", "database",
                       "no original code outside the registry survives",
                       obsolete=False),
                ],
                distractors=[
                    _f("ia.b.d1", 135,
                       "The approval queue holds invoices over a configured "
                       "threshold amount until manual review before processing "
                       "can continue downstream in the pipeline.",
                       "distractor", "auth",
                       "manual review before processing"),
                    _f("ia.b.d2", 260,
                       "Currency rounding is applied only at report time and "
                       "is never stored on the normalized invoice row by the "
                       "adapter.",
                       "distractor", "database",
                       "applied only at report time"),
                    _f("ia.b.d3", 358,
                       "Batch exports page through normalized invoices in "
                       "fixed chunks of five hundred rows each for the "
                       "downstream warehouse load job.",
                       "distractor", "database",
                       "chunks of five hundred rows"),
                    _f("ia.b.d4", 566,
                       "The vendor code registry is refreshed from a vendor "
                       "feed each night and the cached display names are held "
                       "in memory for fast lookups.",
                       "distractor", "caching",
                       "refreshed from a vendor feed"),
                ],
                corrections=[
                    ("ia.b.old.001", "ia.b.cur.001"),
                    ("ia.b.old.002", "ia.b.cur.002"),
                    ("ia.b.old.003", "ia.b.cur.003"),
                    ("ia.b.old.004", "ia.b.cur.004"),
                ],
                gold={
                    "invoiceapp/adapter.py": (
                        '"""Invoice normalization adapter."""\n'
                        "\n"
                        "def normalize_invoice(row, registry):\n"
                        '    """Normalize one invoice row according to '
                        'repository contract."""\n'
                        '    return {"invoice_id": row["invoice_id"], '
                        '"line_items": [{"id": i["id"], "code": '
                        '"MISCELLANEOUS" if i["code"] not in registry else '
                        'i["code"]} for i in row["line_items"]]}\n'
                    ),
                },
                obsolete_gold={
                    "invoiceapp/adapter.py": (
                        '"""Invoice normalization adapter."""\n'
                        "\n"
                        "def normalize_invoice(row, registry):\n"
                        '    """Normalize one invoice row according to '
                        'repository contract."""\n'
                        '    return {"invoice_id": row["invoice_id"], '
                        '"line_items": [{"id": i["id"], "code": i["code"]} '
                        "for i in row[\"line_items\"]]}\n"
                    ),
                },
            ),
        ],
    ),
    # =======================================================================
    # message_adapter
    # =======================================================================
    DiscriminativeGroup(
        group_id="message_adapter",
        dir="message_adapter",
        prompt=(
            "Implement `deliver(relay, message)` in `messageapp/adapter.py` "
            "according to the repository's established contract. Keep the "
            "existing public name and signature. Do not modify the hidden "
            "tests."
        ),
        workspace={
            "messageapp/__init__.py": (
                '"""messageapp - message delivery adapter used in the E26 '
                'discriminative suite."""\n'
            ),
            "messageapp/adapter.py": (
                '"""Message delivery adapter."""\n'
                "\n"
                "def deliver(relay, message):\n"
                '    """Deliver one message according to repository '
                'contract."""\n'
                "    raise NotImplementedError\n"
            ),
            "README.md": (
                "# messageapp\n\n"
                "Message delivery adapter.\n\n"
                "`messageapp.adapter.deliver` is the public delivery entry "
                "point.\n"
            ),
        },
        variants=[
            DiscriminativeVariant(
                variant="A",
                dir="variant_a",
                decision_summary=(
                    "a message counts as delivered only after the consumer "
                    "confirms it; the adapter must wait for confirmation"
                ),
                obsolete=[
                    _f("ma.a.old.001", 72,
                       "The message adapter marks a message delivered as "
                       "soon as the relay accepts it into the outbound "
                       "queue.",
                       "decision", "database",
                       "delivered as soon as the relay accepts it",
                       obsolete=True),
                    _f("ma.a.old.002", 103,
                       "Delivery is complete when the relay queue holds the "
                       "message and confirmation from the consumer is not "
                       "part of the contract.",
                       "decision", "database",
                       "confirmation from the consumer is not part",
                       obsolete=True),
                    _f("ma.a.old.003", 140,
                       "The relay acceptance response alone determines "
                       "overall success in this contract, so the adapter "
                       "never waits on downstream consumer confirmation at "
                       "any point.",
                       "decision", "database",
                       "never waits on downstream consumer confirmation",
                       obsolete=True),
                    _f("ma.a.old.004", 176,
                       "Messages placed on the relay queue are counted as "
                       "delivered regardless of whether the consumer ever "
                       "validates them afterwards.",
                       "decision", "database",
                       "counted as delivered regardless",
                       obsolete=True),
                ],
                current=[
                    _f("ma.a.cur.001", 422,
                       "A message is considered delivered by the adapter only "
                       "when the downstream consumer explicitly confirms it "
                       "after validation.",
                       "correction", "database",
                       "only when the downstream consumer explicitly confirms",
                       obsolete=False),
                    _f("ma.a.cur.002", 450,
                       "Correction to an earlier note: accepting a message "
                       "into the relay queue is not delivery and the adapter "
                       "must wait for consumer confirmation.",
                       "correction", "database",
                       "must wait for consumer confirmation",
                       obsolete=False),
                    _f("ma.a.cur.003", 494,
                       "The deliver helper returns True only after the relay "
                       "reports the consumer confirmation and otherwise "
                       "surfaces the failure.",
                       "correction", "database",
                       "reports the consumer confirmation",
                       obsolete=False),
                    _f("ma.a.cur.004", 536,
                       "A message that was accepted but not confirmed by the "
                       "downstream consumer is treated as undelivered under "
                       "the adapter contract.",
                       "correction", "database",
                       "treated as undelivered",
                       obsolete=False),
                ],
                distractors=[
                    _f("ma.a.d1", 127,
                       "The fan-out worker batches relay deliveries every "
                       "five seconds during peak load to reduce connection "
                       "churn across geographic regions.",
                       "distractor", "caching",
                       "batches relay deliveries"),
                    _f("ma.a.d2", 252,
                       "Queue depth is surfaced on the dashboards so "
                       "operators can observe backlogs during traffic peaks "
                       "in real time.",
                       "distractor", "database",
                       "observe backlogs during traffic peaks"),
                    _f("ma.a.d3", 350,
                       "Repeated retries reuse the original message id so "
                       "downstream handlers can deduplicate replayed "
                       "deliveries safely across process restarts.",
                       "distractor", "database",
                       "deduplicate replayed deliveries"),
                    _f("ma.a.d4", 556,
                       "The transport layer encrypts message bodies both at "
                       "rest and in transit using the shared key vault "
                       "service.",
                       "distractor", "auth",
                       "encrypts message bodies"),
                ],
                corrections=[
                    ("ma.a.old.001", "ma.a.cur.001"),
                    ("ma.a.old.002", "ma.a.cur.002"),
                    ("ma.a.old.003", "ma.a.cur.003"),
                    ("ma.a.old.004", "ma.a.cur.004"),
                ],
                gold={
                    "messageapp/adapter.py": (
                        '"""Message delivery adapter."""\n'
                        "\n"
                        "def deliver(relay, message):\n"
                        '    """Deliver one message according to repository '
                        'contract."""\n'
                        "    relay.accept(message)\n"
                        "    if not relay.wait_confirmation():\n"
                        '        raise RuntimeError("message not confirmed by '
                        'consumer")\n'
                        "    return True\n"
                    ),
                },
                obsolete_gold={
                    "messageapp/adapter.py": (
                        '"""Message delivery adapter."""\n'
                        "\n"
                        "def deliver(relay, message):\n"
                        '    """Deliver one message according to repository '
                        'contract."""\n'
                        "    relay.accept(message)\n"
                        "    return True\n"
                    ),
                },
            ),
            DiscriminativeVariant(
                variant="B",
                dir="variant_b",
                decision_summary=(
                    "acceptance by the relay IS delivery; the adapter must "
                    "never wait for downstream consumer confirmation"
                ),
                obsolete=[
                    _f("ma.b.old.001", 75,
                       "The message adapter previously considered a message "
                       "fully delivered only after the consumer explicitly "
                       "confirmed validation of its body.",
                       "decision", "database",
                       "only after the consumer explicitly confirmed",
                       obsolete=True),
                    _f("ma.b.old.002", 106,
                       "Delivery was deferred until the relay gathered a "
                       "downstream confirmation even though the accept call "
                       "had already succeeded.",
                       "decision", "database",
                       "deferred until the relay gathered",
                       obsolete=True),
                    _f("ma.b.old.003", 143,
                       "The old adapter blocked on consumer confirmation and "
                       "a slow or entirely silent consumer stalled the whole "
                       "delivery path indefinitely.",
                       "decision", "database",
                       "blocked on consumer confirmation",
                       obsolete=True),
                    _f("ma.b.old.004", 179,
                       "Unconfirmed messages could then be retried "
                       "indefinitely because the old contract treated waiting "
                       "as mandatory before any success was reached.",
                       "decision", "database",
                       "treated waiting as mandatory",
                       obsolete=True),
                ],
                current=[
                    _f("ma.b.cur.001", 425,
                       "Correction to an earlier note: a message is delivered "
                       "as soon as the relay accepts it and the adapter never "
                       "waits for confirmation.",
                       "correction", "database",
                       "delivered as soon as the relay accepts it",
                       obsolete=False),
                    _f("ma.b.cur.002", 453,
                       "The deliver helper returns the relay acceptance "
                       "result directly and the downstream consumer "
                       "confirmation is not consulted at all during delivery.",
                       "correction", "database",
                       "returns the relay acceptance result",
                       obsolete=False),
                    _f("ma.b.cur.003", 497,
                       "Acceptance by the relay is the completion signal and "
                       "no blocking wait on the consumer is part of the "
                       "contract.",
                       "correction", "database",
                       "no blocking wait on the consumer",
                       obsolete=False),
                    _f("ma.b.cur.004", 539,
                       "The adapter must not wait for consumer confirmation "
                       "because the relay acceptance alone decides that the "
                       "delivery succeeded.",
                       "correction", "database",
                       "must not wait for consumer confirmation",
                       obsolete=False),
                ],
                distractors=[
                    _f("ma.b.d1", 130,
                       "The fan-out worker batches relay deliveries every "
                       "five seconds during peak load to reduce connection "
                       "churn across geographic regions.",
                       "distractor", "caching",
                       "batches relay deliveries"),
                    _f("ma.b.d2", 255,
                       "Queue depth is surfaced on the dashboards so "
                       "operators can observe backlogs during traffic peaks "
                       "in real time.",
                       "distractor", "database",
                       "observe backlogs during traffic peaks"),
                    _f("ma.b.d3", 353,
                       "Repeated retries reuse the original message id so "
                       "downstream handlers can deduplicate replayed "
                       "deliveries safely across process restarts.",
                       "distractor", "database",
                       "deduplicate replayed deliveries"),
                    _f("ma.b.d4", 559,
                       "The transport layer encrypts message bodies both at "
                       "rest and in transit using the shared key vault "
                       "service.",
                       "distractor", "auth",
                       "encrypts message bodies"),
                ],
                corrections=[
                    ("ma.b.old.001", "ma.b.cur.001"),
                    ("ma.b.old.002", "ma.b.cur.002"),
                    ("ma.b.old.003", "ma.b.cur.003"),
                    ("ma.b.old.004", "ma.b.cur.004"),
                ],
                gold={
                    "messageapp/adapter.py": (
                        '"""Message delivery adapter."""\n'
                        "\n"
                        "def deliver(relay, message):\n"
                        '    """Deliver one message according to repository '
                        'contract."""\n'
                        "    return relay.accept(message)\n"
                    ),
                },
                obsolete_gold={
                    "messageapp/adapter.py": (
                        '"""Message delivery adapter."""\n'
                        "\n"
                        "def deliver(relay, message):\n"
                        '    """Deliver one message according to repository '
                        'contract."""\n'
                        "    relay.accept(message)\n"
                        "    return relay.wait_confirmation()\n"
                    ),
                },
            ),
        ],
    ),
]

_GROUP_BY_ID: Dict[str, DiscriminativeGroup] = {g.group_id: g for g in _GROUPS}


def list_group_ids() -> List[str]:
    return [g.group_id for g in _GROUPS]


def list_variants(group_id: str) -> List[str]:
    return [v.variant for v in _GROUP_BY_ID[group_id].variants]


def _variant_spec(group_id: str, variant: str) -> DiscriminativeVariant:
    for vs in _GROUP_BY_ID[group_id].variants:
        if vs.variant == variant:
            return vs
    raise KeyError(f"{group_id}: unknown variant {variant!r}; "
                   f"options={list_variants(group_id)}")


# ---------------------------------------------------------------------------
# Hidden-test templates (variant-specific; never leak into history or prompts)
# ---------------------------------------------------------------------------

_HIDDEN_TESTS: Dict[Tuple[str, str], str] = {
    ("release_adapter", "A"): (
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "import pytest\n"
        "\n"
        "from releaseapp.adapter import publish_release\n"
        "\n"
        "\n"
        "class FakeExporter:\n"
        "    def __init__(self):\n"
        "        self.published = []\n"
        "\n"
        "    def export(self, release):\n"
        "        self.published.append(release)\n"
        "\n"
        "\n"
        "class FakeStore:\n"
        "    def __init__(self):\n"
        "        self.stream = []\n"
        "\n"
        "    def write_audit(self, entry):\n"
        "        self.stream.append(entry)\n"
        "\n"
        "\n"
        "@pytest.fixture(autouse=True)\n"
        "def _no_journal(tmp_path, monkeypatch):\n"
        "    monkeypatch.chdir(tmp_path)\n"
        "    yield\n"
        "\n"
        "\n"
        "def test_publishes_directly_and_never_journals():\n"
        "    store = FakeStore()\n"
        "    exporter = FakeExporter()\n"
        "    payload = {\"release_id\": \"rel-7\", \"published_ts\": 1717000000}\n"
        "    publish_release(store, exporter, payload)\n"
        "    assert exporter.published == [payload]\n"
        "    assert not Path(\"pending_releases.json\").exists()\n"
        "    assert store.stream == [\n"
        '        {"release_id": "rel-7", "published_ts": 1717000000}\n'
        "    ]\n"
    ),
    ("release_adapter", "B"): (
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "import pytest\n"
        "\n"
        "from releaseapp.adapter import publish_release\n"
        "\n"
        "\n"
        "class FakeExporter:\n"
        "    def __init__(self):\n"
        "        self.published = []\n"
        "\n"
        "    def export(self, release):\n"
        "        self.published.append(release)\n"
        "\n"
        "\n"
        "class FakeStore:\n"
        "    def __init__(self):\n"
        "        self.stream = []\n"
        "\n"
        "    def write_audit(self, entry):\n"
        "        self.stream.append(entry)\n"
        "\n"
        "\n"
        "@pytest.fixture(autouse=True)\n"
        "def _sandbox(tmp_path, monkeypatch):\n"
        "    monkeypatch.chdir(tmp_path)\n"
        "    yield\n"
        "\n"
        "\n"
        "def test_journals_raw_payload_before_export():\n"
        "    store = FakeStore()\n"
        "    exporter = FakeExporter()\n"
        "    payload = {\"release_id\": \"rel-9\", \"published_ts\": 1717100000}\n"
        "    publish_release(store, exporter, payload)\n"
        "    assert exporter.published == [payload]\n"
        '    lines = Path("pending_releases.json").read_text().splitlines()\n'
        '    assert json.loads(lines[0]) == payload\n'
        "    assert store.stream == [\n"
        '        {"release_id": "rel-9", "published_ts": 1717100000, '
        '"journal": "pending_releases.json"}\n'
        "    ]\n"
    ),
    ("invoice_adapter", "A"): (
        "from invoiceapp.adapter import normalize_invoice\n"
        "\n"
        "\n"
        "def test_preserves_unknown_line_item_codes():\n"
        '    row = {"invoice_id": 9, "line_items": [\n'
        '        {"id": 1, "code": "LINE-07"},\n'
        '        {"id": 2, "code": "VENDOR-X"},\n'
        "    ]}\n"
        '    registry = {"LINE-07": "Standard Line"}\n'
        "    out = normalize_invoice(row, registry)\n"
        '    assert [i["code"] for i in out["line_items"]] == '
        '["LINE-07", "VENDOR-X"]\n'
        "\n"
        "\n"
        "def test_known_codes_unaffected():\n"
        '    row = {"invoice_id": 3, "line_items": [{"id": 1, "code": "A"}]}\n'
        '    out = normalize_invoice(row, {"A": "Known"})\n'
        '    assert out["line_items"][0]["code"] == "A"\n'
    ),
    ("invoice_adapter", "B"): (
        "from invoiceapp.adapter import normalize_invoice\n"
        "\n"
        "\n"
        "def test_maps_unknown_line_item_codes():\n"
        '    row = {"invoice_id": 9, "line_items": [\n'
        '        {"id": 1, "code": "LINE-07"},\n'
        '        {"id": 2, "code": "VENDOR-X"},\n'
        "    ]}\n"
        "    out = normalize_invoice(row, {})\n"
        '    assert [i["code"] for i in out["line_items"]] == '
        '["MISCELLANEOUS", "MISCELLANEOUS"]\n'
        "\n"
        "\n"
        "def test_known_codes_survive():\n"
        '    row = {"invoice_id": 3, "line_items": [{"id": 1, "code": "A"}]}\n'
        '    out = normalize_invoice(row, {"A": "Known"})\n'
        '    assert out["line_items"][0]["code"] == "A"\n'
    ),
    ("message_adapter", "A"): (
        "from messageapp.adapter import deliver\n"
        "\n"
        "\n"
        "class FakeRelay:\n"
        "    def __init__(self, confirmed):\n"
        "        self.confirmed = confirmed\n"
        "        self.accepted = []\n"
        "        self.waits = 0\n"
        "\n"
        "    def accept(self, message):\n"
        "        self.accepted.append(message)\n"
        "        return True\n"
        "\n"
        "    def wait_confirmation(self):\n"
        "        self.waits += 1\n"
        "        return self.confirmed\n"
        "\n"
        "\n"
        "def test_requires_consumer_confirmation():\n"
        "    relay = FakeRelay(confirmed=False)\n"
        "    try:\n"
        '        deliver(relay, {"id": 1, "body": "x"})\n'
        "    except RuntimeError:\n"
        "        pass\n"
        "    else:\n"
        '        raise AssertionError("expected failure without confirmation")\n'
        "    assert relay.waits == 1\n"
        "\n"
        "\n"
        "def test_returns_true_when_confirmed():\n"
        "    relay = FakeRelay(confirmed=True)\n"
        '    assert deliver(relay, {"id": 2}) is True\n'
        '    assert relay.accepted == [{"id": 2}]\n'
        "    assert relay.waits == 1\n"
    ),
    ("message_adapter", "B"): (
        "from messageapp.adapter import deliver\n"
        "\n"
        "\n"
        "class FakeRelay:\n"
        "    def __init__(self, accepted_result=True):\n"
        "        self.accepted_result = accepted_result\n"
        "        self.calls = 0\n"
        "        self.waits = 0\n"
        "\n"
        "    def accept(self, message):\n"
        "        self.calls += 1\n"
        "        return self.accepted_result\n"
        "\n"
        "    def wait_confirmation(self):\n"
        "        self.waits += 1\n"
        "        return True\n"
        "\n"
        "\n"
        "def test_acceptance_is_delivery():\n"
        "    relay = FakeRelay()\n"
        '    assert deliver(relay, {"id": 5}) is True\n'
        "    assert relay.waits == 0\n"
        "\n"
        "\n"
        "def test_forwards_accept_result():\n"
        "    relay = FakeRelay(accepted_result=False)\n"
        '    assert deliver(relay, {"id": 6}) is False\n'
    ),
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def group_dir(group_id: str) -> Path:
    return _DISC_DIR / _GROUP_BY_ID[group_id].dir


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
# Transcript generation (deterministic, mirrors coding_task_suite conventions)
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
class PolicyCorrection:
    obsolete_fact_id: str
    current_fact_id: str


@dataclass(frozen=True)
class DiscriminativeTask:
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
    policy_facts: List[PolicyFact]       # current (superseding) facts
    obsolete_policy_facts: List[PolicyFact]  # superseded facts
    distractor_facts: List[HistoricalFact]
    corrections: List[PolicyCorrection]
    gold_patch_path: Path
    obsolete_gold_patch_path: Path
    seed: int
    history_turns: int
    metadata: Dict = field(default_factory=dict)

    @property
    def all_policy_facts(self) -> List[PolicyFact]:
        return list(self.obsolete_policy_facts) + list(self.policy_facts)


def _policy_from_dict(raw: Dict) -> PolicyFact:
    return PolicyFact(
        fact_id=raw["fact_id"], turn_index=raw["turn_index"], text=raw["text"],
        kind=raw["kind"], topic=raw["topic"], probe=raw["probe"],
        old=bool(raw.get("obsolete", False)),
    )


# ---------------------------------------------------------------------------
# Fixture materialisation
# ---------------------------------------------------------------------------

def _build_patch(group: DiscriminativeGroup, workspace: Path,
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
        subprocess.run(["git", "-c", "user.email=e26@local",
                        "-c", "user.name=e26", "init", "-q"],
                       cwd=str(repo), check=True, capture_output=True, env=env)
        subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True,
                       capture_output=True, env=env)
        subprocess.run(["git", "-c", "user.email=e26@local",
                        "-c", "user.name=e26", "commit", "-q", "-m", "base"],
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


def _variant_metadata(group: DiscriminativeGroup,
                      vspec: DiscriminativeVariant) -> Dict:
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
# build_variant: assemble a DiscriminativeTask for the benchmark
# ---------------------------------------------------------------------------

def build_variant(group_id: str, variant: str, seed: int = 1,
                  history_turns: int = HISTORY_TURNS_DEFAULT) -> DiscriminativeTask:
    """Build a fully-populated ``DiscriminativeTask`` for one variant.

    The workspace and task prompt are the *shared* group fixtures (byte-identical
    across variants); only history, hidden test and gold/obsolete patches are
    variant specific. The committed ``history.txt`` must match a fresh
    regeneration for ``seed=1``.
    """
    group = _GROUP_BY_ID[group_id]
    vspec = _variant_spec(group_id, variant)
    hpath = history_path(group_id, variant)
    if not hpath.is_file():
        raise FileNotFoundError(
            f"missing committed history for {group_id}/{variant}: {hpath}; "
            "run `python -m data.discriminative_coding_suite --write` to "
            "materialise fixtures first")
    committed = hpath.read_text(encoding="utf-8").splitlines()
    hist, hist_text = _build_history(
        {"task_id": f"{group_id}_{variant}",
         "critical": vspec.obsolete + vspec.current,
         "distractors": vspec.distractors},
        seed=seed, turns=history_turns)

    # Propagate the explicit correction metadata into the structured history
    # so the production consolidation path can target each superseded turn
    # directly (the E26 suite is oracle-pre-extracted, mirroring E20's repair).
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
        obsolete_fact["superseded_by"] = current_turn

    if seed == 1 and hist_text != committed:
        raise RuntimeError(
            f"{group_id}/{variant}: committed history.txt is out of sync with "
            "the in-code spec; rerun fixture materialisation to refresh")

    import sys
    return DiscriminativeTask(
        task_id=f"{group_id}_{variant}",
        title=(f"discriminative {group_id} variant {variant} "
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
        policy_facts=[_policy_from_dict(r) for r in vspec.current],
        obsolete_policy_facts=[_policy_from_dict(r) for r in vspec.obsolete],
        distractor_facts=_facts_from_dicts(vspec.distractors),
        corrections=[PolicyCorrection(o, c) for o, c in vspec.corrections],
        gold_patch_path=gold_path(group_id, variant),
        obsolete_gold_patch_path=obsolete_gold_path(group_id, variant),
        seed=seed,
        history_turns=history_turns,
        metadata={
            "ingestion": "oracle_pre_extracted",
            "ingestion_note": (
                "discriminative fixtures; history generated from the spec, "
                "hidden tests and gold/obsolete patches are variant-specific "
                "and the per-group shared workspace/prompt are byte-identical"),
            "group": group_id,
            "variant": variant,
            "decision_summary": vspec.decision_summary,
            "discriminative": True,
            "history_sha": history_sha(group_id, variant),
            "prompt_sha": prompt_sha(group_id),
            "workspace_sha": workspace_sha(group_id),
            "visible_workspace_files": sorted(
                str(p.relative_to(workspace_dir(group_id)))
                for p in workspace_dir(group_id).rglob("*") if p.is_file()),
        },
    )


def as_coding_task(task: DiscriminativeTask) -> CodingTask:
    """Convert a ``DiscriminativeTask`` into the harness ``CodingTask`` type."""
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
               for f in task.policy_facts]
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
# workspace or the task prompt (the policy mapping is history-only).
_FORBIDDEN_WORKSPACE_TERMS = {
    "pending_releases",
    "journal",
    "MISCELLANEOUS",
    "fallback display",
    "wait_confirmation",
    "confirmation",
    "confirmed",
}


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


def fact_token_counts(task: DiscriminativeTask) -> Dict[str, int]:
    counts = {}
    for f in task.all_policy_facts:
        counts[f.fact_id] = _word_tokens(f.text)
    for f in task.distractor_facts:
        counts[f.fact_id] = _word_tokens(f.text)
    return counts


def validate_group_variant(task: DiscriminativeTask,
                           other_variant_task: DiscriminativeTask = None) -> Dict:
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
    vspec = _variant_spec(gid, task.variant)

    if other_variant_task is None:
        other = "B" if task.variant == "A" else "A"
        other_variant_task = build_variant(gid, other, seed=1)

    # --- group identity (workspace/prompt byte-identical across variants) ---
    check("shared_workspace_identity",
          task.metadata["workspace_sha"] == other_variant_task.metadata["workspace_sha"],
          f"{task.metadata['workspace_sha']} vs "
          f"{other_variant_task.metadata['workspace_sha']}")
    check("shared_prompt_identity",
          task.metadata["prompt_sha"] == other_variant_task.metadata["prompt_sha"],
          f"{task.metadata['prompt_sha']} vs "
          f"{other_variant_task.metadata['prompt_sha']}")

    # --- variant differences (history/hidden/gold/obsolete all differ) ---
    check("history_differs_across_variants",
          task.metadata["history_sha"] != other_variant_task.metadata["history_sha"],
          f"{task.metadata['history_sha']} vs "
          f"{other_variant_task.metadata['history_sha']}")
    check("hidden_differs_across_variants",
          hidden_test_sha(gid, task.variant) != hidden_test_sha(gid, other_variant_task.variant))
    check("gold_differs_across_variants",
          gold_patch_sha(gid, task.variant) != gold_patch_sha(gid, other_variant_task.variant))
    check("obsolete_differs_across_variants",
          obsolete_gold_patch_sha(gid, task.variant)
          != obsolete_gold_patch_sha(gid, other_variant_task.variant))

    # --- history depth & placement ---
    turns = task.history_turns
    current_turns = [f.turn_index for f in task.policy_facts]
    obsolete_turns = [f.turn_index for f in task.obsolete_policy_facts]
    check("four_current_policy_facts", len(task.policy_facts) == 4)
    check("four_obsolete_policy_facts", len(task.obsolete_policy_facts) == 4)
    check("four_distractors", len(task.distractor_facts) == 4)
    check("four_corrections", len(task.corrections) == 4)
    check("current_fact_placement_420_540",
          all(420 <= t <= 540 for t in current_turns),
          f"current turns={sorted(current_turns)}")
    check("obsolete_fact_placement_70_180",
          all(70 <= t <= 180 for t in obsolete_turns),
          f"obsolete turns={sorted(obsolete_turns)}")
    check("obsolete_facts_more_than_300_turns_old",
          all((turns - t) > 300 for t in obsolete_turns),
          f"ages={sorted((turns - t) for t in obsolete_turns)}")
    check("history_is_600_turns", len(task.history) == HISTORY_TURNS_DEFAULT)
    check("facts_spread_across_history",
          max(current_turns) - min(obsolete_turns) > 250,
          f"span {min(obsolete_turns)}..{max(current_turns)}")

    # --- fact sizing (18-28 shared-word tokens) ---
    counts = fact_token_counts(task)
    bad = {fid: c for fid, c in counts.items()
           if not (MIN_FACT_TOKENS <= c <= MAX_FACT_TOKENS)}
    check("fact_sizes_18_to_28", not bad, f"bad={bad}")

    # --- structured metadata presence ---
    by_id = {fact["fact_id"]: fact
             for entry in task.history
             for fact in entry.get("facts", [])}
    meta_ok = True
    for c in task.corrections:
        cur = by_id.get(c.current_fact_id)
        obs = by_id.get(c.obsolete_fact_id)
        if cur is None or obs is None:
            meta_ok = False
            continue
        cur_ok = (
            cur.get("category") == "project_context"
            and cur.get("is_correction_target") is True
            and cur.get("is_current_correction") is True
            and cur.get("supersedes_turn") == obs.get("source_turn_id")
            and cur.get("superseded_prior_fact_id") == c.obsolete_fact_id
        )
        obs_ok = (
            obs.get("category") == "project_context"
            and obs.get("superseded_by") == cur.get("source_turn_id")
        )
        meta_ok = meta_ok and cur_ok and obs_ok
    check("correction_metadata_structured", meta_ok)

    obs_ids = {f.fact_id for f in task.obsolete_policy_facts}
    cur_ids = {f.fact_id for f in task.policy_facts}
    check("correction_pairs_consistent",
          all(c.obsolete_fact_id in obs_ids and c.current_fact_id in cur_ids
              for c in task.corrections))
    check("fact_ids_unique",
          len(obs_ids | cur_ids) == len(obs_ids) + len(cur_ids))

    # --- no policy mapping in visible workspace/prompt ---
    visible = _workspace_prompt_text(gid)
    hits = sorted(t for t in _FORBIDDEN_WORKSPACE_TERMS if t in visible)
    check("no_policy_mapping_in_visible_workspace", not hits, f"hits={hits}")

    # --- distractor categories are transient ---
    transient_ok = all(
        by_id.get(f.fact_id, {}).get("category") == "transient"
        or f.fact_id not in by_id or True
        for f in task.distractor_facts
    )
    # (distractor facts are carried in `distractor_facts`; structured history
    # includes them as `transient` via `_CATEGORY_BY_KIND["distractor"]`.)
    check("distractor_category_transient",
          all(by_id.get(f.fact_id, {}).get("category") == "transient"
              for f in task.distractor_facts))

    # --- harness-level offline gates (gold passes / obsolete fails / base fails),
    #     purely deterministic (StaticCoder, no LLM) ---
    if not transient_ok:
        check("base_workspace_fails", False, "distractor category mismatch")

    base_fail, gold_pass, obsolete_fail = _run_offline_hidden_checks(task)
    check("base_workspace_fails_hidden", base_fail)
    check("gold_patch_passes_hidden", gold_pass)
    check("obsolete_patch_fails_hidden", obsolete_fail)

    return {
        "group_id": gid,
        "variant": task.variant,
        "checks": checks,
        "failing": [c["check"] for c in checks if not c["passed"]],
        "passed": all(c["passed"] for c in checks),
    }


def _run_offline_hidden_checks(task: DiscriminativeTask) -> Tuple[bool, bool, bool]:
    """Deterministic base/gold/obsolete hidden-test outcomes.

    Applies the empty patch (base), the gold patch, and the obsolete patch in
    three throwaway git repos and runs each hidden test. No LLM/embedding call.
    """
    from experiments import coding_benchmark as cb

    def run_one(patch: Optional[str]) -> bool:
        import shutil
        import subprocess
        import sys as _sys
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
# self-test (python -m data.discriminative_coding_suite)
# ---------------------------------------------------------------------------

def self_test(require_fixtures: bool = False) -> None:
    assert len(_GROUPS) == 3, "expected exactly 3 discriminative groups"
    for gid in list_group_ids():
        assert _GROUP_BY_ID[gid].prompt.strip(), "prompt required"
        for variant in list_variants(gid):
            vs = _variant_spec(gid, variant)
            assert variant in ("A", "B")
            assert (len(vs.obsolete) == len(vs.current) == 4), \
                f"{gid}/{variant} needs 4 obsolete + 4 current policy facts"
            assert len(vs.distractors) == 4, \
                f"{gid}/{variant} needs 4 distractors"
            assert len(vs.corrections) == 4, \
                f"{gid}/{variant} needs 4 corrections"
            task = build_variant(gid, variant, seed=1)
            assert len(task.history) == HISTORY_TURNS_DEFAULT
            # deterministic regeneration matches committed history (seed=1)
            fresh = build_history_text(gid, variant)
            assert "\n".join(fresh) + "\n" == lineage_read(gid, variant), \
                f"{gid}/{variant} history drifted"
            if require_fixtures:
                assert task.gold_patch_path.is_file()
                assert task.obsolete_gold_patch_path.is_file()
                assert task.hidden_test_path.is_file()
                for f in task.all_policy_facts + task.distractor_facts:
                    assert MIN_FACT_TOKENS <= _word_tokens(f.text) <= MAX_FACT_TOKENS,\
                        f"{gid}/{variant}:{f.fact_id} fact size out of range"
                    assert f.probe.lower() in f.text.lower(), \
                        f"{gid}/{variant}:{f.fact_id} probe not in text"
            else:
                for f in task.all_policy_facts:
                    assert MIN_FACT_TOKENS <= _word_tokens(f.text) <= MAX_FACT_TOKENS,\
                        f"{gid}/{variant}:{f.fact_id} fact size out of range"
    # byte-identical workspace/prompt across variants
    for gid in list_group_ids():
        a = build_variant(gid, "A", seed=1)
        b = build_variant(gid, "B", seed=1)
        assert a.metadata["workspace_sha"] == b.metadata["workspace_sha"]
        assert a.metadata["prompt_sha"] == b.metadata["prompt_sha"]
        assert no_history_prompt(gid, "A") == no_history_prompt(gid, "B")
    print("discriminative_coding_suite self-test OK")


def lineage_read(gid: str, variant: str) -> str:
    return history_path(gid, variant).read_text(encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Materialise/verify E26 discriminative fixtures")
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