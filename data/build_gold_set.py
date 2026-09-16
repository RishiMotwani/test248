"""Deterministic generator for the E0 extraction gold set (200 turns, task D).

Regenerates ``data/gold_labels/extraction_gold_200.json`` from seeded templates
so the set is reproducible and versioned. Schema matches the existing gold file:
    {"turn_id": int, "user_turn": str,
     "expected_facts": [{"fact": str, "category": str, "confidence": float}]}

Categories follow the system taxonomy (personal / project_context /
technical_preference / transient). Trap turns sound transient but encode a fact
that must survive (the E2/E4 trap machinery); correction turns state a rule and
then reverse it, with the expected fact being the corrected rule. `notes` on a
turn records which trap/correction property it exercises, so e0 can report
per-property extraction quality without re-deriving it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "gold_labels" / "extraction_gold_200.json"
SEED = 7
TARGET = 200


def range_pool(n: int) -> list:
    """Deterministic, non-repeating pool of engineering-flavored values."""
    rng = random.Random(SEED + n)
    return rng.sample(range(1, 10**6), n)


def build() -> list:
    random.seed(SEED)
    entries = []
    ports = range_pool(TARGET)
    keys = range_pool(TARGET)
    jobs = range_pool(TARGET)
    names = ["Dana", "Priya", "Marco", "Elena", "Tom", "Aisha", "Ken", "Yuki"]
    services = ["orders-api", "auth-svc", "billing", "search", "ingest", "gateway", "render", "sync"]

    def personal(idx):
        n = names[idx % len(names)]
        tid = idx + 1
        kind = idx % 5
        if kind == 0:
            return (f"Reminder for the team: {n} prefers asynchronous updates over sync standups.",
                    [{"fact": f"{n} prefers asynchronous updates over sync standups",
                      "category": "personal", "confidence": 0.92}])
        if kind == 1:
            return (f"Don't forget, {n}'s birthday is next week and we usually rotate treats.",
                    [{"fact": f"{n}'s birthday is next week",
                      "category": "personal", "confidence": 0.85}])
        if kind == 2:
            return (f"By the way, {n} is on call for the whole of next month.",
                    [{"fact": f"{n} is on call for the whole of next month",
                      "category": "personal", "confidence": 0.9}])
        if kind == 3:
            return (f"{n} mentioned they only check PRs after lunch, so schedule merges accordingly.",
                    [{"fact": f"{n} only checks PRs after lunch",
                      "category": "personal", "confidence": 0.88}])
        return (f"FYI: {n} uses spaced repetition and prefers written context over verbal handoff.",
                [{"fact": f"{n} uses spaced repetition",
                  "category": "personal", "confidence": 0.8},
                 {"fact": f"{n} prefers written context over verbal handoff",
                  "category": "personal", "confidence": 0.8}])

    def project(idx):
        svc = services[idx % len(services)]
        kind = idx % 5
        if kind == 0:
            return (f"Project note: {svc} is being migrated to the regional cluster next sprint.",
                    [{"fact": f"{svc} is being migrated to the regional cluster next sprint",
                      "category": "project_context", "confidence": 0.95}])
        if kind == 1:
            return (f"Decision: {svc} owns its feature flags starting Q3, no override from the platform team.",
                    [{"fact": f"{svc} owns its feature flags starting Q3",
                      "category": "project_context", "confidence": 0.9},
                     {"fact": "no override of the feature flags from the platform team",
                      "category": "project_context", "confidence": 0.8}])
        if kind == 2:
            return (f"Architecture decision under discussion: {svc} will split its queue into two partitions.",
                    [{"fact": f"{svc} will split its queue into two partitions",
                      "category": "project_context", "confidence": 0.85},
                     {"fact": "splitting {svc} queue is an architecture decision under discussion",
                      "category": "project_context", "confidence": 0.7}])
        if kind == 3:
            return (f"The {svc} rollout is gated on the load test passing at 8k RPS.",
                    [{"fact": f"{svc} rollout is gated on load test passing at 8k RPS",
                      "category": "project_context", "confidence": 0.9}])
        return (f"Standards call: all new services must run under {svc}-style canary deploy.",
                [{"fact": "all new services must run under canary deploy",
                  "category": "project_context", "confidence": 0.9}])

    def tech(idx):
        kind = idx % 6
        if kind == 0:
            return (f"Requirement: the {services[idx % len(services)]} database must use PostgreSQL {15 + idx % 3}.",
                    [{"fact": f"{services[idx % len(services)]} database must use PostgreSQL {15 + idx % 3}",
                      "category": "technical_preference", "confidence": 0.95}])
        if kind == 1:
            return (f"Important: distribution keys must not exceed {1000 + idx * 17 % 5000} bytes in size.",
                    [{"fact": f"distribution keys must not exceed {1000 + idx * 17 % 5000} bytes",
                      "category": "technical_preference", "confidence": 0.9}])
        if kind == 2:
            return (f"Setup fact: the {services[idx % len(services)]} container uses the slim image, never the full one.",
                    [{"fact": f"{services[idx % len(services)]} container uses the slim image",
                      "category": "technical_preference", "confidence": 0.88}])
        if kind == 3:
            return (f"Note: keep the read replicas on region-local storage to avoid cross-region charges.",
                    [{"fact": "read replicas use region-local storage to avoid cross-region charges",
                      "category": "technical_preference", "confidence": 0.85}])
        if kind == 4:
            return (f"Config: {services[idx % len(services)]} needs its connect timeout raised to 30s after the incident.",
                    [{"fact": f"{services[idx % len(services)]} connect timeout raised to 30s",
                      "category": "technical_preference", "confidence": 0.9}])
        return (f"Remember: never pin patch versions in the {services[idx % len(services)]} pipeline, only feature ones.",
                [{"fact": f"never pin patch versions in the {services[idx % len(services)]} pipeline",
                  "category": "technical_preference", "confidence": 0.9}])

    def transient(idx):
        kind = idx % 3
        if kind == 0:
            return (f"Just a heads up, the coffee machine on floor 4 is out this morning.",
                    [{"fact": "coffee machine on floor 4 is out this morning",
                      "category": "transient", "confidence": 0.7}])
        if kind == 1:
            return (f"FYI the ETA on the {services[idx % len(services)]} redeploy slipped to 4pm.",
                    [{"fact": f"{services[idx % len(services)]} redeploy slipped to 4pm",
                      "category": "transient", "confidence": 0.75}])
        return (f"Quick note, the load test spinner is showing green right now.",
                [{"fact": "load test spinner is showing green right now",
                  "category": "transient", "confidence": 0.6}])

    def trap(idx):
        svc = services[idx % len(services)]
        return (f"Sounds like routine noise, but log this anyway: the {svc} failover port must stay {5432 + idx % 100}.",
                [{"fact": f"{svc} failover port must stay {5432 + idx % 100}",
                  "category": "technical_preference", "confidence": 0.9,
                  "is_trap": True}])

    def correction(idx):
        svc = services[idx % len(services)]
        if idx % 2 == 0:
            return (f"Correction from the earlier note: {svc} now runs on the staging lane, the previous statement is void.",
                    [{"fact": f"{svc} now runs on the staging lane",
                      "category": "project_context", "confidence": 0.95,
                      "correction_target": True}])
        return (f"Actually, scrap the last decision: {svc} keeps the old table name, rename is cancelled.",
                [{"fact": f"{svc} keeps the old table name",
                  "category": "project_context", "confidence": 0.9,
                  "correction_target": True}])

    builders = [personal, project, tech, transient, trap, correction]
    for i in range(TARGET):
        # deterministic cycling with spread across builders
        builder = builders[i % len(builders)]
        user, facts = builder(i)
        entries.append({"turn_id": i + 1, "user_turn": user, "expected_facts": facts})
    return entries


if __name__ == "__main__":
    entries = build()
    counts = {}
    for e in entries:
        for f in e["expected_facts"]:
            counts[f["category"]] = counts.get(f["category"], 0) + 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"wrote {len(entries)} turns -> {OUT}")
    print("expected-fact category counts:", json.dumps(counts))