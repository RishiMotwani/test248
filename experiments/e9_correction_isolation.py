"""E9 — correction isolation experiment (task E, requirement TEST-05).

A from-scratch, self-contained *probe* of the correction-supersession behavior,
independent of the batch paper run: it builds its own tiny stream of
correction/negation turns, replays ONLY the adaptive pipeline over it, and
reports how the store handled each supersession.

Isolation means the scenario is hand-rolled (not the shared generator) so a
regression is attributable to the compression/dedupe path, not to a generator
quirk. The adaptive method is the only one run — there are no baseline
comparisons here; E1-E6 in the paper manifest remain the numbers that count.

Output: ``experiments/results/e9_correction_isolation.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from memory_optimizer.compression import MemoryCompressor, _is_supersession  # noqa: E402
from memory_optimizer.decay import CategoryDecayEngine  # noqa: E402
from memory_optimizer.retrieval import MemoryRetriever, _token_overlap  # noqa: E402
from memory_optimizer.scoring import ImportanceScorer  # noqa: E402
from experiments.paper import load_settings, replay_adaptive  # noqa: E402


def build_isolation_stream() -> list:
    """A deterministic stream of planted facts, corrections, and negations.

    Turn layout (each turn is one planted fact carried into replay):
      1  plant   "key_14 is enabled"
      5  plant   "key_22 is enabled"
      9  negate  "NOT ... key_22 is enabled"            (negation)
     13 correct "key_14 ... is no longer the case"      (correction)
     17 plant   "key_27 is enabled"
     21 negate  "NOT ... key_27 is enabled"             (negation)
     25 correct the key_14 correction is superseded by a re-assert (fresh)
    Each turn's user message echoes the fact so write-time salience is non-zero.
    """
    turns = [
        {"turn_id": 1, "user": "key_14 is enabled",
         "facts": [{"fact": "Project feature flag configuration key_14 is enabled",
                    "category": "project_context", "confidence": 0.5, "is_correction_target": False}]},
        {"turn_id": 5, "user": "key_22 is enabled",
         "facts": [{"fact": "Project feature flag configuration key_22 is enabled",
                    "category": "project_context", "confidence": 0.5, "is_correction_target": False}]},
        {"turn_id": 9, "user": "key_22 is now banned",
         "facts": [{"fact": "Correction to requirement: NOT project feature flag configuration key_22 is enabled",
                    "category": "project_context", "confidence": 0.7, "is_negation": True}]},
        {"turn_id": 13, "user": "key_14 is no longer in use",
         "facts": [{"fact": "Revised requirement: Project feature flag configuration key_14 is enabled is no longer the case",
                    "category": "project_context", "confidence": 0.7, "is_correction_target": True,
                    "supersedes_turn": 1}]},
        {"turn_id": 17, "user": "key_27 is enabled",
         "facts": [{"fact": "Project feature flag configuration key_27 is enabled",
                    "category": "project_context", "confidence": 0.5, "is_correction_target": False}]},
        {"turn_id": 21, "user": "key_27 was a mistake",
         "facts": [{"fact": "Correction to requirement: NOT project feature flag configuration key_27 is enabled",
                    "category": "project_context", "confidence": 0.5, "is_negation": True}]},
    ]
    for t in turns:
        t["tokens"] = max(1, len(t["user"].split()))
    return turns


def run_isolation(turns: list) -> dict:
    settings = load_settings()
    scorer = ImportanceScorer(weights=settings["scoring_weights"])
    decay = CategoryDecayEngine(lambdas=settings["decay_lambdas"],
                                pruning_threshold=float(settings["pruning"]["threshold"]))
    retriever = MemoryRetriever(top_k=int(settings["top_k"]),
                                sim_threshold=float(settings.get("similarity_threshold", 0.35)))
    compressor = MemoryCompressor()
    rep = replay_adaptive(turns, settings, scorer, decay, retriever, compressor,
                          fact_tokens=lambda t: max(1, len(t.split())))

    store = rep["memories"]
    # Ground truth we expect the store to reflect (superseded originals gone).
    expected_neg = {"Correction to requirement: NOT project feature flag "
                    "configuration key_22 is enabled",
                    "Correction to requirement: NOT project feature flag "
                    "configuration key_27 is enabled"}
    expected_corr = {"Revised requirement: Project feature flag configuration "
                     "key_14 is enabled is no longer the case"}
    stored_facts = {m["fact"] for m in store}

    neg_ok = expected_neg.issubset(stored_facts)
    corr_ok = expected_corr.issubset(stored_facts)
    superseded_gone = all(
        "key_14 is enabled" not in f and "key_22 is enabled" not in f and "key_27 is enabled" not in f
        for f in stored_facts if "Revised" not in f and "Correction" not in f and "NOT" not in f)

    entries = []
    for m in sorted(store, key=lambda x: x["source_turn_id"]):
        entries.append({
            "source_turn_id": m.get("source_turn_id"),
            "fact": m["fact"],
            "superseded_prior_fact": m.get("superseded_prior_fact"),
            "confidence": round(m.get("confidence", 0), 3),
        })

    return {
        "scenario": "6-turn hand-rolled correction/negation isolation",
        "store_entries": entries,
        "negations_recorded": (len(expected_neg), len(expected_neg & stored_facts)),
        "correction_recorded": (1, 1 if corr_ok else 0),
        "superseded_originals_purged": superseded_gone,
        "isolation_pass": bool(neg_ok and corr_ok and superseded_gone),
        "stale_same_category_count": sum(
            1 for m in store
            if m["category"] == "project_context"
            and m["fact"] not in expected_neg | expected_corr
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="E9 correction isolation probe")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    turns = build_isolation_stream()
    payload = run_isolation(turns)
    payload["run_at"] = time.time()
    payload["quick"] = args.quick

    out = BASE_DIR / "experiments/results" / "e9_correction_isolation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))

    print(f"E9 correction isolation ({'quick' if args.quick else 'full'}):")
    print(f"  negations recorded: {payload['negations_recorded'][0]}/{payload['negations_recorded'][1]}")
    print(f"  correction recorded: {payload['correction_recorded'][0]}/{payload['correction_recorded'][1]}")
    print(f"  superseded originals purged: {payload['superseded_originals_purged']}")
    print(f"  isolation_pass: {payload['isolation_pass']}")
    for e in payload["store_entries"]:
        print(f"    turn {e['source_turn_id']:>2} :: {e['fact'][:70]}" +
              (f"  [prior: {e['superseded_prior_fact']}]" if e["superseded_prior_fact"] else ""))
    print(f"-> {out}")
    sys.exit(0 if payload["isolation_pass"] else 1)


if __name__ == "__main__":
    main()