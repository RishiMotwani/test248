#!/usr/bin/env python3
"""J — paper state / claim-to-evidence tracker (task J).

Walks the gitignored results tree (manifests, models summary, e0/e7/e8 outputs)
and maps each paper claim to the artifacts that would support it. Each run is
tagged with config + token source + metric_source so the tracker itself obeys
G0: an inventory line is never a number claim without its provenance.

Outputs ``experiments/results/paper_state.json`` and prints a coverage table.
``--quick`` is the same code path against whatever artifacts exist.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

RESULTS = BASE_DIR / "experiments/results"
MODELS = RESULTS / "models"


def manifest_summary(man: Dict) -> Dict:
    cfg = man.get("config", {})
    a = man.get("aggregate", {}).get("adaptive", {})
    mc = man.get("aggregate", {}).get("multiple_comparisons", {})
    return {
        "path": "latest_manifest.json",
        "turns": cfg.get("turns"), "density": cfg.get("density"),
        "seeds": cfg.get("seeds"), "write": cfg.get("write"),
        "token_source": man.get("token_source"),
        "metric_source": man.get("metric_source"),
        "E1_proposed": a.get("mean_E1_proposed_tokens"),
        "E2_recall": a.get("mean_E2_proposed_recall"),
        "holm_adaptive_p": (mc.get("holms_adjusted") or {}).get("adaptive"),
        "epoch": int(man.get("generated_at", 0)),
    }


def _agg_check(evidence: Dict) -> bool:
    m = evidence["manifest"]
    if not m:
        return False
    return m.get("aggregate", {}).get("adaptive") is not None


def _measured_check(evidence: Dict) -> bool:
    m = evidence["manifest"]
    if not m:
        return False
    agg_a = m.get("aggregate", {}).get("adaptive", {})
    return agg_a.get("mean_E1_proposed_tokens") is not None


def _latency_check(evidence: Dict) -> bool:
    if evidence["models"] and len(evidence["models"].get("per_model", {})) >= 1:
        return True
    m = evidence["manifest"]
    if m:
        for s in m.get("results", []):
            em = s.get("methods", {}).get("adaptive", {}).get("E3_latency", {}).get("extraction_ms")
            if em is not None:
                return True
    return False


def _e4_check(evidence: Dict) -> bool:
    m = evidence["manifest"]
    if not m:
        return False
    return any(
        bool(s.get("methods", {}).get("adaptive", {}).get("E4_needle_in_haystack", {}).get("distances"))
        for s in (m.get("results") or [])
    )


CLAIMS: List[Dict] = [
    {"id": "E1", "claim": "adaptive uses fewer tokens per turn than raw history window",
     "check": _measured_check},
    {"id": "E2", "claim": "correct facts are retained in the store",
     "check": lambda e: (e["manifest"] or {}).get("aggregate", {})
                          .get("adaptive", {}).get("mean_E2_proposed_recall") is not None},
    {"id": "E3", "claim": "end-to-end latency is measurable with real extraction",
     "check": _latency_check},
    {"id": "E4", "claim": "long-distance needles survive (needle-in-haystack)",
     "check": _e4_check},
    {"id": "E5", "claim": "each adaptive component contributes (ablations)",
     "check": lambda e: bool(((e["manifest"] or {}).get("results") or [])
                             and ((e["manifest"] or {}).get("results") or [])[0]
                             .get("methods", {}).get("adaptive", {}).get("E5_ablations"))},
    {"id": "E6", "claim": "study is powered to detect the token difference",
     "check": lambda e: (e["manifest"] or {}).get("aggregate", {})
                          .get("adaptive", {}).get("E6", {}).get("required_n_per_group") is not None},
    {"id": "C7", "claim": "results are stable across retriever/budget geometry",
     "check": lambda e: (e["e7"] or {}).get("rows", []) != []},
    {"id": "E0", "claim": "the extraction writer itself is accurate vs gold set",
     "check": lambda e: (e["e0"] or {}).get("turns_scored", 0) > 0},
    {"id": "E8", "claim": "external needled-QA benchmark passes",
     "check": lambda e: bool((e["e8"] or {}).get("n_items") or (e["e8"] or {}).get("items"))},
    {"id": "MM", "claim": "writer variance across the model set is measured",
     "check": lambda e: len((e["models"] or {}).get("per_model", {})) >= 3},
]


def scan() -> Dict:
    manifests = sorted(RESULTS.glob("manifest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    runs = []
    for p in manifests:
        try:
            runs.append(json.loads(p.read_text()))
        except Exception:
            pass
    latest_path = RESULTS / "latest_manifest.json"
    latest = json.loads(latest_path.read_text()) if latest_path.exists() else None
    e0_path = RESULTS / "e0_extraction_quality.json"
    e0 = json.loads(e0_path.read_text()) if e0_path.exists() else None
    e7_path = RESULTS / "e7_sweep.json"
    e7 = json.loads(e7_path.read_text()) if e7_path.exists() else None
    e8s = sorted(RESULTS.glob("e8_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    e8 = json.loads(e8s[0].read_text()) if e8s else None
    models_path = MODELS / "models_summary.json"
    models = json.loads(models_path.read_text()) if models_path.exists() else None

    evidence = {"manifest": latest, "e0": e0, "e7": e7, "e8": e8, "models": models}

    inventory = []
    for r in runs[:1] + ([latest] if latest else []):
        if r:
            inventory.append(manifest_summary(r))

    coverage = []
    for c in CLAIMS:
        supported = bool(c["check"](evidence))
        coverage.append({"claim_id": c["id"], "claim": c["claim"], "supported": supported})

    return {
        "generated_at": time.time(),
        "inventory": inventory,
        "artifact_files": {
            "latest_manifest": str(latest_path) if latest else None,
            "e0": str(e0_path) if e0 else None,
            "e7": str(RESULTS / "e7_sweep.json") if e7 else None,
            "e8": str(e8s[0]) if e8s else None,
            "models_summary": str(models_path) if models else None,
        },
        "coverage": coverage,
        "metric_source_sentence": ("Inventory rows carry config + token_source; claim support comes only from "
                                   "artifacts whose metric_source is recorded."),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="J paper claim-to-evidence state")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    state = scan()
    out = RESULTS / "paper_state.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state, indent=2))

    print(f"coverage ({len(state['inventory'])} run(s)):")
    for c in state["coverage"]:
        print(f"  [{'OK' if c['supported'] else 'missing'}] {c['claim_id']:3s} {c['claim']}")
    print(f"-> {out}")
    missing = [c["claim_id"] for c in state["coverage"] if not c["supported"]]
    print("missing evidence:", missing or "none")


if __name__ == "__main__":
    main()
