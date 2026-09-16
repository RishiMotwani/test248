#!/usr/bin/env python3
"""H — multi-model experiment runner (task H).

Runs the paper pipeline (D10) with real LLM extraction across the agreed model
set (brain.md D's "multi-model set": {llama3.1:8b, qwen2.5:7b, mistral:7b-instruct})
so extraction-writer variance across models is measured, not assumed. Each model
gets its own manifest under ``experiments/results/models/<slug>/`` so the
canonical ``latest_manifest.json`` is never clobbered (G0/F0 discipline).

``--quick`` is a wiring gate: one model, 6 turns, 1 seed — still 6 real LLM
extraction calls, so it validates the full extractor path cheaply.

Run for real with, e.g.::

    ./venv/bin/python experiments/run_multi_model.py --seeds 3 --turns 150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from experiments.paper import run_paper, _sanitize  # noqa: E402

MODEL_SET = ["llama3.1:8b", "qwen2.5:7b", "mistral:7b-instruct"]


def model_slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def run_multi_model(turns: int, density: float, seeds: List[int], methods: List[str],
                    models: List[str], embedding_model: str = "",
                    label: str = "") -> Dict:
    run_dir = BASE_DIR / "experiments/results/models"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "pipeline": "run_multi_model",
        "config": {"turns": turns, "density": density, "seeds": seeds,
                   "methods": methods, "models": models,
                   "embedding_model": embedding_model, "label": label},
        "per_model": {},
        "note": ("Each model ran the same seeded paper pipeline with real LLM "
                 "extraction (--write extract, --measured). Writer held identical "
                 "across methods within a model; cross-model differences are writer variance."),
    }

    for model in models:
        slug = model_slug(model)
        print("=" * 72)
        print(f" MODEL {model}  turns={turns} density={density}% seeds={len(seeds)}")
        print("=" * 72)
        payload = run_paper(turns=turns, density=density, seeds=seeds, methods=methods,
                            write="extract", measured=True, model=model,
                            ollama_endpoint="http://localhost:11434",
                            embedding_model=embedding_model, label=label,
                            conflict_density=0.1, negation_density=0.1)
        model_dir = run_dir / slug
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "manifest.json").write_text(json.dumps(_sanitize(payload), indent=2, default=str))

        e3_ms = []
        for s in payload["results"]:
            m = s["methods"]
            em = m.get("adaptive", {}).get("E3_latency", {}).get("extraction_ms")
            if em:
                e3_ms.append(em)
        row = {
            "model": model,
            "seeds": len(seeds),
            "mean_E1_proposed_tokens": payload["aggregate"]["adaptive"]["mean_E1_proposed_tokens"],
            "mean_E2_recall": payload["aggregate"]["adaptive"]["mean_E2_proposed_recall"],
            "mean_extraction_ms": round(sum(e3_ms) / len(e3_ms), 2) if e3_ms else None,
            "e1_ci95": payload["aggregate"]["adaptive"]["E1_proposed_pooled_ci95"],
        }
        summary["per_model"][slug] = row
        print(f"  adaptive: E1 {row['mean_E1_proposed_tokens']} tok, "
              f"E2 {row['mean_E2_recall']}, extraction {row['mean_extraction_ms']} ms")

    (run_dir / "models_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n-> {run_dir/'models_summary.json'}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="H multi-model corpus runner")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--turns", type=int, default=150)
    ap.add_argument("--density", type=float, default=20.0)
    ap.add_argument("--models", default=",".join(MODEL_SET))
    ap.add_argument("--methods", default="adaptive,vanilla_rag,sliding_window")
    ap.add_argument("--embeddings", action="store_true", help="enable nomic-embed-text retrieval/dedupe")
    args = ap.parse_args()

    if args.quick:
        args.turns, args.seeds, args.models = 6, 1, "llama3.1:8b"

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [42 + i for i in range(args.seeds)]
    run_multi_model(args.turns, args.density, seeds, methods, models,
                    embedding_model="nomic-embed-text" if args.embeddings else "")


if __name__ == "__main__":
    main()