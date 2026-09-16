#!/usr/bin/env python3
"""Paper experiment pipeline CLI (task B).

Deterministic, seeded, batch E1-E6 replays over the shared fact stream, with the
writer held fixed across the adaptive system and every baseline (brain.md D9).
Legacy synthetic e-modules (e1..e6_*.py) remain importable; this script is the
authoritative paper entrypoint.

Examples:
    python run_experiments.py --quick
    python run_experiments.py --seeds 5 --turns 150 --density 20.0
    python run_experiments.py --seeds 3 --write extract --measured            # real LLM
    python run_experiments.py --seeds 1 --methods adaptive,vanilla_rag
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from experiments.paper import load_settings, run_paper, write_manifest  # noqa: E402

DEFAULT_METHODS = ["adaptive", "sliding_window", "memgpt_style", "summarization_only", "vanilla_rag"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper experiment suite (E1-E6, seeded, batch).")
    parser.add_argument("--quick", action="store_true",
                        help="fast LLM-free gate: 30 turns, 1 seed, oracle facts, no measured tokens")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--turns", type=int, default=150)
    parser.add_argument("--density", type=float, default=20.0, help="signal density %%")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--write", choices=["oracle", "extract"], default="oracle",
                        help="oracle = planted facts (writer held fixed); extract = real LLM extraction")
    parser.add_argument("--measured", action="store_true",
                        help="measure tokens via Ollama prompt_eval_count (needs llama3.1:8b or --model)")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--ollama-endpoint", default="http://localhost:11434")
    parser.add_argument("--label", default="")
    parser.add_argument("--conflict-density", type=float, default=0.0,
                        help="fraction of durable facts at risk of a later correction turn (task E)")
    parser.add_argument("--negation-density", type=float, default=0.0,
                        help="fraction of durable signal turns stated as negated requirements (task E)")
    parser.add_argument("--budget", type=int, default=None,
                        help="override max_context_tokens budget for this run")
    parser.add_argument("--top-k", type=int, default=None,
                        help="override retrieval top_k for this run")
    args = parser.parse_args()

    if args.quick:
        args.turns, args.seeds, args.write, args.measured, args.embedding_model = 50, 1, "oracle", False, ""
        args.conflict_density, args.negation_density = 0.1, 0.1

    seeds = [42 + i for i in range(args.seeds)]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    overrides = {}
    if args.budget is not None:
        overrides["max_context_tokens"] = max(256, int(args.budget))
    if args.top_k is not None:
        overrides["top_k"] = max(1, int(args.top_k))

    cfg = load_settings(overrides)
    print("=" * 72)
    print(f" PAPER EXPERIMENT SUITE  turns={args.turns} density={args.density}% seeds={args.seeds}")
    print(f" methods={methods}  write={args.write}  measured={args.measured}")
    print(f" budget={cfg['max_context_tokens']} top_k={cfg['top_k']} model={args.model}")
    if args.conflict_density or args.negation_density:
        print(f" hard cases: conflict_density={args.conflict_density} negation_density={args.negation_density}")
    print("=" * 72)

    payload = run_paper(
        turns=args.turns, density=args.density, seeds=seeds, methods=methods,
        write=args.write, measured=args.measured, model=args.model,
        ollama_endpoint=args.ollama_endpoint, embedding_model=args.embedding_model,
        label=args.label, conflict_density=args.conflict_density,
        negation_density=args.negation_density, overrides=overrides,
    )

    path, run_id = write_manifest(payload)
    print(f"\n[SUCCESS] manifest {run_id} -> {path}")
    print("\naggregate (mean over seeds):")
    for m, a in payload["aggregate"].items():
        if isinstance(a, dict) and "mean_E1_proposed_tokens" in a:
            print(f"  {m:18s} E1_prop_tok={a['mean_E1_proposed_tokens']:8.1f} "
                  f"E1_base_tok={a['mean_E1_baseline_tokens']:8.1f} "
                  f"E2_prop_recall={a['mean_E2_proposed_recall']:.3f} "
                  f"E2_base_recall={a['mean_E2_baseline_recall']:.3f}")


if __name__ == "__main__":
    main()