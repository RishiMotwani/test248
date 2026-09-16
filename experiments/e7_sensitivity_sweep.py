"""C7 — sensitivity sweep (task C).

Sweeps the paper pipeline (D10) over retriever/window hyperparameters and
scenario geometry, one dimension at a time from a fixed baseline point so
interactions are readable in the ordering (AGENTS.md discipline: never report a
number without its operative configuration). Each grid cell is a seeded oracle
replay (lexical, estimated word-count tokens unless --measured) so the whole
grid stays cheap and deterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from experiments.paper import load_settings, run_seed  # noqa: E402

BASELINE_POINT = {"turns": 150, "density": 20.0, "budget": 4096, "top_k": 5}

DIMENSIONS = {
    "turns": [60, 100, 200, 300],
    "density": [5.0, 10.0, 20.0, 50.0],
    "budget": [1024, 2048, 4096, 8192],
    "top_k": [1, 3, 5, 10],
}


def grid_points(dims: Dict[str, List]) -> List[Dict]:
    points = []
    for dim, values in dims.items():
        for v in values:
            pt = dict(BASELINE_POINT)
            pt[dim] = v
            points.append(pt)
    return points


def _method_summary(seed_res: Dict, method: str) -> Dict:
    e = seed_res["methods"][method]
    return {
        "E1_proposed_tokens": round(e["E1_token_efficiency"]["proposed_mean_tokens"], 2),
        "E1_baseline_tokens": round(e["E1_token_efficiency"]["baseline_mean_tokens"], 2),
        "E2_recall": e["E2_memory_accuracy"]["proposed_positive_recall"],
    }


def run_sweep(points: List[Dict], methods: List[str], seeds: List[int],
              measured: bool = False) -> Dict:
    rows = []
    for pt in points:
        settings = load_settings()
        settings["max_context_tokens"] = int(pt["budget"])
        settings["top_k"] = int(pt["top_k"])
        seed_aggs = {m: {"e1": [], "e2": []} for m in methods}
        for seed in seeds:
            sr = run_seed(seed, int(pt["turns"]), float(pt["density"]), settings,
                          methods, write="oracle", measured=measured,
                          model=settings["system"].get("llm_model", "llama3.1:8b"),
                          embedding_model="")
            for m in methods:
                sm = _method_summary(sr, m)
                seed_aggs[m]["e1"].append(sm["E1_proposed_tokens"])
                seed_aggs[m]["e2"].append(sm["E2_recall"])
        methods_out = {}
        for m in methods:
            methods_out[m] = {
                "mean_E1_proposed_tokens": round(sum(seed_aggs[m]["e1"]) / len(seed_aggs[m]["e1"]), 2),
                "mean_E2_recall": round(sum(seed_aggs[m]["e2"]) / len(seed_aggs[m]["e2"]), 3),
            }
        rows.append({"point": pt, "methods": methods_out})

    one_at_a_time = []
    for dim, values in DIMENSIONS.items():
        entry = {"dimension": dim, "points": []}
        for v in values:
            row = next(r for r in rows if r["point"][dim] == v and all(
                r["point"][k] == BASELINE_POINT[k] for k in DIMENSIONS if k != dim))
            entry["points"].append({"value": v, "methods": row["methods"]})
        one_at_a_time.append(entry)

    return {
        "pipeline": "e7",
        "baseline_point": BASELINE_POINT,
        "methods": methods,
        "seeds": seeds,
        "rows": rows,
        "one_at_a_time": one_at_a_time,
        "metric_source": "offline seeded oracle replay (writer held fixed), lexical retrieval; "
                         + ("LLM token counts" if measured else "estimated(word-count) tokens"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="C7 sensitivity sweep")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--methods", default="adaptive,vanilla_rag,sliding_window")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--measured", action="store_true")
    args = ap.parse_args()

    if args.quick:
        DIMENSIONS.update({"turns": [100, 200], "density": [10.0, 30.0], "budget": [4096], "top_k": [5]})

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [42 + i for i in range(args.seeds)]
    points = grid_points(DIMENSIONS)
    print(f"C7 sweep: {len(points)} grid cells x {len(seeds)} seeds, methods={methods}")

    payload = run_sweep(points, methods, seeds, measured=args.measured)
    out = BASE_DIR / "experiments/results" / "e7_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"-> {out}")
    print("one-at-a-time (value: method=recall/tokens):")
    for o in payload["one_at_a_time"]:
        line = [f"{p['value']}: " + ", ".join(f"{m}={p['methods'][m]['mean_E2_recall']:.2f}"
                                              f"/{p['methods'][m]['mean_E1_proposed_tokens']:.0f}"
                                              for m in methods) for p in o["points"]]
        print(f"  {o['dimension']:10s} " + " | ".join(line))


if __name__ == "__main__":
    main()
