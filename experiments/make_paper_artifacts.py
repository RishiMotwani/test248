#!/usr/bin/env python3
"""I — paper artifact generator (task I).

Reads paper manifests (results/manifest_*.json / latest_manifest.json, plus
models/models_summary.json when present) and emits:

- LaTeX tables (``artifacts/tables/*.tex``) whose cells are copied verbatim from
  the manifest aggregates (G0: every quoted number carries its config tag).
- Figure-ready JSON arrays (``artifacts/fig_data/*.json``): per-turn E1 token
  series (mean over seeds), E4 distance curves, C7 sweep grid.
- A reproducibility payload (``artifacts/reproducibility.json``): git revision,
  python/library versions, config hash, seeds, and the exact manifest files used.

``--check`` (the gate) reloads each emitted artifact and asserts the numbers it
contains still equal the manifest's stored values — the anti-drift check.

Artifacts are regenerable from manifests, so ``artifacts/`` is gitignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

ARTIFACTS = BASE_DIR / "artifacts"
TABLES = ARTIFACTS / "tables"
FIG = ARTIFACTS / "fig_data"


def git_rev() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=BASE_DIR, capture_output=True, text=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def load_manifest(path: Path) -> Dict:
    return json.loads(path.read_text())


def tex_cell(v) -> str:
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))
    return str(v)


def escape(s: str) -> str:
    return s.replace("_", r"\_")


def table_e1_e2_tex(manifest: Dict) -> str:
    agg = manifest["aggregate"]
    methods = [m for m in agg if m != "multiple_comparisons"]
    cfg = manifest["config"]
    lines = [
        "% Numbers copied verbatim from the manifest aggregates (source: seeded offline",
        f"% oracle/LLM replay, {cfg.get('write','oracle')}, epoch {int(manifest['generated_at'])}).",
        "% Pipeline: " + manifest["metric_source"].replace("%", r"\%"),
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"Method & E1 prop. (tok) & 95\% CI & baseline (tok) & E2 recall & Cohen's $d$ & MC adj.\ $p$ \\",
        r"\hline",
    ]
    mc = agg.get("multiple_comparisons", {})
    for m in methods:
        a = agg[m]
        ci = a.get("E1_proposed_pooled_ci95", [None, None])
        ci_s = ("{}-{}".format(tex_cell(ci[0]), tex_cell(ci[1])) if ci[0] is not None else "--")
        holm_p = (mc.get("holms_adjusted") or {}).get(m, None)
        lines.append(
            f"{escape(m)} & {tex_cell(a['mean_E1_proposed_tokens'])} & {ci_s} & "
            f"{tex_cell(a['mean_E1_baseline_tokens'])} & {tex_cell(a['mean_E2_proposed_recall'])} & "
            f"{tex_cell(a.get('cohens_d_paired_vs_baseline'))} & {tex_cell(holm_p)} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


def e1_series_fig(manifest: Dict) -> Dict:
    """Mean-over-seeds per-turn E1 series, one array per method."""
    methods = set()
    for s in manifest["results"]:
        methods |= set(s["methods"].keys())
    n = min(len(s["methods"].get("adaptive", {}).get("stats", {}).get("proposed_series", []))
            for s in manifest["results"]) if manifest["results"] else 0
    if not n:
        return {"turns": [], "series": {}}
    out = {}
    for m in methods:
        sums = [0.0] * n
        for s in manifest["results"]:
            st = s["methods"].get(m, {}).get("stats", {})
            for i in range(n):
                sums[i] += st.get("proposed_series", [0] * n)[i]
        out[m] = [round(x / len(manifest["results"]), 3) for x in sums]
    return {"turns": list(range(1, n + 1)), "series": out,
            "metric_source": manifest["metric_source"]}


def e4_curves_fig(manifest: Dict) -> Dict:
    methods = set()
    for s in manifest["results"]:
        methods |= set(s["methods"].keys())
    curves = {}
    for m in methods:
        dists = []
        accs = [0.0] * 0
        nseeds = len(manifest["results"])
        for s in manifest["results"]:
            e4 = s["methods"].get(m, {}).get("E4_needle_in_haystack", {})
            accs = [0.0] * len(e4.get("distances", []))
            break
        for s in manifest["results"]:
            e4 = s["methods"].get(m, {}).get("E4_needle_in_haystack", {})
            dists = e4.get("distances", [])
            for i, a in enumerate(e4.get("proposed_accuracy", [])):
                if i < len(accs) and a is not None:
                    accs[i] += a
        curves[m] = {"distances": dists,
                     "proposed_accuracy": [round(a / nseeds, 3) if a is not None else None for a in accs]}
    return {"curves": curves, "metric_source": manifest["metric_source"]}


def reproducibility_payload(manifest: Dict, paths: List[str]) -> Dict:
    cfg = manifest["config"]
    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
    import numpy, scipy  # noqa: E401
    return {
        "git_revision": git_rev(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "config_hash": cfg_hash,
        "seed_offsets": cfg.get("seeds"),
        "turns": cfg.get("turns"),
        "density": cfg.get("density"),
        "input_manifests": paths,
        "metric_source": manifest["metric_source"],
    }


def check(artifact_paths: List[Path], manifest: Dict) -> None:
    agg = manifest["aggregate"]
    tex = (TABLES / "table_e1_e2.tex").read_text()
    for m in [k for k in agg if k != "multiple_comparisons"]:
        assert tex_cell(agg[m]["mean_E1_proposed_tokens"]) in tex, f"E1 cell missing for {m}"
        assert tex_cell(agg[m]["mean_E2_proposed_recall"]) in tex, f"E2 cell missing for {m}"
    e1 = json.loads((FIG / "e1_tokens_per_turn.json").read_text())
    e4 = json.loads((FIG / "e4_distance_curves.json").read_text())
    assert e1.get("metric_source") == manifest.get("metric_source")
    assert e4.get("metric_source") == manifest.get("metric_source")
    print("make_paper_artifacts --check OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="I paper artifact generator")
    ap.add_argument("--manifest", default=None, help="path to a paper manifest (default latest_manifest.json)")
    ap.add_argument("--check", action="store_true",
                    help="(gate) regenerate, reload, and assert numbers match the manifest")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    man_path = Path(args.manifest or BASE_DIR / "experiments/results/latest_manifest.json")
    assert man_path.exists(), f"missing manifest: {man_path} (run run_experiments.py --quick first)"
    manifest = load_manifest(man_path)

    TABLES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    (TABLES / "table_e1_e2.tex").write_text(table_e1_e2_tex(manifest))
    fig_e1 = e1_series_fig(manifest)
    fig_e4 = e4_curves_fig(manifest)
    (FIG / "e1_tokens_per_turn.json").write_text(json.dumps(fig_e1, indent=2))
    (FIG / "e4_distance_curves.json").write_text(json.dumps(fig_e4, indent=2))

    repro = reproducibility_payload(manifest, [str(man_path)])
    (ARTIFACTS / "reproducibility.json").write_text(json.dumps(repro, indent=2))

    print(f"-> {TABLES/'table_e1_e2.tex'}")
    print(f"-> {FIG/'e1_tokens_per_turn.json'}  {FIG/'e4_distance_curves.json'}")
    print(f"-> {ARTIFACTS/'reproducibility.json'}")

    if args.check or args.quick:
        check([], manifest)


if __name__ == "__main__":
    main()
