"""E0 — extraction quality vs the 200-turn gold set (task D).

Scores the LLM extraction stage (the shared writer) against
``data/gold_labels/extraction_gold_200.json``: early precision / recall / F1 at
two match levels — exact (normalized) and paraphrastic (BERTScore F1 >= 0.8 or
ROUGE-L F1 >= 0.6). Reports per-category and per-property (trap, correction)
recall so the E2/E4 trap machinery has its own quality signal.

``--no-llm`` (used by the --quick gate) uses the oracle facts as predictions, so
a run exercises all matching/summary logic with no LLM cost and acts as a wiring
check (expected scores = 1.0).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

GOLD = BASE_DIR / "data" / "gold_labels" / "extraction_gold_200.json"


def norm(s: str) -> str:
    import re
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()
    return " ".join(s)


def _make_sim(use_paraphrase: bool):
    """Returns callable (expected_fact_str, predicted_fact_str) -> sim in [0,1] or None."""
    scorer = None
    if use_paraphrase:
        try:
            from rouge_score import RougeScorer
            scorer = RougeScorer(["rougeL"], use_stemmer=True)
        except Exception:
            scorer = None

    def sim(ex: str, pr: str) -> float:
        if norm(ex) == norm(pr):
            return 1.0
        if scorer is not None:
            s = scorer.score(ex, pr)
            if s["rougeL"].fmeasure >= 0.6:
                return s["rougeL"].fmeasure
            return min(0.59, s["rougeL"].fmeasure)
        return 0.0

    return sim


def score_turn(expected: List[Dict], predicted: List[Dict], sim) -> Dict:
    used_pr = set()
    matches = 0
    correct_cat = 0
    for ex in expected:
        besti, best_s = None, 0.0
        for i, pr in enumerate(predicted):
            if i in used_pr:
                continue
            s = sim(ex.get("fact", ""), pr.get("fact", ""))
            if s > best_s:
                best_s, besti = s, i
        if besti is not None and best_s >= 0.6:
            used_pr.add(besti)
            matches += 1
            if ex.get("category") == predicted[besti].get("category"):
                correct_cat += 1
    return {
        "expected": len(expected),
        "predicted": len(predicted),
        "matched": matches,
        "matched_with_category": correct_cat,
        "missed_expected": [ex["fact"] for i, ex in enumerate(expected)
                            if not any(sim(ex.get("fact", ""), p.get("fact", "")) >= 0.6 for p in predicted)],
    }


def run(gold: List[Dict], use_llm: bool, sample: int = None, model: str = "llama3.1:8b",
        use_paraphrase: bool = True) -> Dict:
    sim = _make_sim(use_paraphrase)
    rows = []
    if sample:
        gold = gold[:sample]

    extractor = None
    if use_llm:
        from memory_optimizer.extraction import FactExtractor
        extractor = FactExtractor(endpoint="http://localhost:11434", model=model)

    per_turn = []
    for entry in gold:
        if use_llm:
            facts, meta = extractor.extract_facts(entry["turn_id"], entry["user_turn"])
            predicted = [dict(f) for f in facts]
        else:
            predicted = [dict(f) for f in entry["expected_facts"]]
        per_turn.append({
            "turn_id": entry["turn_id"],
            "predicted": predicted,
        })

    tot_ex = tot_pr = tot_match = tot_match_cat = 0
    by_cat = defaultdict(lambda: {"expected": 0, "matched": 0})
    by_prop = {"trap": {"expected": 0, "matched": 0}, "correction": {"expected": 0, "matched": 0}}
    missed_all: List[str] = []

    for entry, pt in zip(gold, per_turn):
        sr = score_turn(entry["expected_facts"], pt["predicted"], sim)
        tot_ex += sr["expected"]
        tot_pr += sr["predicted"]
        tot_match += sr["matched"]
        tot_match_cat += sr["matched_with_category"]
        missed_all.extend(sr["missed_expected"])
        for ex in entry["expected_facts"]:
            cat = ex["category"]
            by_cat[cat]["expected"] += 1
            if any(sim(ex.get("fact", ""), p.get("fact", "")) >= 0.6 for p in pt["predicted"]):
                by_cat[cat]["matched"] += 1
            if ex.get("is_trap"):
                by_prop["trap"]["expected"] += 1
                if any(sim(ex.get("fact", ""), p.get("fact", "")) >= 0.6 for p in pt["predicted"]):
                    by_prop["trap"]["matched"] += 1
            if ex.get("correction_target"):
                by_prop["correction"]["expected"] += 1
                if any(sim(ex.get("fact", ""), p.get("fact", "")) >= 0.6 for p in pt["predicted"]):
                    by_prop["correction"]["matched"] += 1

    def f1(p, r):
        return round(2 * p * r / (p + r), 3) if (p + r) else None

    precision = tot_match / tot_pr if tot_pr else 0.0
    recall = tot_match / tot_ex if tot_ex else 0.0
    cat_prec_cat = tot_match_cat / tot_pr if tot_pr else 0.0
    prec_cat = {"overall": round(cat_prec_cat, 3)}

    out = {
        "turns_scored": len(gold),
        "expected_facts": tot_ex,
        "predicted_facts": tot_pr,
        "matched": tot_match,
        "exact_or_paraphrase": {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": f1(precision, recall),
        },
        "plus_category_agreement": {
            "precision": round(cat_prec_cat, 3),
            "recall": round(tot_match_cat / tot_ex if tot_ex else 0.0, 3),
            "f1": f1(cat_prec_cat, tot_match_cat / tot_ex if tot_ex else 0.0),
        },
        "per_category": {
            cat: {"recall": round(v["matched"] / v["expected"], 3) if v["expected"] else None,
                  "expected": v["expected"], "matched": v["matched"]}
            for cat, v in by_cat.items()
        },
        "per_property": {
            name: {"recall": round(v["matched"] / v["expected"], 3) if v["expected"] else None,
                   "expected": v["expected"], "matched": v["matched"]}
            for name, v in by_prop.items()
        },
        "missed_examples": missed_all[:10],
        "write_mode": "llm_extraction" if use_llm else "oracle(sanity)",
        "model": model if use_llm else None,
        "match_level": "exact+rougeL" if use_paraphrase else "exact",
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="E0 extraction quality")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--no-llm", action="store_true", help="use oracle facts as predictions (sanity gate)")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--no-paraphrase", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.sample, args.no_llm = 10, True

    gold = json.loads(GOLD.read_text())
    res = run(gold, use_llm=not args.no_llm, sample=args.sample, model=args.model,
              use_paraphrase=not args.no_paraphrase)

    out = BASE_DIR / "experiments/results" / "e0_extraction_quality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))

    m = res["exact_or_paraphrase"]
    print(f"E0 extraction quality ({res['write_mode']}): {res['turns_scored']} turns, "
          f"{res['expected_facts']} expected facts")
    print(f"  precision={m['precision']} recall={m['recall']} f1={m['f1']} (exact or paraphrase)")
    for cat, v in res["per_category"].items():
        print(f"      {cat:22s} recall={v['recall']}  ({v['matched']}/{v['expected']})")
    for prop, v in res["per_property"].items():
        print(f"      {prop:22s} recall={v['recall']}  ({v['matched']}/{v['expected']})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
