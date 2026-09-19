"""Phase 19: repair counterfactual correction metadata and rerun E19."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments import e19_coding_generalization as e19


RESULTS_DIR = Path(__file__).resolve().parent / "results"
REPAIRED_JSON = RESULTS_DIR / "e19_coding_generalization_full_repaired.json"
REPAIRED_REPORT = (
    RESULTS_DIR / "e19_coding_generalization_full_repaired_report.md"
)


def run_preflight() -> int:
    """Run the 27 primary offline correction cells and hard-gate before LLM runs."""
    e19.check_models(
        e19.DEFAULT_MODEL,
        e19.DEFAULT_EMBED_MODEL,
        e19.DEFAULT_ENDPOINT,
        True,
    )

    embed_fn = e19._resolve_embed(
        True,
        e19.DEFAULT_EMBED_MODEL,
        e19.DEFAULT_ENDPOINT,
    )

    cells = []

    for task_id in e19.PRIMARY_TASKS:
        for seed in e19.FULL_SEEDS:
            for budget in e19.FULL_BUDGETS:
                key = f"{task_id}:{seed}:{budget}"
                print(f"[preflight] {key}", flush=True)
                cell = e19.run_offline_cell(
                    task_id,
                    seed,
                    budget,
                    embed_fn=embed_fn,
                    embedding_model=e19.DEFAULT_EMBED_MODEL,
                    endpoint=e19.DEFAULT_ENDPOINT,
                    model=e19.DEFAULT_MODEL,
                )
                cells.append(cell)

    gate = e19.offline_gate(cells, e19.PRIMARY_TASKS)

    print(json.dumps(gate, indent=2))

    if not gate["embedding_gate"]:
        print("PRE-FLIGHT FAIL: embedding_gate", file=sys.stderr)
        return 1

    if not gate["identity_gate"]:
        print("PRE-FLIGHT FAIL: correction_identity", file=sys.stderr)
        return 1

    if gate["embedding_consistent_cells"] != 27:
        print(
            "PRE-FLIGHT FAIL: expected 27/27 embedding-consistent cells",
            file=sys.stderr,
        )
        return 1

    if gate["identity_ok_cells"] != 27:
        print(
            "PRE-FLIGHT FAIL: expected 27/27 correction-identity cells",
            file=sys.stderr,
        )
        return 1

    print("PRE-FLIGHT PASS: 27/27 embedding + 27/27 identity")
    return 0


def run_full(resume: bool, force: bool) -> int:
    old_json = e19.OUT_JSON
    old_report = e19.OUT_REPORT

    e19.OUT_JSON = REPAIRED_JSON
    e19.OUT_REPORT = REPAIRED_REPORT

    try:
        argv = ["--full"]

        if resume:
            argv.append("--resume")
        if force:
            argv.append("--force")

        result = e19.main(argv)

        if result:
            result["repair_provenance"] = {
                "phase": 19,
                "type": "counterfactual_correction_metadata",
                "history_text_unchanged": True,
                "workspace_unchanged": True,
                "prompt_unchanged": True,
                "hidden_tests_unchanged": True,
                "gold_patches_unchanged": True,
                "memory_optimizer_unchanged": True,
            }

            REPAIRED_JSON.write_text(json.dumps(result, indent=2, default=str))
            REPAIRED_REPORT.write_text(e19.generate_report(result))
            print(f"Wrote {REPAIRED_JSON}")
            print(f"Wrote {REPAIRED_REPORT}")

        return 0 if result else 1
    finally:
        e19.OUT_JSON = old_json
        e19.OUT_REPORT = old_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--preflight",
        action="store_true",
        help="run only the 27-cell offline correction/embedding gate",
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="run the fresh 225-cell E19 full grid",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.preflight:
        return run_preflight()

    return run_full(
        resume=args.resume,
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())