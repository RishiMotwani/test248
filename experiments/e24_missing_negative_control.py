"""Phase-19 closure: rerun the single missing negative-control cell."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from experiments import e19_coding_generalization as e19
from experiments import coding_benchmark as cb


RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_JSON = RESULTS_DIR / "e24_missing_negative_control.json"

# The exactly missing cell from Phase 19
MISSING_TASK = "write_retry"
MISSING_SEED = 3
MISSING_BUDGET = 256
MISSING_METHOD = "llm_summarization"


def identify_missing() -> int:
    """Identify and print the single missing cell from the Phase-19 artifact."""
    json_path = RESULTS_DIR / "e19_coding_generalization_full_repaired.json"
    if not json_path.exists():
        print(f"ERROR: {json_path} not found", file=sys.stderr)
        return 1

    data = json.loads(json_path.read_text())
    recs = data.get("records", [])

    # Build expected grid from Phase-19 configuration
    primary_tasks = ["routing_policy", "retry_policy", "serialization_policy"]
    negative_tasks = ["cache_readonly", "write_retry"]
    all_tasks = primary_tasks + negative_tasks
    seeds = [1, 2, 3]
    budgets = [256, 512, 1024]
    methods = ["raw_clipped", "sliding_window", "llm_summarization", "vanilla_rag", "adaptive"]

    expected = {
        (t, s, b, m)
        for t in all_tasks
        for s in seeds
        for b in budgets
        for m in methods
    }

    actual = {
        (r["task_id"], r["seed"], r["historical_budget"], r["method"])
        for r in recs
    }

    missing = expected - actual

    if len(missing) == 0:
        print("ERROR: No missing cells found", file=sys.stderr)
        return 1
    if len(missing) > 1:
        print(f"ERROR: Expected 1 missing cell, found {len(missing)}", file=sys.stderr)
        for m in sorted(missing):
            print(f"  {m}", file=sys.stderr)
        return 1

    task_id, seed, budget, method = missing.pop()
    expected_key = (MISSING_TASK, MISSING_SEED, MISSING_BUDGET, MISSING_METHOD)
    if (task_id, seed, budget, method) != expected_key:
        print(f"ERROR: Missing cell {task_id}/{seed}/{budget}/{method} does not match expected {expected_key}", file=sys.stderr)
        return 1

    print(f"{task_id} / seed={seed} / budget={budget} / method={method}")
    return 0


def run_missing_cell() -> int:
    """Run the single missing negative-control cell using Phase-19 infrastructure."""
    # Verify the target cell
    target_key = (MISSING_TASK, MISSING_SEED, MISSING_BUDGET, MISSING_METHOD)
    print(f"Running missing cell: {target_key}")

    # Load models and verify
    e19.check_models(
        e19.DEFAULT_MODEL,
        e19.DEFAULT_EMBED_MODEL,
        e19.DEFAULT_ENDPOINT,
        True,  # use_embeddings
    )

    # Build embed function
    embed_fn = e19._resolve_embed(
        True,
        e19.DEFAULT_EMBED_MODEL,
        e19.DEFAULT_ENDPOINT,
    )

    # Build the task using Phase-19's builder (normalizes task_id)
    task = e19.build_e19_task(MISSING_TASK, seed=MISSING_SEED)

    # Build method
    method = cb.build_method(
        MISSING_METHOD,
        model=e19.DEFAULT_MODEL,
        endpoint=e19.DEFAULT_ENDPOINT,
        embed_fn=embed_fn,
        embedding_model=e19.DEFAULT_EMBED_MODEL,
        summarizer_generate_fn=lambda p, mx: cb.OllamaCoder(e19.DEFAULT_MODEL, e19.DEFAULT_ENDPOINT, e19.MAX_OUTPUT_TOKENS).generate(p)["text"],
        task=task,
    )

    # Prepare work root (matching Phase-19 naming)
    work_root = e19.WORK_ROOT / f"{MISSING_TASK}_{MISSING_SEED}_{MISSING_BUDGET}_{MISSING_METHOD}"

    # Run the cell
    start = time.time()
    rec = cb.run_method_run(
        task=task,
        seed=MISSING_SEED,
        historical_budget=MISSING_BUDGET,
        method=method,
        coder=cb.OllamaCoder(e19.DEFAULT_MODEL, e19.DEFAULT_ENDPOINT, e19.MAX_OUTPUT_TOKENS),
        work_root=work_root,
    )
    latency_ms = (time.time() - start) * 1000

    # Build result in same schema as E19 records
    result = {
        "experiment_id": "e24_missing_negative_control",
        "source_experiment": "e19_coding_generalization_full_repaired",
        "task_id": MISSING_TASK,
        "seed": MISSING_SEED,
        "budget": MISSING_BUDGET,
        "method": MISSING_METHOD,
        "success": bool(rec.get("final_success", False)),
        "first_pass_success": bool(rec.get("first_pass_success", False)),
        "final_success": bool(rec.get("final_success", False)),
        "attempt_count": rec.get("coding_attempts", 0),
        "failure_class": rec.get("failure_class"),
        "model": e19.DEFAULT_MODEL,
        "temperature": cb.CODING_TEMPERATURE,
        "timestamp": time.time(),
        "historical_context_tokens": rec.get("historical_context_tokens"),
        "workspace_context_tokens": rec.get("workspace_context_tokens"),
        "task_prompt_tokens": rec.get("task_prompt_tokens"),
        "total_prompt_tokens": rec.get("total_prompt_tokens"),
        "latency_ms": latency_ms,
        "model_calls": rec.get("model_calls", {}),
        "uses_production_pipeline": rec.get("uses_production_pipeline", False),
        "leakage_detected": rec.get("leakage_detected", False),
        "diagnostics": rec.get("diagnostics", {}),
        "historical_budget": MISSING_BUDGET,
        "context_sha": rec.get("context_sha"),
        "patch_sha": rec.get("patch_sha"),
    }

    OUTPUT_JSON.write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote {OUTPUT_JSON}")
    print(f"  task_id={result['task_id']} seed={result['seed']} budget={result['budget']} method={result['method']}")
    print(f"  final_success={result['final_success']} first_pass={result['first_pass_success']} failure_class={result['failure_class']}")

    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase-19 missing negative-control cell closure")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--identify", action="store_true", help="Identify the missing cell")
    group.add_argument("--run", action="store_true", help="Run the missing cell")
    args = parser.parse_args(argv)

    if args.identify:
        return identify_missing()
    if args.run:
        return run_missing_cell()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())