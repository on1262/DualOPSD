"""Main benchmark results: tab:main and the per-seed tables (App. Results by training seed).

Usage: python -m analysis.main_results [--print-tables]
Reads eval/main/<model>/<run>/<benchmark>/samples.jsonl and writes results/main/main_results.json.
"""

from __future__ import annotations

import argparse
from typing import Any

from analysis.common import (
    MODEL_LABELS,
    MODELS,
    SEEDS,
    add_common_args,
    aggregate_seeds,
    base_run,
    eval_file,
    find_step_run,
    output_dir,
    pm,
    print_tables,
    run_name,
    write_json,
)

EXP = "main"
# (key, run-name template); matched OPSD resolves to the single evaluated opsd_matched_seed<S>_step<N>.
METHODS = ("base", "sft", "grpo", "opsd", "opsd_matched", "dualopsd_direct", "dualopsd")
LABELS = {
    "base": "Base",
    "sft": "SFT",
    "grpo": "GRPO",
    "opsd": "OPSD",
    "opsd_matched": "OPSD, matched",
    "dualopsd_direct": r"\methodplain{}",
    "dualopsd": r"\method{}",
}


def method_runs(model: str, method: str) -> dict[int, str]:
    if method == "base":
        return {seed: base_run(EXP, model, seed) for seed in SEEDS}
    if method == "opsd_matched":
        return {seed: find_step_run(EXP, model, run_name("opsd_matched", seed))[0] for seed in SEEDS}
    return {seed: run_name(method, seed) for seed in SEEDS}


def build() -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model in MODELS:
        methods = {}
        for method in METHODS:
            runs = method_runs(model, method)
            methods[method] = aggregate_seeds(lambda seed, bench, runs=runs: eval_file(EXP, model, runs[seed], bench))
            methods[method]["runs"] = {str(seed): run for seed, run in runs.items()}
        best = max(("opsd", "opsd_matched"), key=lambda m: methods[m]["across_seeds"]["macro_avg_at_12"]["mean"])
        models[model] = {"methods": methods, "opsd_best": best}
    return {"training_seeds": list(SEEDS), "base_seed_role": "generation seed", "models": models}


def tables(result: dict[str, Any]) -> dict[str, list[list[str]]]:
    main_rows, out = [], {}
    for model in MODELS:
        data = result["models"][model]
        methods = data["methods"]
        for method in ("base", "sft", "grpo", data["opsd_best"], "dualopsd_direct", "dualopsd"):
            across = methods[method]["across_seeds"]
            label = r"OPSD$^{\ast}$" if method.startswith("opsd") else LABELS[method]
            cells = [pm(across["datasets"][b]["avg_at_12"]) for b in ("aime24", "aime25", "hmmt25")]
            main_rows.append([label, *cells, pm(across["macro_avg_at_12"])])
        seed_rows = []
        for method in METHODS:
            per_seed = methods[method]["per_seed"]
            cells = [
                " / ".join(f"{100 * per_seed[str(s)]['datasets'][b]['avg_at_12']:.2f}" for s in SEEDS)
                for b in ("aime24", "aime25", "hmmt25")
            ]
            cells.append(" / ".join(f"{100 * per_seed[str(s)]['macro_avg_at_12']:.2f}" for s in SEEDS))
            seed_rows.append([LABELS[method], *cells])
        suffix = {"qwen3-1.7b": "", "qwen3-4b": "-4b", "qwen3-8b": "-8b"}[model]
        out[f"tab:seed-results{suffix}"] = seed_rows
    return {"tab:main": main_rows, **out}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, EXP)
    args = parser.parse_args(argv)
    out = output_dir(args, EXP)
    result = build()
    print(write_json(out / "main_results.json", result, overwrite=args.overwrite))
    if args.print_tables:
        print_tables(tables(result))
    for model in MODELS:
        acc = {m: 100 * result["models"][model]["methods"][m]["across_seeds"]["macro_avg_at_12"]["mean"] for m in METHODS}
        print(MODEL_LABELS[model], " ".join(f"{m}={v:.2f}" for m, v in acc.items()))
    return result


if __name__ == "__main__":
    main()
