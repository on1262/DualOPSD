"""Qwen3-4B checkpoint accuracy curves (fig:acc-curves).

Reads eval/curves/qwen3-4b/{opsd,dualopsd}_seed<S>_step<N>/<benchmark>/samples.jsonl for steps
20..100 and writes results/curves/curves.{json,csv} (mean and sample SD over training seeds).

Usage: python -m analysis.curves
"""

from __future__ import annotations

import argparse
import csv
import io
from typing import Any

from analysis.common import (
    BENCHMARKS,
    CHECKPOINT_STEPS,
    SEEDS,
    add_common_args,
    check_same_problems,
    eval_file,
    mean_sd,
    output_dir,
    run_name,
    summarize_run,
    write_json,
    write_text,
)

EXP = "curves"
MODEL = "qwen3-4b"
METHODS = ("opsd", "dualopsd")


def build() -> dict[str, Any]:
    raw: dict[str, Any] = {}
    for method in METHODS:
        raw[method] = {}
        for step in CHECKPOINT_STEPS:
            raw[method][str(step)] = {
                str(seed): summarize_run({b: eval_file(EXP, MODEL, run_name(method, seed, step=step), b) for b in BENCHMARKS})
                for seed in SEEDS
            }
    check_same_problems(row for steps in raw.values() for seeds in steps.values() for row in seeds.values())
    curves: dict[str, Any] = {}
    for method in METHODS:
        curves[method] = {}
        for label in (*BENCHMARKS, "macro"):
            stats = [
                mean_sd(
                    row["macro_avg_at_12"] if label == "macro" else row["datasets"][label]["avg_at_12"]
                    for row in raw[method][str(step)].values()
                )
                for step in CHECKPOINT_STEPS
            ]
            curves[method][label] = {"steps": list(CHECKPOINT_STEPS), "mean": [s["mean"] for s in stats],
                                     "sample_sd": [s["sample_sd"] for s in stats]}
    for steps in raw.values():
        for seeds in steps.values():
            for row in seeds.values():
                row.pop("_signatures", None)
    return {"model": MODEL, "training_seeds": list(SEEDS), "evaluation_seed": 42, "raw": raw, "curves": curves}


def to_csv(result: dict[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["method", "training_seed", "step", "benchmark", "avg_at_12"])
    for method, steps in result["raw"].items():
        for step, seeds in steps.items():
            for seed, row in seeds.items():
                for bench in BENCHMARKS:
                    writer.writerow([method, seed, step, bench, row["datasets"][bench]["avg_at_12"]])
                writer.writerow([method, seed, step, "macro", row["macro_avg_at_12"]])
    return buffer.getvalue()


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, EXP)
    args = parser.parse_args(argv)
    out = output_dir(args, EXP)
    result = build()
    print(write_json(out / "curves.json", result, overwrite=args.overwrite))
    print(write_text(out / "curves.csv", to_csv(result), overwrite=args.overwrite))
    for method in METHODS:
        print(method, "macro", " ".join(f"{100 * v:.2f}" for v in result["curves"][method]["macro"]["mean"]))
    return result


if __name__ == "__main__":
    main()
