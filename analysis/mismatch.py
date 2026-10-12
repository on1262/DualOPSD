"""OPSD mismatch diagnostics on Qwen3-4B: fig:diagnostic, tab:diagnostic-ci, tab:recurrence-sensitivity.

Aggregates the GPU re-scoring outputs over training seeds 42/43/44:
  diagnostics/mismatch/posthoc/qwen3-4b/<run>_step<N>/summary.json   (KL, clip rate, JS retention, ...)
  diagnostics/mismatch/recurrence/qwen3-4b/<run>.json                (top-k recurrence R_k, step 20 -> 100)
Every reported value is the problem-level mean within a seed, then mean and sample SD across seeds.

Usage: python -m analysis.mismatch [--print-tables]
"""

from __future__ import annotations

import argparse
import math
from typing import Any

from analysis.common import (
    CHECKPOINT_STEPS,
    SEEDS,
    add_common_args,
    diagnostics_dir,
    mean_sd,
    output_dir,
    print_tables,
    read_json,
    run_name,
    write_json,
)

EXP = "mismatch"
MODEL = "qwen3-4b"
METHODS = ("opsd", "dualopsd_direct", "dualopsd")
LABELS = {"opsd": "OPSD", "dualopsd_direct": r"\methodplain{}", "dualopsd": r"\method{}"}
POSTHOC_METRICS = (
    "positive_kl_clipped_rate",
    "teacher_to_student_kl",
    "student_to_teacher_kl",
    "js_retention",
    "current_privileged_js",
    "initial_privileged_js",
    "target_fit_kl",
    "target_to_student_kl",
)
RECURRENCE_SIGNALS = ("positive_clipped_t2s", "positive_t2s", "positive_s2t")
K_VALUES = ("1", "5", "10", "20")


def posthoc_summary(method: str, seed: int, step: int) -> dict[str, Any]:
    path = diagnostics_dir(EXP, "posthoc", MODEL, f"{run_name(method, seed)}_step{step}", "summary.json")
    summary = read_json(path)
    if summary.get("status") != "complete" or int(summary["checkpoint_step"]) != step:
        raise ValueError(f"incomplete or mismatched post-hoc summary: {path}")
    return summary


def posthoc_trends() -> dict[str, Any]:
    trends: dict[str, Any] = {}
    for method in METHODS:
        trends[method] = {}
        for step in CHECKPOINT_STEPS:
            summaries = {seed: posthoc_summary(method, seed, step) for seed in SEEDS}
            metrics = {}
            for metric in POSTHOC_METRICS:
                if not all(metric in s["problem_metric_means"] for s in summaries.values()):
                    continue
                problem = {str(seed): s["problem_metric_means"][metric]["mean"] for seed, s in summaries.items()}
                token = {str(seed): (s["token_metrics"].get(metric) or {}).get("mean") for seed, s in summaries.items()}
                metrics[metric] = {"problem_mean": {**mean_sd(problem.values()), "per_seed": problem}}
                if None not in token.values():
                    metrics[metric]["token_weighted_mean"] = {**mean_sd(token.values()), "per_seed": token}
            trends[method][str(step)] = {
                "metrics": metrics,
                "processed_examples": mean_sd(float(s["processed_examples"]) for s in summaries.values()),
                "retokenization_coverage": mean_sd(float(s["retokenization_coverage"]) for s in summaries.values()),
            }
    return trends


def recurrence_summary() -> dict[str, Any]:
    payloads = {
        (method, seed): read_json(diagnostics_dir(EXP, "recurrence", MODEL, f"{run_name(method, seed)}.json"))
        for method in METHODS for seed in SEEDS
    }
    for key, payload in payloads.items():
        if payload.get("status") != "complete" or payload["processed_problems"] != payload["requested_problems"]:
            raise ValueError(f"incomplete recurrence result: {key}")
    for field in ("canonical_response_sha256", "threshold", "processed_problems"):
        if len({str(p[field]) for p in payloads.values()}) != 1:
            raise ValueError(f"recurrence inputs differ in {field}")
    methods: dict[str, Any] = {}
    for method in METHODS:
        methods[method] = {}
        for signal in RECURRENCE_SIGNALS:
            methods[method][signal] = {}
            for k in K_VALUES:
                rows = {seed: payloads[(method, seed)]["metrics"][signal][k] for seed in SEEDS}
                entry = {}
                for field in ("problem_mean", "shuffled_problem_mean", "token_weighted_mean"):
                    values = {str(seed): float(row[field]) for seed, row in rows.items()}
                    if not all(math.isfinite(v) for v in values.values()):
                        raise ValueError(f"non-finite recurrence value: {method}/{signal}/{k}/{field}")
                    entry[field] = {**mean_sd(values.values()), "per_seed": values}
                methods[method][signal][k] = entry
    first = next(iter(payloads.values()))
    return {"canonical_response_sha256": first["canonical_response_sha256"], "threshold": first["threshold"],
            "processed_problems": first["processed_problems"], "primary_k": 10, "methods": methods}


def build() -> dict[str, Any]:
    return {"model": MODEL, "training_seeds": list(SEEDS), "steps": list(CHECKPOINT_STEPS),
            "posthoc": posthoc_trends(), "recurrence": recurrence_summary()}


def _pm3(stats: dict[str, float], digits: int = 3) -> str:
    return f"{stats['mean']:.{digits}f}±{stats['sample_sd']:.{digits}f}"


def tables(result: dict[str, Any]) -> dict[str, list[list[str]]]:
    diag = []
    for method in METHODS:
        for step in CHECKPOINT_STEPS:
            metrics = result["posthoc"][method][str(step)]["metrics"]
            diag.append([str(step), *(_pm3(metrics[m]["problem_mean"]) for m in POSTHOC_METRICS[:3])])
    recurrence = []
    for signal in RECURRENCE_SIGNALS:
        rows = result["recurrence"]["methods"]
        for k in K_VALUES:
            recurrence.append([f"k={k}", *(_pm3(rows[m][signal][k]["problem_mean"]) for m in METHODS)])
        recurrence.append(["Random-pair", *(_pm3(rows[m][signal]["10"]["shuffled_problem_mean"], 4) for m in METHODS)])
    return {"tab:diagnostic-ci": diag, "tab:recurrence-sensitivity": recurrence}


def figure_data(result: dict[str, Any]) -> dict[str, Any]:
    def series(metric: str) -> dict[str, Any]:
        out = {}
        for method in METHODS:
            stats = [result["posthoc"][method][str(s)]["metrics"][metric]["problem_mean"] for s in CHECKPOINT_STEPS]
            out[method] = {"mean": [x["mean"] for x in stats], "sample_sd": [x["sample_sd"] for x in stats]}
        return out

    r10 = {
        method: {signal: result["recurrence"]["methods"][method][signal]["10"]["problem_mean"]
                 for signal in ("positive_clipped_t2s", "positive_s2t")}
        for method in METHODS
    }
    return {"steps": list(CHECKPOINT_STEPS), "teacher_to_student_kl": series("teacher_to_student_kl"),
            "student_to_teacher_kl": series("student_to_teacher_kl"), "r10": r10}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, EXP)
    args = parser.parse_args(argv)
    out = output_dir(args, EXP)
    result = build()
    result["figure"] = figure_data(result)
    print(write_json(out / "mismatch.json", result, overwrite=args.overwrite))
    if args.print_tables:
        print_tables(tables(result))
    return result


if __name__ == "__main__":
    main()
