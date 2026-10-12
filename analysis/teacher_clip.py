"""Teacher clipping ablation: tab:core-ablation, tab:teacher-clip-scales.

The teacher minimizes the DualOPSD objective D_KL(sg[q] || p_T) (residual target, selected beta)
with an upper (clip_high), lower (clip_low) or no clip per vocabulary contribution;
threshold = student.kl_clip. The unclipped arm is the main DualOPSD run. 100 steps, seeds 42/43/44.
Reads eval/teacher_clip/<model>/clip_<variant>_seed<S>/<benchmark>/samples.jsonl and
eval/main/<model>/dualopsd_seed<S>/<benchmark>/samples.jsonl.

Usage: python -m analysis.teacher_clip [--print-tables]
"""

from __future__ import annotations

import argparse
from statistics import mean
from typing import Any

from analysis.common import (
    BENCHMARKS,
    MODEL_LABELS,
    MODELS,
    SEEDS,
    add_common_args,
    aggregate_seeds,
    eval_file,
    mean_sd,
    output_dir,
    pm,
    print_tables,
    run_name,
    write_json,
)

EXP = "teacher_clip"
# variant -> (experiment, method) of its evaluation files
VARIANTS = {
    "clip_high": (EXP, "clip_high"),
    "clip_low": (EXP, "clip_low"),
    "clip_none": ("main", "dualopsd"),
}
THRESHOLDS = {"qwen3-1.7b": 1e-6, "qwen3-4b": 1e-6, "qwen3-8b": 1e-7}
BETAS = {"qwen3-1.7b": 0.8, "qwen3-4b": 0.7, "qwen3-8b": 0.5}
SUMMARY_METRICS = ("macro_avg_at_12", "nontruncated_mean_completion_tokens_macro", "truncation_rate")


def scale_average(models: dict[str, Any]) -> dict[str, Any]:
    """Per seed, average each metric over the three scales; then mean / sample SD over seeds."""

    return {
        variant: {
            metric: mean_sd(mean(models[m][variant]["per_seed"][str(seed)][metric] for m in MODELS) for seed in SEEDS)
            for metric in SUMMARY_METRICS
        }
        for variant in VARIANTS
    }


def build() -> dict[str, Any]:
    models = {
        model: {
            variant: aggregate_seeds(
                lambda seed, bench, m=model, e=exp, r=method: eval_file(e, m, run_name(r, seed), bench)
            )
            for variant, (exp, method) in VARIANTS.items()
        }
        for model in MODELS
    }
    return {"teacher_objective": "KL(sg[q] || p_T)", "thresholds": THRESHOLDS, "residual_beta": BETAS,
            "training_seeds": list(SEEDS), "models": models, "scale_average": scale_average(models)}


def tables(result: dict[str, Any]) -> dict[str, list[list[str]]]:
    core = [["Removed KL", "Upper", "Lower", "None"]]
    scales = []
    for model in MODELS:
        across = {v: result["models"][model][v]["across_seeds"] for v in VARIANTS}
        core += [
            [r"Acc. (\%)", *(pm(across[v]["macro_avg_at_12"]) for v in VARIANTS)],
            ["Len. (K)", *(pm(across[v]["nontruncated_mean_completion_tokens_macro"], scale=1e-3) for v in VARIANTS)],
            [r"Trunc. (\%)", *(pm(across[v]["truncation_rate"]) for v in VARIANTS)],
        ]
        for v, label in zip(VARIANTS, ("High clipping", "Low clipping", "No clipping")):
            scales.append([label, *(f"{100 * across[v]['datasets'][b]['avg_at_12']['mean']:.2f}" for b in BENCHMARKS),
                           pm(across[v]["macro_avg_at_12"]),
                           pm(across[v]["nontruncated_mean_completion_tokens_macro"], scale=1e-3),
                           pm(across[v]["truncation_rate"])])
    return {"tab:core-ablation": core, "tab:teacher-clip-scales": scales}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, EXP)
    args = parser.parse_args(argv)
    out = output_dir(args, EXP)
    result = build()
    print(write_json(out / "teacher_clip.json", result, overwrite=args.overwrite))
    if args.print_tables:
        print_tables(tables(result))
    for model in MODELS:
        print(MODEL_LABELS[model], " ".join(
            f"{v}={pm(result['models'][model][v]['across_seeds']['macro_avg_at_12'])}" for v in VARIANTS))
    print("Scale average", " ".join(
        f"{v}={pm(result['scale_average'][v]['macro_avg_at_12'])}" for v in VARIANTS))
    return result


if __name__ == "__main__":
    main()
