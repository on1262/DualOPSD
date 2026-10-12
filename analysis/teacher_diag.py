"""Teacher adaptation diagnostics on Qwen3-4B: fig:ablation-dynamics, tab:role-dynamics-data,
tab:role-clipped-mass.

Accuracy: judged teacher / student generations on the 500-problem diagnostic set
(generations/teacher_diag/qwen3-4b/<run>_step<N>/{teacher,student}_judged.jsonl).
JS retention and clipped forward-KL mass: post-hoc re-scoring summaries (see analysis.mismatch).

Usage: python -m analysis.teacher_diag [--print-tables]
"""

from __future__ import annotations

import argparse
from statistics import mean
from typing import Any

from analysis.common import (
    CHECKPOINT_STEPS,
    SEEDS,
    add_common_args,
    generation_file,
    mean_sd,
    output_dir,
    print_tables,
    read_jsonl,
    run_name,
    write_json,
)
from analysis.mismatch import LABELS, METHODS, MODEL, posthoc_trends

EXP = "teacher_diag"
NUM_PROBLEMS = 500


def pair_accuracy(method: str, seed: int, step: int) -> dict[str, Any]:
    run = run_name(method, seed)
    rows = {p: read_jsonl(generation_file(EXP, MODEL, run, step, p, judged=True)) for p in ("student", "teacher")}
    by_id = {p: {str(r["id"]): r for r in rows[p]} for p in rows}
    if any(len(rows[p]) != NUM_PROBLEMS or len(by_id[p]) != NUM_PROBLEMS for p in rows) or by_id["student"].keys() != by_id["teacher"].keys():
        raise ValueError(f"unpaired diagnostic generations: {run} step {step}")
    accuracy = {f"{p}_accuracy": mean(bool(r["correct"]) for r in rows[p]) for p in rows}
    gap = mean(float(by_id["teacher"][i]["correct"]) - float(by_id["student"][i]["correct"]) for i in by_id["student"])
    return {**accuracy, "gap": gap,
            "judge_parse_failures": sum(not bool(r["judge_parse_valid"]) for p in rows for r in rows[p])}


def accuracy_summary() -> dict[str, Any]:
    """Judged accuracy of DualOPSD w/o residual and DualOPSD (OPSD is not judged)."""
    out: dict[str, Any] = {}
    for method in METHODS[1:]:
        per_seed = {}
        for seed in SEEDS:
            steps = {str(step): pair_accuracy(method, seed, step) for step in CHECKPOINT_STEPS}
            per_seed[str(seed)] = {
                "steps": steps,
                **{f"five_step_{k}": mean(steps[str(s)][k] for s in CHECKPOINT_STEPS)
                   for k in ("student_accuracy", "teacher_accuracy", "gap")},
            }
        out[method] = {
            "per_seed": per_seed,
            "steps": {
                str(step): {k: mean_sd(per_seed[str(seed)]["steps"][str(step)][k] for seed in SEEDS)
                            for k in ("student_accuracy", "teacher_accuracy", "gap")}
                for step in CHECKPOINT_STEPS
            },
            **{f"five_step_{k}": mean_sd(per_seed[str(seed)][f"five_step_{k}"] for seed in SEEDS)
               for k in ("student_accuracy", "teacher_accuracy", "gap")},
        }
    return out


def build() -> dict[str, Any]:
    posthoc = posthoc_trends()
    clipped = {
        method: [posthoc[method][str(s)]["metrics"]["positive_kl_clipped_rate"]["problem_mean"] for s in CHECKPOINT_STEPS]
        for method in METHODS
    }
    js = {
        method: [posthoc[method][str(s)]["metrics"]["js_retention"]["problem_mean"] for s in CHECKPOINT_STEPS]
        for method in METHODS[1:]
    }
    return {
        "model": MODEL,
        "training_seeds": list(SEEDS),
        "steps": list(CHECKPOINT_STEPS),
        "accuracy": accuracy_summary(),
        "js_retention": js,
        "clipped_mass": {m: {"steps": rows, "five_step_mean": mean(r["mean"] for r in rows)} for m, rows in clipped.items()},
    }


def tables(result: dict[str, Any]) -> dict[str, list[list[str]]]:
    def pm(stats: dict[str, float], scale: float, digits: int) -> str:
        return f"{scale * stats['mean']:.{digits}f}±{scale * stats['sample_sd']:.{digits}f}"

    dynamics = []
    for method in METHODS[1:]:
        for index, step in enumerate(CHECKPOINT_STEPS):
            acc = result["accuracy"][method]["steps"][str(step)]
            dynamics.append([str(step), pm(acc["teacher_accuracy"], 100, 2), pm(acc["student_accuracy"], 100, 2),
                             pm(result["js_retention"][method][index], 1, 3)])
    clipped = [
        [LABELS[method],
         *(pm(row, 100, 1) for row in result["clipped_mass"][method]["steps"]),
         f"{100 * result['clipped_mass'][method]['five_step_mean']:.1f}"]
        for method in METHODS
    ]
    return {"tab:role-dynamics-data": dynamics, "tab:role-clipped-mass": clipped}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, EXP)
    args = parser.parse_args(argv)
    out = output_dir(args, EXP)
    result = build()
    print(write_json(out / "teacher_diag.json", result, overwrite=args.overwrite))
    if args.print_tables:
        print_tables(tables(result))
    for method in METHODS[1:]:
        acc = result["accuracy"][method]
        print(method, "five-step teacher/student/gap",
              " / ".join(f"{100 * acc[f'five_step_{k}']['mean']:.2f}" for k in ("teacher_accuracy", "student_accuracy", "gap")))
    return result


if __name__ == "__main__":
    main()
