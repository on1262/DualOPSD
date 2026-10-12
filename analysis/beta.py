"""Residual-weight selection data: tab:beta-validation, fig:beta-validation and tab:beta.

Held-out validation: one judged step-100 student response per problem on the 500-problem set
(generations/beta/<model>/dualopsd_beta<b>_seed42_step100/student_judged.jsonl).
Test-set sensitivity: Qwen3-8B AIME24 avg@12 of eval/beta/qwen3-8b/<run>/aime24.

Usage: python -m analysis.beta [--print-tables]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from analysis.common import (
    BENCHMARKS,
    MODELS,
    add_common_args,
    eval_file,
    generation_file,
    heldout_file,
    output_dir,
    print_tables,
    read_jsonl,
    run_name,
    scratch,
    summarize_run,
    summarize_samples,
    write_json,
)

EXP = "beta"
SENSITIVITY_MODEL = "qwen3-8b"
SENSITIVITY_BETAS = (0.1, 0.2, 0.5, 0.7, 1.0)


def candidate_betas(model: str) -> list[float]:
    root = scratch() / "generations" / EXP / model
    betas = []
    for path in root.glob("dualopsd_beta*_seed42_step100"):
        if (path / "student_judged.jsonl").exists():
            betas.append(float(path.name.split("_beta")[1].split("_seed")[0].replace("p", ".")))
    return sorted(betas)


def validation_summary(model: str, beta: float, expected_ids: list[str]) -> dict[str, Any]:
    path = generation_file(EXP, model, run_name("dualopsd", 42, beta=beta), 100, "student", judged=True)
    rows = read_jsonl(path)
    if [str(row["id"]) for row in rows] != expected_ids:
        raise ValueError(f"held-out coverage/order mismatch: {path}")
    if any(int(row["sampling_seed"]) != 42 + index for index, row in enumerate(rows)):
        raise ValueError(f"sampling seed mismatch: {path}")
    return {
        "beta": beta,
        "source": str(path),
        "correct_count": sum(bool(row["correct"]) for row in rows),
        "accuracy": mean(bool(row["correct"]) for row in rows),
        "truncation_rate": mean(bool(row["truncated"]) for row in rows),
        "observed_mean_completion_tokens": mean(int(row["completion_tokens"]) for row in rows),
        "boxed_rate": mean(row["terminal_extraction_mode"] == "last_boxed" for row in rows),
        "judge_parse_failure_rate": mean(not bool(row["judge_parse_valid"]) for row in rows),
    }


def held_out_ids(path: Path | None = None) -> list[str]:
    return [str(row["id"]) for row in read_jsonl(path or heldout_file())]


def validation(models=MODELS, heldout: Path | None = None) -> dict[str, Any]:
    ids = held_out_ids(heldout)
    return {model: {f"{b:.1f}": validation_summary(model, b, ids) for b in candidate_betas(model)} for model in models}


def sensitivity() -> dict[str, Any]:
    """Qwen3-8B seed-42 step-100 AIME24 results for each beta (and w/o residual, beta=0)."""
    rows: dict[str, Any] = {}
    runs = {"direct": run_name("dualopsd_direct", 42)}
    runs.update({f"{b:.1f}": run_name("dualopsd", 42, beta=b) for b in SENSITIVITY_BETAS})
    for key, run in runs.items():
        rows[key] = {"run": run, "aime24": summarize_samples(eval_file(EXP, SENSITIVITY_MODEL, run, "aime24"))}
        rows[key]["aime24"] = {k: v for k, v in rows[key]["aime24"].items() if not k.startswith("_")}
        paths = {b: eval_file(EXP, SENSITIVITY_MODEL, run, b) for b in BENCHMARKS}
        if all(path.exists() for path in paths.values()):
            pooled = summarize_run(paths)
            pooled.pop("_signatures")
            rows[key]["three_benchmarks"] = pooled
    return rows


def rank_betas(candidates: dict[str, Any]) -> list[str]:
    """Selection rule: most correct, then lower truncation, fewer tokens, smaller beta."""
    return sorted(candidates, key=lambda key: (
        -candidates[key]["correct_count"],
        candidates[key]["truncation_rate"],
        candidates[key]["observed_mean_completion_tokens"],
        candidates[key]["beta"],
    ))


def build(heldout: Path | None = None) -> dict[str, Any]:
    held_out = validation(heldout=heldout)
    selected = {model: held_out[model][rank_betas(held_out[model])[0]]["beta"] for model in MODELS}
    return {"selected_beta": selected, "validation": held_out, "sensitivity": {SENSITIVITY_MODEL: sensitivity()}}


def tables(result: dict[str, Any]) -> dict[str, list[list[str]]]:
    validation_rows = [
        [f"{row['beta']:.1f}", str(row["correct_count"]), f"{100 * row['accuracy']:.1f}",
         f"{100 * row['truncation_rate']:.1f}", f"{row['observed_mean_completion_tokens']:,.0f}"]
        for model in MODELS for row in result["validation"][model].values()
    ]
    beta_rows = []
    for key, row in result["sensitivity"][SENSITIVITY_MODEL].items():
        label = r"\methodplain{}" if key == "direct" else rf"\method{{}} ($\beta={float(key):g}$)"
        aime = row["aime24"]
        beta_rows.append([label, f"{100 * aime['avg_at_12']:.2f} / {100 * aime['pass_at_12']:.2f}",
                          f"{100 * aime['truncation_rate']:.2f}", f"{aime['nontruncated_mean_completion_tokens'] / 1000:.2f}K"])
    return {"tab:beta-validation": validation_rows, "tab:beta": beta_rows}


def figure_data(result: dict[str, Any]) -> dict[str, Any]:
    return {
        model: {
            "betas": [row["beta"] for row in result["validation"][model].values()],
            "accuracy": [100 * row["accuracy"] for row in result["validation"][model].values()],
            "selected_beta": result["selected_beta"][model],
        }
        for model in MODELS
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, EXP)
    parser.add_argument("--heldout-file", type=Path, default=None, help="default: $HELDOUT_FILE")
    args = parser.parse_args(argv)
    out = output_dir(args, EXP)
    result = build(args.heldout_file)
    result["figure"] = figure_data(result)
    print(write_json(out / "beta.json", result, overwrite=args.overwrite))
    if args.print_tables:
        print_tables(tables(result))
    return result


if __name__ == "__main__":
    main()
