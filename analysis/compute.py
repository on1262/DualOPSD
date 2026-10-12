"""Training compute, GPU hours, length and truncation of the main distillation runs.

Produces tab:full-compute, tab:tradeoff, tab:tradeoff-by-scale and tab:full-generation.
PFLOPs are a dense-linear estimate from the logged rollouts::

    PFLOPs = N * (c_S * student_sequence_tokens + c_T * teacher_sequence_tokens) / 1e15

with (c_S, c_T) = (8, 2) for OPSD, (10, 8) for DualOPSD w/o residual and (14, 10) for DualOPSD
(2N per frozen forward token, 6N per trainable forward+backward token, 2N per generated
student-sequence token).  GPU hours are Slurm elapsed time x GPUs of the seed-42 training
jobs, read from GPU-monitor logs (``--gpu-log-dir``) or ``sacct``.  The jobs come from
``--gpu-jobs`` (JSON ``{"jobs": {model: {method: [slurm job ids]}}}``) or, by default, from the
submission log ``$DUALOPSD_SCRATCH/logs/jobs.tsv`` written by scripts/lib.sh.  Compute-matched
OPSD counts its OPSD step-100 job plus every continuation job.  GPU hours are reported as
n/a when neither source lists the jobs.

Usage: python -m analysis.compute [--print-tables] [--gpu-log-dir DIR ...]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from analysis import main_results
from analysis.common import (
    MODELS,
    SEEDS,
    add_common_args,
    find_step_run,
    mean_sd,
    output_dir,
    pm,
    print_tables,
    read_json,
    read_jsonl,
    results_dir,
    run_dir,
    run_name,
    scratch,
    write_json,
)

EXP = "main"
PARAMETERS = {"qwen3-1.7b": 2_031_739_904, "qwen3-4b": 4_411_424_256, "qwen3-8b": 8_190_735_360}
MODEL_ENV = {"qwen3-1.7b": "MODEL_QWEN3_1P7B", "qwen3-4b": "MODEL_QWEN3_4B", "qwen3-8b": "MODEL_QWEN3_8B"}
COEFFICIENTS = {"opsd": (8, 2), "dualopsd_direct": (10, 8), "dualopsd": (14, 10)}
METHODS = ("opsd", "opsd_matched", "dualopsd_direct", "dualopsd")
LABELS = {
    "opsd": "OPSD",
    "opsd_matched": "OPSD, matched compute",
    "dualopsd_direct": r"\methodplain{}",
    "dualopsd": r"\method{}",
}
GPU_HOURS_SEED = 42

# ------------------------------------------------------------------------------ PFLOPs


@lru_cache(maxsize=None)
def tokenizer_for(model: str):
    from transformers import AutoTokenizer

    path = os.environ.get(MODEL_ENV[model])
    if not path:
        raise RuntimeError(f"{MODEL_ENV[model]} is not set; source configs/user.env")
    return AutoTokenizer.from_pretrained(path, local_files_only=Path(path).is_dir())


@lru_cache(maxsize=None)
def training_examples(path: str) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in read_jsonl(Path(path))}


def resolve_train_file(config: dict[str, Any], override: Path | None) -> Path:
    if override:
        return override
    recorded = Path(config["data"]["train_file"])
    data_dir = os.environ.get("DATA_DIR")
    if data_dir and (Path(data_dir) / recorded.name).is_file():
        return Path(data_dir) / recorded.name
    return recorded


def lengths(tokenizer, texts: list[str], batch_size: int = 128) -> list[int]:
    out: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], add_special_tokens=False, return_length=True)
        out.extend(int(value) for value in batch["length"])
    return out


_TEACHER_LENGTHS: dict[tuple[Any, ...], int] = {}


def token_vectors(rows: list[dict[str, Any]], config: dict[str, Any], model: str,
                  train_file: Path | None = None) -> tuple[list[int], list[int], list[int]]:
    """Per-rollout student prompt, teacher prompt and completion token counts."""
    from dualopsd.prompts import build_teacher_prompt, render_chat, upstream_teacher_messages

    tokenizer = tokenizer_for(model)
    path = str(resolve_train_file(config, train_file))
    examples = training_examples(path)
    # Same prompt budget as DistillTrainer._render_teacher.
    max_prompt = int(config["rollout"]["max_model_len"]) - int(config["rollout"]["max_new_tokens"])
    key = (model, path, max_prompt)
    ids = [str(row["example_id"]) for row in rows]
    missing = [i for i in dict.fromkeys(ids) if (*key, i) not in _TEACHER_LENGTHS]
    prompts = [
        render_chat(tokenizer, upstream_teacher_messages(str(examples[i]["problem"]), str(examples[i]["reference"])),
                    thinking=False)
        for i in missing
    ]
    for example_id, length in zip(missing, lengths(tokenizer, prompts)):
        if length > max_prompt:  # reference truncated to fit, as in the trainer
            example = examples[example_id]
            fitted = build_teacher_prompt(
                tokenizer, str(example["problem"]), str(example["reference"]), domain=example.get("domain"),
                thinking=False, max_tokens=max_prompt,
            )
            length = len(tokenizer(fitted, add_special_tokens=False)["input_ids"])
        _TEACHER_LENGTHS[(*key, example_id)] = length
    return (
        lengths(tokenizer, [str(row["input"]) for row in rows]),
        [_TEACHER_LENGTHS[(*key, i)] for i in ids],
        [int(row["resp_token"]) for row in rows],
    )


def token_counts(vectors: tuple[list[int], list[int], list[int]], count: int | None = None) -> dict[str, int]:
    student, teacher, completion = (sum(vector[:count]) for vector in vectors)
    return {
        "rollout_rows": len(vectors[0][:count]),
        "student_prompt_tokens": student,
        "teacher_prompt_tokens": teacher,
        "completion_tokens": completion,
        "student_sequence_tokens": student + completion,
        "teacher_sequence_tokens": teacher + completion,
    }


def pflops(objective: str, counts: dict[str, int], model: str) -> float:
    c_student, c_teacher = COEFFICIENTS[objective]
    total = PARAMETERS[model] * (c_student * counts["student_sequence_tokens"] + c_teacher * counts["teacher_sequence_tokens"])
    return total / 1e15


def checkpoint_steps(root: Path) -> list[int]:
    steps = []
    for path in root.glob("checkpoint-*"):
        suffix = path.name.rsplit("-", 1)[1]
        if suffix.isdigit() and (path / "student").is_dir():
            steps.append(int(suffix))
    return sorted(steps)


@lru_cache(maxsize=None)
def run_vectors(root: Path, train_file: Path | None = None) -> tuple[tuple[list[int], list[int], list[int]], int]:
    """Token vectors of every logged rollout of a run and its rollout batch size."""
    steps = checkpoint_steps(root)
    config_step = 100 if 100 in steps else steps[-1]
    config = yaml.safe_load((root / f"checkpoint-{config_step}/config.yaml").read_text(encoding="utf-8"))
    rows = read_jsonl(root / "rollouts.jsonl")
    batch = int(config["data"]["batch_size"])
    if len(rows) != steps[-1] * batch:
        raise ValueError(f"{root}: expected {steps[-1] * batch} rollouts, found {len(rows)}")
    return token_vectors(rows, config, root.parent.name, train_file), batch


def run_pflops(root: Path, objective: str, step: int, train_file: Path | None = None) -> float:
    vectors, batch = run_vectors(root, train_file)
    return pflops(objective, token_counts(vectors, step * batch), root.parent.name)


def pflops_from_rollouts(train_file: Path | None) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for model in MODELS:
        out[model] = {method: {} for method in METHODS}
        for seed in SEEDS:
            # OPSD step 100 and the matched checkpoint lie on the same (continued) OPSD trajectory.
            matched_root = run_dir(EXP, model, run_name("opsd_matched", seed))
            _, matched_step = find_step_run(EXP, model, run_name("opsd_matched", seed))
            out[model]["opsd"][str(seed)] = run_pflops(matched_root, "opsd", 100, train_file)
            out[model]["opsd_matched"][str(seed)] = run_pflops(matched_root, "opsd", matched_step, train_file)
            for method in ("dualopsd_direct", "dualopsd"):
                out[model][method][str(seed)] = run_pflops(run_dir(EXP, model, run_name(method, seed)), method, 100, train_file)
    return out


# --------------------------------------------------------------------------- GPU hours


def elapsed_from_gpu_log(job: str, log_dirs: list[Path]) -> float:
    matches = [p for d in log_dirs for p in d.rglob(f"*_{job}_gpu.csv")]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one GPU-monitor log for job {job}, found {len(matches)}")
    with matches[0].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, skipinitialspace=True))
    stamps = [datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f") for row in rows]
    gpus = len({row["index"] for row in rows})
    return (stamps[-1] - stamps[0]).total_seconds() * gpus


def elapsed_from_sacct(jobs: list[str]) -> dict[str, float]:
    command = ["sacct", "-X", "-n", "-P", "-j", ",".join(jobs), "--format=JobID,ElapsedRaw,AllocTRES"]
    output = subprocess.run(command, check=True, text=True, capture_output=True, timeout=120).stdout
    records = {}
    for line in output.splitlines():
        job, elapsed, tres = line.split("|")[:3]
        gpus = re.search(r"(?:^|,)gres/gpu=([0-9]+)", tres)
        records[job] = int(elapsed) * (int(gpus.group(1)) if gpus else 1)
    missing = sorted(set(jobs) - set(records))
    if missing:
        raise ValueError(f"missing sacct records: {missing}")
    return records


def jobs_from_submission_log(path: Path) -> dict[str, dict[str, list[str]]]:
    """Seed-42 training jobs per (model, method) from the ``jobs.tsv`` submission log."""
    names: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                names.setdefault(fields[2], []).append(fields[1])
    out: dict[str, dict[str, list[str]]] = {}
    for model in MODELS:
        prefix = f"{EXP}_{model}_"
        rows = {m: names.get(f"{prefix}{run_name(m, GPU_HOURS_SEED)}_train", [])
                for m in ("opsd", "dualopsd_direct", "dualopsd")}
        matched = re.escape(f"{prefix}{run_name('opsd_matched', GPU_HOURS_SEED)}_train") + r"[0-9]*"
        continuation = [job for name, ids in names.items() if re.fullmatch(matched, name) for job in ids]
        if continuation:
            rows["opsd_matched"] = rows["opsd"] + continuation
        out[model] = {m: ids for m, ids in rows.items() if ids}
    return out


def gpu_hours(jobs_file: Path | None, log_dirs: list[Path] | None) -> tuple[dict[str, dict[str, float | None]], str]:
    unavailable: dict[str, dict[str, float | None]] = {model: {} for model in MODELS}
    if jobs_file is not None:
        jobs = read_json(jobs_file)["jobs"]
    else:
        submitted = scratch() / "logs" / "jobs.tsv"
        if not submitted.is_file():
            print(f"GPU hours: n/a (no --gpu-jobs file and no {submitted})")
            return unavailable, "unavailable"
        jobs = jobs_from_submission_log(submitted)
    all_jobs = sorted({job for model in jobs.values() for ids in model.values() for job in ids})
    if not all_jobs:
        print("GPU hours: n/a (no training jobs listed)")
        return unavailable, "unavailable"
    if log_dirs:
        seconds, source = {job: elapsed_from_gpu_log(job, log_dirs) for job in all_jobs}, "gpu-monitor logs"
    elif shutil.which("sacct"):
        seconds, source = elapsed_from_sacct(all_jobs), "sacct"
    else:
        print("GPU hours: n/a (no --gpu-log-dir given and sacct is not available)")
        return unavailable, "unavailable"
    hours = {model: {m: sum(seconds[j] for j in ids) / 3600 for m, ids in methods.items()}
             for model, methods in jobs.items()}
    return {model: hours.get(model, {}) for model in MODELS}, source


# ------------------------------------------------------------------------------- build


def build(main: dict[str, Any], flops: dict[str, Any], hours: dict[str, Any]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model in MODELS:
        methods = main["models"][model]["methods"]
        base = methods["base"]["across_seeds"]
        rows = {}
        for method in METHODS:
            across = methods[method]["across_seeds"]
            delta_acc = 100 * (across["macro_avg_at_12"]["mean"] - base["macro_avg_at_12"]["mean"])
            delta_tok = across["observed_mean_completion_tokens"]["mean"] - base["observed_mean_completion_tokens"]["mean"]
            rows[method] = {
                "pflops": {**mean_sd(flops[model][method][str(s)] for s in SEEDS), "per_seed": flops[model][method]},
                "gpu_hours": hours[model].get(method),
                "delta_accuracy_pp": delta_acc,
                "delta_observed_tokens": delta_tok,
                "length_efficiency": delta_acc / (delta_tok / 1000),
                "truncation_rate": across["truncation_rate"],
                "nontruncated_mean_completion_tokens": across["nontruncated_mean_completion_tokens"],
            }
        models[model] = rows
    overall = {}
    for method in METHODS:
        hours_values = [models[m][method]["gpu_hours"] for m in MODELS]
        overall[method] = {
            "pflops": mean(models[m][method]["pflops"]["mean"] for m in MODELS),
            "gpu_hours": mean(hours_values) if None not in hours_values else None,
            "delta_accuracy_pp": mean(models[m][method]["delta_accuracy_pp"] for m in MODELS),
            "delta_observed_tokens": mean(models[m][method]["delta_observed_tokens"] for m in MODELS),
            "truncation_rate": mean(models[m][method]["truncation_rate"]["mean"] for m in MODELS),
            "length_efficiency": mean(models[m][method]["length_efficiency"] for m in MODELS),
        }
    return {"models": models, "overall": overall}


def _hours(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _tradeoff_row(method: str, row: dict[str, Any]) -> list[str]:
    flops = row["pflops"]["mean"] if isinstance(row["pflops"], dict) else row["pflops"]
    trunc = row["truncation_rate"]["mean"] if isinstance(row["truncation_rate"], dict) else row["truncation_rate"]
    return [LABELS[method], f"{flops:.2f}", _hours(row["gpu_hours"]), f"{row['delta_accuracy_pp']:+.2f}",
            f"{row['delta_observed_tokens'] / 1000:+.2f}K", f"{100 * trunc:.2f}", f"{row['length_efficiency']:.2f}"]


def tables(result: dict[str, Any]) -> dict[str, list[list[str]]]:
    distill = ("opsd", "dualopsd_direct", "dualopsd")
    out: dict[str, list[list[str]]] = {"tab:full-compute": [], "tab:tradeoff": [], "tab:tradeoff-by-scale": [],
                                       "tab:full-generation": []}
    out["tab:tradeoff"] = [_tradeoff_row(m, result["overall"][m]) for m in METHODS]
    for model in MODELS:
        rows = result["models"][model]
        for method in distill:
            row = rows[method]
            out["tab:full-compute"].append([LABELS[method], _hours(row["gpu_hours"]),
                                            pm(row["pflops"], scale=1)])
            tokens = row["nontruncated_mean_completion_tokens"]
            out["tab:full-generation"].append(
                [LABELS[method], f"{tokens['mean'] / 1000:.2f}K±{tokens['sample_sd'] / 1000:.2f}", pm(row["truncation_rate"])])
        out["tab:tradeoff-by-scale"].extend(_tradeoff_row(m, rows[m]) for m in METHODS)
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, EXP)
    parser.add_argument("--main-results", type=Path, default=None,
                        help="main_results.json from analysis.main_results (recomputed if missing)")
    parser.add_argument("--train-file", type=Path, default=None, help="training jsonl (default: $DATA_DIR/<config name>)")
    parser.add_argument("--gpu-jobs", type=Path, default=None,
                        help='JSON {"jobs": {model: {method: [slurm job ids]}}} of the training jobs '
                             "(default: seed-42 jobs from $DUALOPSD_SCRATCH/logs/jobs.tsv)")
    parser.add_argument("--gpu-log-dir", type=Path, nargs="+", default=None,
                        help="directories with GPU-monitor logs *_<job>_gpu.csv (default: query sacct)")
    args = parser.parse_args(argv)
    out = output_dir(args, EXP)
    main_path = args.main_results or results_dir(EXP) / "main_results.json"
    main = read_json(main_path) if main_path.exists() else main_results.build()
    flops = pflops_from_rollouts(args.train_file)
    hours, source = gpu_hours(args.gpu_jobs, args.gpu_log_dir)
    result = build(main, flops, hours)
    result["sources"] = {"pflops": "rollouts", "gpu_hours": source,
                         "gpu_jobs": str(args.gpu_jobs) if args.gpu_jobs else "logs/jobs.tsv"}
    print(write_json(out / "compute.json", result, overwrite=args.overwrite))
    if args.print_tables:
        print_tables(tables(result))
    return result


if __name__ == "__main__":
    main()
