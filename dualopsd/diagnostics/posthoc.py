"""Post-hoc KL, clipping and privileged-context metrics on training rollouts.

For checkpoint ``N`` of a run, re-score the Student rollouts of the last
``--rollout-window-size`` training steps up to ``N`` with the checkpoint's
Student and Teacher adapters (and the frozen base model for adaptive-Teacher
methods). Per token we record KL(p_T||p_S), KL(p_S||p_T), the fraction of
positive forward-KL mass removed by the point-wise cap ``tau``, the residual
target fit KL(q||p_T) / KL(q||p_S), and the Teacher's privileged-context JS
divergence relative to the initial model (JS retention).

Outputs ``<output-dir>/<run>_step<N>/{summary,status}.json``,
``problem_metrics.jsonl`` and ``token_metrics.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch
import yaml

from dualopsd.diagnostics.common import (
    check_output_path,
    cuda_memory_stats,
    env_dir,
    forward_completion_logits,
    load_adapters,
    read_jsonl,
    resolve_base_model,
    write_json,
)
from dualopsd.losses import js_divergence_values, privileged_residual_target_kl_values
from dualopsd.prompts import build_teacher_prompt

METHODS = ("opsd", "dualopsd_direct", "dualopsd")
FROZEN_TEACHER_METHODS = ("opsd",)
TEACHER_PROMPT_MAX_TOKENS = 15_360


def select_rollout_window(
    rollouts: list[dict[str, Any]],
    *,
    checkpoint_step: int,
    window_size: int,
    max_examples_per_step: int,
) -> tuple[list[dict[str, Any]], list[int], dict[str, int]]:
    """Rows of the ``window_size`` rollout batches that end at checkpoint ``N``.

    Rollout ``step`` is zero-based, so checkpoint ``N`` (one-based) uses rollout
    steps ``N-window_size, ..., N-1`` = training steps ``N-window_size+1, ..., N``.
    """

    rollout_steps = list(range(checkpoint_step - window_size, checkpoint_step))
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for rollout_step in rollout_steps:
        step_rows = [row for row in rollouts if int(row["step"]) == rollout_step]
        step_rows = step_rows[:max_examples_per_step]
        counts[str(rollout_step)] = len(step_rows)
        rows.extend(step_rows)
    if any(count != max_examples_per_step for count in counts.values()):
        raise ValueError(
            f"incomplete rollout window: expected {max_examples_per_step} rows "
            f"per step, found {counts}"
        )
    return rows, rollout_steps, counts


def percentile(values: list[float], q: float) -> float | None:
    finite = torch.tensor([v for v in values if math.isfinite(v)], dtype=torch.float64)
    return float(torch.quantile(finite, q).item()) if finite.numel() else None


def summarize_values(values: list[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "count": len(values),
        "finite_count": len(finite),
        "mean": mean(finite) if finite else None,
        "p50": percentile(finite, 0.5),
        "p90": percentile(finite, 0.9),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def core_metrics(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, torch.Tensor]:
    """Per-token KL(p_T||p_S) clipping diagnostics.

    With contributions ``c_v = p_T(v)(log p_T(v) - log p_S(v))``, the clipped rate
    is ``1 - sum_v min(c_v^+, tau) / sum_v c_v^+`` (NaN when no positive mass).
    """

    teacher_logp = torch.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logp = torch.log_softmax(student_logits, dim=-1, dtype=torch.float32)
    teacher_prob = teacher_logp.exp()
    student_prob = student_logp.exp()
    contributions = teacher_prob * (teacher_logp - student_logp)
    positive = contributions.clamp_min(0)
    positive_mass = positive.sum(dim=-1)
    kept_positive_mass = positive.clamp(max=float(threshold)).sum(dim=-1)
    positive_valid = positive_mass > 0
    clipped_rate = torch.where(
        positive_valid,
        1 - kept_positive_mass / positive_mass,
        torch.full_like(positive_mass, torch.nan),
    )
    clipped = contributions > float(threshold)
    return {
        "positive_kl_clipped_rate": clipped_rate,
        "positive_kl_valid": positive_valid.float(),
        "teacher_to_student_kl": contributions.sum(dim=-1),
        "student_to_teacher_kl": (student_prob * (student_logp - teacher_logp)).sum(dim=-1),
        "positive_kl_mass": positive_mass,
        "clipped_vocab_fraction": clipped.float().mean(dim=-1),
        "clipped_teacher_probability_mass": (teacher_prob * clipped).sum(dim=-1),
        "clipped_student_probability_mass": (student_prob * clipped).sum(dim=-1),
    }


def privileged_metrics(
    current_privileged_logits: torch.Tensor,
    current_unprivileged_logits: torch.Tensor,
    student_logits: torch.Tensor,
    initial_privileged_logits: torch.Tensor,
    initial_unprivileged_logits: torch.Tensor,
    *,
    beta: float,
) -> dict[str, torch.Tensor]:
    """Residual-target fit and privileged-context JS of the current/initial Teacher."""

    mask = torch.ones(
        current_privileged_logits.shape[:-1],
        dtype=torch.bool,
        device=current_privileged_logits.device,
    )
    target_fit, target_to_student = privileged_residual_target_kl_values(
        current_privileged_logits,
        student_logits,
        initial_privileged_logits,
        initial_unprivileged_logits,
        mask,
        beta=beta,
    )
    return {
        "target_fit_kl": target_fit,
        "target_to_student_kl": target_to_student,
        "current_privileged_js": js_divergence_values(
            current_privileged_logits, current_unprivileged_logits, mask
        ),
        "initial_privileged_js": js_divergence_values(
            initial_privileged_logits, initial_unprivileged_logits, mask
        ),
    }


def problem_means(
    token_metrics: dict[str, list[float]], *, with_retention: bool
) -> dict[str, float]:
    """Finite-position mean of each metric; JS retention = current / initial mean."""

    result: dict[str, float] = {}
    for key, values in token_metrics.items():
        finite = [float(v) for v in values if math.isfinite(float(v))]
        result[key] = mean(finite) if finite else math.nan
    if with_retention:
        initial = result["initial_privileged_js"]
        current = result["current_privileged_js"]
        valid = math.isfinite(initial) and math.isfinite(current) and initial > 0
        result["js_retention"] = current / initial if valid else math.nan
    return result


def score_example(
    model: Any,
    prompt_ids: dict[str, list[int]],
    completion_ids: list[int],
    *,
    adaptive_teacher: bool,
    threshold: float,
    beta: float,
) -> dict[str, torch.Tensor]:
    student_logits = forward_completion_logits(
        model, prompt_ids["student"], completion_ids, adapter="student"
    )
    teacher_logits = forward_completion_logits(
        model, prompt_ids["teacher"], completion_ids, adapter="teacher"
    )
    metrics = core_metrics(teacher_logits, student_logits, threshold=threshold)
    if adaptive_teacher:
        metrics.update(
            privileged_metrics(
                teacher_logits,
                forward_completion_logits(
                    model, prompt_ids["student"], completion_ids, adapter="teacher"
                ),
                student_logits,
                forward_completion_logits(
                    model, prompt_ids["teacher"], completion_ids, adapter=None
                ),
                forward_completion_logits(
                    model, prompt_ids["student"], completion_ids, adapter=None
                ),
                beta=beta,
            )
        )
    return metrics


def resolve_train_file(train_file: Path | None, checkpoint: Path) -> Path:
    """Training data of the run: ``--train-file`` or ``$DATA_DIR/<basename of data.train_file>``."""

    if train_file is not None:
        return train_file
    config = yaml.safe_load((checkpoint / "config.yaml").read_text(encoding="utf-8"))
    recorded = Path(config["data"]["train_file"])
    data_dir = env_dir("DATA_DIR")
    for candidate in ((data_dir / recorded.name) if data_dir else None, recorded):
        if candidate is not None and candidate.exists():
            return candidate
    raise FileNotFoundError(f"training data {recorded.name} not found; pass --train-file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dualopsd posthoc",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="runs/<exp>/<model>/<run> with checkpoint-N/{student,teacher} and rollouts.jsonl")
    parser.add_argument("--step", type=int, required=True, help="checkpoint step N")
    parser.add_argument("--method", choices=METHODS, required=True,
                        help="opsd keeps a frozen Teacher: privileged metrics are skipped")
    parser.add_argument("--base-model", type=Path,
                        help="default: $MODEL_<MODEL> named by the run directory")
    parser.add_argument("--train-file", type=Path,
                        help="default: $DATA_DIR/<basename of the checkpoint's data.train_file>")
    parser.add_argument("--threshold", type=float, default=1.0e-6,
                        help="point-wise cap tau on forward-KL contributions")
    parser.add_argument("--residual-beta", type=float, default=0.7,
                        help="beta of the residual target q (the paper uses 0.7 for every Qwen3-4B method)")
    parser.add_argument("--rollout-window-size", type=int, default=5,
                        help="consecutive rollout batches ending at the checkpoint")
    parser.add_argument("--max-examples", type=int, default=32, help="rollout rows per batch")
    parser.add_argument("--limit", type=int, help="score only the first LIMIT window rows")
    parser.add_argument("--smoke", action="store_true", help="score 2 rows only")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="results go to <output-dir>/<run>_step<N>/")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.limit = 2
    if (
        args.threshold <= 0
        or args.max_examples < 1
        or not 1 <= args.rollout_window_size <= args.step
        or (args.limit is not None and args.limit < 1)
    ):
        raise ValueError(
            "threshold, max-examples and limit must be positive; the rollout "
            "window must lie within training steps 1..N"
        )
    adaptive_teacher = args.method not in FROZEN_TEACHER_METHODS
    run_dir: Path = args.run_dir
    checkpoint = run_dir / f"checkpoint-{args.step}"
    examples = {
        str(row["id"]): row
        for row in read_jsonl(resolve_train_file(args.train_file, checkpoint))
    }
    rows, rollout_steps, rollout_step_counts = select_rollout_window(
        read_jsonl(run_dir / "rollouts.jsonl"),
        checkpoint_step=args.step,
        window_size=args.rollout_window_size,
        max_examples_per_step=args.max_examples,
    )
    example_ids = [str(row["example_id"]) for row in rows]
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("rollout window contains duplicate example IDs")
    if args.limit is not None:
        rows = rows[: args.limit]

    output_dir = args.output_dir / f"{run_dir.name}_step{args.step}"
    check_output_path(output_dir, overwrite=args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    status: dict[str, Any] = {
        "status": "running",
        "method": args.method,
        "run_dir": str(run_dir),
        "checkpoint_step": args.step,
        "rollout_steps": rollout_steps,
        "rollout_training_steps": [step + 1 for step in rollout_steps],
        "rollout_window_size": args.rollout_window_size,
        "rollout_step_counts": rollout_step_counts,
        "threshold": args.threshold,
        "residual_beta": args.residual_beta,
        "requested_examples": len(rows),
        "processed_examples": 0,
        "skipped_retokenization_mismatch": [],
        "completion_token_source_counts": {"stored_ids": 0, "retokenized_text": 0},
    }
    write_json(status_path, status)

    adapters = {"student": checkpoint / "student", "teacher": checkpoint / "teacher"}
    tokenizer, model = load_adapters(resolve_base_model(args.base_model, run_dir), adapters)
    all_metrics: dict[str, list[float]] = defaultdict(list)
    retention_values: list[float] = []
    per_problem_metrics: list[dict[str, float]] = []
    start = time.monotonic()
    with (output_dir / "token_metrics.jsonl").open("w", encoding="utf-8") as token_handle, (
        output_dir / "problem_metrics.jsonl"
    ).open("w", encoding="utf-8") as problem_handle:
        for ordinal, row in enumerate(rows, start=1):
            # Rollout rows without token ids are re-tokenized; rows whose
            # length disagrees with the recorded completion length are skipped.
            if row.get("completion_token_ids") is not None:
                completion_ids = [int(token) for token in row["completion_token_ids"]]
                status["completion_token_source_counts"]["stored_ids"] += 1
            else:
                completion_ids = tokenizer(str(row["output"]), add_special_tokens=False)["input_ids"]
                status["completion_token_source_counts"]["retokenized_text"] += 1
            if len(completion_ids) != int(row["resp_token"]):
                status["skipped_retokenization_mismatch"].append({
                    "example_id": row["example_id"],
                    "stored_tokens": int(row["resp_token"]),
                    "retokenized_tokens": len(completion_ids),
                })
                write_json(status_path, status)
                continue
            example = examples[str(row["example_id"])]
            teacher_prompt = build_teacher_prompt(
                tokenizer,
                str(example["problem"]),
                str(example["reference"]),
                thinking=False,
                max_tokens=TEACHER_PROMPT_MAX_TOKENS,
            )
            prompt_ids = {
                "student": tokenizer(str(row["input"]), add_special_tokens=False)["input_ids"],
                "teacher": tokenizer(teacher_prompt, add_special_tokens=False)["input_ids"],
            }
            metrics = score_example(
                model,
                prompt_ids,
                completion_ids,
                adaptive_teacher=adaptive_teacher,
                threshold=args.threshold,
                beta=args.residual_beta,
            )
            cpu_metrics = {
                key: value.detach().float().cpu().tolist() for key, value in metrics.items()
            }
            del metrics
            for key, values in cpu_metrics.items():
                all_metrics[key].extend(float(value) for value in values)
            per_problem = problem_means(cpu_metrics, with_retention=adaptive_teacher)
            if adaptive_teacher:
                retention_values.append(per_problem["js_retention"])
            per_problem_metrics.append(per_problem)
            base_record = {
                "method": args.method,
                "checkpoint_step": args.step,
                "rollout_step": int(row["step"]),
                "example_id": row["example_id"],
            }
            problem_handle.write(json.dumps(
                {**base_record, "completion_tokens": len(completion_ids), "metrics": per_problem},
                sort_keys=True,
            ) + "\n")
            tokens = tokenizer.convert_ids_to_tokens(completion_ids)
            for index, token_id in enumerate(completion_ids):
                token_handle.write(json.dumps({
                    **base_record,
                    "token_index": index,
                    "token_id": token_id,
                    "token": tokens[index],
                    "metrics": {key: values[index] for key, values in cpu_metrics.items()},
                }, sort_keys=True) + "\n")
            token_handle.flush()
            problem_handle.flush()
            status["processed_examples"] += 1
            status["last_example_id"] = row["example_id"]
            status["elapsed_seconds"] = time.monotonic() - start
            cuda_memory_stats(status)
            write_json(status_path, status)
            print(
                f"method={args.method} step={args.step} example={ordinal}/{len(rows)} "
                f"tokens={len(completion_ids)} elapsed={status['elapsed_seconds']:.1f}s",
                flush=True,
            )

    problem_keys = sorted({key for metrics in per_problem_metrics for key in metrics})
    summary = {
        **status,
        "status": "complete",
        "elapsed_seconds": time.monotonic() - start,
        "retokenization_coverage": status["processed_examples"] / len(rows),
        "token_metrics": {
            key: summarize_values(values) for key, values in sorted(all_metrics.items())
        },
        "problem_metric_means": {
            key: summarize_values([m[key] for m in per_problem_metrics if key in m])
            for key in problem_keys
        },
        "problem_only_metrics": (
            {"js_retention": summarize_values(retention_values)} if adaptive_teacher else {}
        ),
    }
    if adaptive_teacher:
        initial_total = sum(all_metrics["initial_privileged_js"])
        summary["js_retention_ratio_of_token_means"] = (
            sum(all_metrics["current_privileged_js"]) / initial_total
            if initial_total > 0
            else None
        )
    write_json(status_path, summary)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
