"""Recurrence R_k of top-k vocabulary contributions between checkpoints 20 and 100.

All checkpoints score the same canonical responses (OPSD seed-42 step-20 Student
generations on the 500-problem diagnostic set, deterministically re-tokenized).
At each position we keep the top-20 contributions of forward KL(p_T||p_S) and
reverse KL(p_S||p_T) and compare, for k in {1,5,10,20}, the step-20 and step-100
sets with Jaccard overlap for three signals:

- ``positive_clipped_t2s``: forward contributions above the cap tau,
- ``positive_t2s``: positive forward contributions,
- ``positive_s2t``: positive reverse contributions.

Values are averaged within each problem (positions with an empty union are
dropped) and then across problems (``problem_mean``); ``token_weighted_mean``
pools positions. The random-pair baseline pairs problem i at step 20 with
problem perm[i] at step 100 for a fixed derangement. OPSD keeps its frozen
step-20 Teacher at step 100.

Outputs ``<output-dir>/<run>.json`` and ``<output-dir>/<run>_status.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch

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
from dualopsd.prompts import build_student_prompt, build_teacher_prompt

METHODS = ("opsd", "dualopsd_direct", "dualopsd")
FROZEN_TEACHER_METHODS = ("opsd",)
SIGNALS = ("positive_clipped_t2s", "positive_t2s", "positive_s2t")
K_VALUES = (1, 5, 10, 20)
TOP_K = 20
STEPS = (20, 100)
TEACHER_PROMPT_MAX_TOKENS = 8192
DIAGNOSTIC_SET = "heldout_500.jsonl"
CANONICAL_RESPONSES = "generations/mismatch/qwen3-4b/opsd_seed42_step20/student.jsonl"

Ranking = tuple[torch.Tensor, torch.Tensor]  # (token ids, contribution values)


def bootstrap_ci(values: list[float], *, samples: int = 10_000, seed: int = 42) -> list[float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return [math.nan, math.nan]
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choice(finite) for _ in finite) for _ in range(samples))
    return [estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]]


def deterministic_derangement(size: int, *, seed: int = 42) -> list[int]:
    """Sattolo permutation (a single cycle, hence no fixed points)."""

    if size < 2:
        raise ValueError("derangement requires at least two rows")
    permutation = list(range(size))
    rng = random.Random(seed)
    for right in range(size - 1, 0, -1):
        left = rng.randrange(right)
        permutation[left], permutation[right] = permutation[right], permutation[left]
    return permutation


def recurrence_values(
    left_ids: torch.Tensor,
    left_values: torch.Tensor,
    right_ids: torch.Tensor,
    right_values: torch.Tensor,
    *,
    k: int,
    cutoff: float,
) -> torch.Tensor:
    """Per-position Jaccard of the top-k ids whose value exceeds ``cutoff``.

    Positions are truncated to the shorter sequence; empty unions give NaN.
    """

    positions = min(left_ids.shape[0], right_ids.shape[0])
    left_ids = left_ids[:positions, :k]
    right_ids = right_ids[:positions, :k]
    left_valid = left_values[:positions, :k] > cutoff
    right_valid = right_values[:positions, :k] > cutoff
    matches = (
        (left_ids[:, :, None] == right_ids[:, None, :])
        & left_valid[:, :, None]
        & right_valid[:, None, :]
    )
    intersection = matches.any(dim=2).sum(dim=1)
    union = left_valid.sum(dim=1) + right_valid.sum(dim=1) - intersection
    result = torch.full((positions,), torch.nan, dtype=torch.float32)
    valid = union > 0
    result[valid] = intersection[valid].float() / union[valid].float()
    return result


def ranked_contributions(
    teacher_logits: torch.Tensor, student_logits: torch.Tensor, *, k: int = TOP_K
) -> dict[str, Ranking]:
    """Top-k forward (``teacher``) and reverse (``student``) KL contributions per position."""

    teacher_logp = torch.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logp = torch.log_softmax(student_logits, dim=-1, dtype=torch.float32)
    contributions = {
        "teacher": teacher_logp.exp() * (teacher_logp - student_logp),
        "student": student_logp.exp() * (student_logp - teacher_logp),
    }
    result = {}
    for name, values in contributions.items():
        top_values, top_ids = torch.topk(values, k=k, dim=-1)
        result[name] = (
            top_ids.detach().to(dtype=torch.int32, device="cpu"),
            top_values.detach().to(dtype=torch.float32, device="cpu"),
        )
    return result


def checkpoint_adapters(run_dir: Path, method: str) -> tuple[dict[str, Path], dict[int, dict[str, str]]]:
    """Adapter paths to load and the (student, teacher) adapter names per step."""

    early, late = STEPS
    paths = {
        f"student_{early}": run_dir / f"checkpoint-{early}/student",
        f"teacher_{early}": run_dir / f"checkpoint-{early}/teacher",
        f"student_{late}": run_dir / f"checkpoint-{late}/student",
    }
    late_teacher = f"teacher_{early}"
    if method not in FROZEN_TEACHER_METHODS:
        late_teacher = f"teacher_{late}"
        paths[late_teacher] = run_dir / f"checkpoint-{late}/teacher"
    roles = {
        early: {"student": f"student_{early}", "teacher": f"teacher_{early}"},
        late: {"student": f"student_{late}", "teacher": late_teacher},
    }
    return paths, roles


def checkpoint_rankings(
    model: Any,
    prompt_ids: dict[str, list[int]],
    completion_ids: list[int],
    roles: dict[str, str],
) -> dict[str, Ranking]:
    student_logits = forward_completion_logits(
        model, prompt_ids["student"], completion_ids, adapter=roles["student"]
    )
    teacher_logits = forward_completion_logits(
        model, prompt_ids["teacher"], completion_ids, adapter=roles["teacher"]
    )
    return ranked_contributions(teacher_logits, student_logits)


def summarize_recurrence(
    rankings: list[dict[int, dict[str, Ranking]]],
    *,
    threshold: float,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """R_k per signal and k: problem mean, token-weighted mean, coverage, random-pair baseline."""

    early, late = STEPS
    permutation = deterministic_derangement(len(rankings), seed=42)
    signal_source = {
        "positive_clipped_t2s": ("teacher", threshold),
        "positive_t2s": ("teacher", 0.0),
        "positive_s2t": ("student", 0.0),
    }
    output: dict[str, Any] = {}
    for signal_index, signal in enumerate(SIGNALS):
        source, cutoff = signal_source[signal]
        signal_output: dict[str, Any] = {}
        for k in K_VALUES:
            problem_values: list[float] = []
            shuffled_values: list[float] = []
            position_values: list[float] = []
            coverage_early: list[float] = []
            coverage_late: list[float] = []
            for problem_index, row in enumerate(rankings):
                left_ids, left_scores = row[early][source]
                right_ids, right_scores = row[late][source]
                values = recurrence_values(
                    left_ids, left_scores, right_ids, right_scores, k=k, cutoff=cutoff
                )
                finite = values[torch.isfinite(values)].tolist()
                problem_values.append(mean(finite) if finite else math.nan)
                position_values.extend(float(value) for value in finite)
                coverage_early.append(float((left_scores[:, :k] > cutoff).any(dim=1).float().mean()))
                coverage_late.append(float((right_scores[:, :k] > cutoff).any(dim=1).float().mean()))

                other_ids, other_scores = rankings[permutation[problem_index]][late][source]
                shuffled = recurrence_values(
                    left_ids, left_scores, other_ids, other_scores, k=k, cutoff=cutoff
                )
                shuffled_finite = shuffled[torch.isfinite(shuffled)].tolist()
                shuffled_values.append(mean(shuffled_finite) if shuffled_finite else math.nan)
            finite_problem = [v for v in problem_values if math.isfinite(v)]
            finite_shuffle = [v for v in shuffled_values if math.isfinite(v)]
            signal_output[str(k)] = {
                "problem_mean": mean(finite_problem),
                "problem_bootstrap_ci95": bootstrap_ci(
                    finite_problem, samples=bootstrap_samples, seed=42 + signal_index * 100 + k
                ),
                "token_weighted_mean": mean(position_values),
                "problems": len(finite_problem),
                "compared_positions": len(position_values),
                f"step{early}_nonempty_position_coverage": mean(coverage_early),
                f"step{late}_nonempty_position_coverage": mean(coverage_late),
                "shuffled_problem_mean": mean(finite_shuffle),
                "shuffled_problem_bootstrap_ci95": bootstrap_ci(
                    finite_shuffle, samples=bootstrap_samples, seed=142 + signal_index * 100 + k
                ),
            }
        output[signal] = signal_output
    return output


def build_parser() -> argparse.ArgumentParser:
    scratch = env_dir("DUALOPSD_SCRATCH")
    data_dir = env_dir("DATA_DIR")
    parser = argparse.ArgumentParser(
        prog="python -m dualopsd recurrence",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="runs/<exp>/<model>/<run> with checkpoint-20 and checkpoint-100")
    parser.add_argument("--method", choices=METHODS, required=True,
                        help="opsd reuses the step-20 Teacher at step 100")
    parser.add_argument("--canonical-responses", type=Path,
                        default=scratch / CANONICAL_RESPONSES if scratch else None,
                        help="default: $DUALOPSD_SCRATCH/" + CANONICAL_RESPONSES)
    parser.add_argument("--dataset", type=Path,
                        default=data_dir / DIAGNOSTIC_SET if data_dir else None,
                        help="500-problem diagnostic set (default: $DATA_DIR/" + DIAGNOSTIC_SET + ")")
    parser.add_argument("--base-model", type=Path,
                        help="default: $MODEL_<MODEL> named by the run directory")
    parser.add_argument("--threshold", type=float, default=1.0e-6,
                        help="cap tau defining the clipped forward signal")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--limit", type=int, help="score only the first LIMIT responses")
    parser.add_argument("--smoke", action="store_true", help="score 2 responses only")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.limit = 2
    if args.canonical_responses is None or args.dataset is None:
        raise ValueError("pass --canonical-responses and --dataset (or set DUALOPSD_SCRATCH / DATA_DIR)")
    if args.threshold <= 0 or (args.limit is not None and args.limit < 2):
        raise ValueError("threshold must be positive and limit at least 2")

    run_dir: Path = args.run_dir
    canonical = read_jsonl(args.canonical_responses)
    dataset = {str(row["id"]): row for row in read_jsonl(args.dataset)}
    canonical_ids = {str(row["id"]) for row in canonical}
    if args.limit is not None:
        # Truncated runs (smoke or --limit) may use a partial canonical set.
        if not canonical_ids <= set(dataset):
            raise ValueError("canonical response IDs are not in the diagnostic dataset")
    elif len(canonical) != len(dataset) or canonical_ids != set(dataset):
        raise ValueError("canonical response IDs do not match the diagnostic dataset")
    if args.limit is not None:
        canonical = canonical[: args.limit]

    output_path = args.output_dir / f"{run_dir.name}.json"
    status_path = args.output_dir / f"{run_dir.name}_status.json"
    for path in (output_path, status_path):
        check_output_path(path, overwrite=args.overwrite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "status": "running",
        "method": args.method,
        "run_dir": str(run_dir),
        "canonical_responses": str(args.canonical_responses),
        "canonical_response_sha256": hashlib.sha256(args.canonical_responses.read_bytes()).hexdigest(),
        "requested_problems": len(canonical),
        "processed_problems": 0,
        "threshold": args.threshold,
        "k_values": list(K_VALUES),
        "completion_token_source": "deterministic retokenization of canonical response text",
    }
    write_json(status_path, status)

    adapter_paths, roles = checkpoint_adapters(run_dir, args.method)
    tokenizer, model = load_adapters(resolve_base_model(args.base_model, run_dir), adapter_paths)
    rankings = []
    start = time.monotonic()
    for ordinal, response_row in enumerate(canonical, start=1):
        row = dataset[str(response_row["id"])]
        student_prompt = build_student_prompt(tokenizer, str(row["problem"]), thinking=False)
        teacher_prompt = build_teacher_prompt(
            tokenizer,
            str(row["problem"]),
            str(row["reference"]),
            thinking=False,
            max_tokens=TEACHER_PROMPT_MAX_TOKENS,
        )
        prompt_ids = {
            "student": tokenizer(student_prompt, add_special_tokens=False)["input_ids"],
            "teacher": tokenizer(teacher_prompt, add_special_tokens=False)["input_ids"],
        }
        completion_ids = tokenizer(str(response_row["response"]), add_special_tokens=False)["input_ids"]
        if not completion_ids:
            raise ValueError(f"empty canonical completion for {response_row['id']}")
        rankings.append({
            step: checkpoint_rankings(model, prompt_ids, completion_ids, roles[step])
            for step in STEPS
        })
        status["processed_problems"] = ordinal
        status["last_problem_id"] = response_row["id"]
        status["elapsed_seconds"] = time.monotonic() - start
        cuda_memory_stats(status)
        write_json(status_path, status)
        print(
            f"method={args.method} problem={ordinal}/{len(canonical)} "
            f"tokens={len(completion_ids)} elapsed={status['elapsed_seconds']:.1f}s",
            flush=True,
        )

    result = {
        **status,
        "status": "complete",
        "elapsed_seconds": time.monotonic() - start,
        "metrics": summarize_recurrence(
            rankings, threshold=args.threshold, bootstrap_samples=args.bootstrap_samples
        ),
        "shuffle_baseline": (
            "step-20 problem is paired with a deterministic derangement of "
            "step-100 problems; token positions are truncated to the shorter response"
        ),
    }
    write_json(output_path, result)
    write_json(status_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
