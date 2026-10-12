"""Benchmark avg@k evaluation (AIME24, AIME25, HMMT25) with prompt-inclusive token limits.

Writes ``<output-dir>/<benchmark>/samples.jsonl`` and ``summary.json``. Each finished
problem is checkpointed under ``<output-dir>/<benchmark>/.parts`` so an interrupted job
resumes without regenerating completed problems.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from dualopsd.evaluation import check_output, read_jsonl, shutdown_llm, write_json, write_jsonl_atomic

BENCHMARKS = ("aime24", "aime25", "hmmt25")
PROMPT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def benchmark_path(bench_dir: str | Path, name: str) -> Path:
    return Path(bench_dir) / f"{name}.jsonl"


def percentile(values: list[int], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-benchmark metrics. Keys keep the ``_at_12`` suffix for every sample count."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["problem_id"], []).append(record)
    lengths = [r["completion_tokens"] for r in records if not r["truncated"]]
    return {
        "num_problems": len(grouped),
        "num_generations": len(records),
        "avg_at_12": mean(r["correct"] for r in records),
        "pass_at_12": mean(any(r["correct"] for r in group) for group in grouped.values()),
        "rule_based_avg_at_12": mean(r["rule_based_correct"] for r in records),
        "rule_based_pass_at_12": mean(any(r["rule_based_correct"] for r in group) for group in grouped.values()),
        "verifier_disagreement_rate": mean(r["correct"] != r["rule_based_correct"] for r in records),
        "truncation_rate": mean(r["truncated"] for r in records),
        "nontruncated_mean_completion_tokens": mean(lengths) if lengths else None,
        "nontruncated_p99_completion_tokens": percentile(lengths, 0.99),
        "nontruncated_p100_completion_tokens": max(lengths) if lengths else None,
    }


def generate_incrementally(
    llm: Any,
    prompts: list[str],
    sampling: list[Any],
    on_finished: Callable[[int, Any], None],
) -> None:
    """Submit all prompts at once (same vLLM schedule as ``llm.generate``) and report each finished one."""
    if not prompts:
        return
    from tqdm.auto import tqdm

    request_ids: list[str] = []
    original_add_request = llm.llm_engine.add_request

    def capture_add_request(request_id: str, *args: Any, **kwargs: Any) -> Any:
        request_ids.append(str(request_id))
        return original_add_request(request_id, *args, **kwargs)

    llm.llm_engine.add_request = capture_add_request
    try:
        llm._validate_and_add_requests(
            prompts=prompts, params=sampling, use_tqdm=True, lora_request=None, priority=None
        )
    finally:
        llm.llm_engine.add_request = original_add_request
    if len(request_ids) != len(prompts) or len(set(request_ids)) != len(prompts):
        raise RuntimeError("vLLM request-ID capture did not cover every prompt")
    offsets = {request_id: offset for offset, request_id in enumerate(request_ids)}
    progress = tqdm(total=len(prompts), desc="Finished prompts", dynamic_ncols=True)
    finished = 0
    while llm.llm_engine.has_unfinished_requests():
        for output in llm.llm_engine.step():
            if not output.finished:
                continue
            request_id = str(output.request_id)
            if request_id not in offsets:
                raise RuntimeError(f"unexpected vLLM request id: {output.request_id}")
            on_finished(offsets[request_id], output)
            finished += 1
            progress.update(1)
    progress.close()
    if finished != len(prompts):
        raise RuntimeError(f"vLLM finished {finished} prompt requests, expected {len(prompts)}")


def records_for_request(
    *, benchmark: str, row_index: int, row: dict[str, Any], request: Any, prompt_length: int, cap: int
) -> list[dict[str, Any]]:
    from dualopsd.verifiers.math import verify_math
    from dualopsd.verifiers.math_official import verify_math_official

    reference = str(row["answer"])
    records = []
    for sample_index, candidate in enumerate(request.outputs):
        completion_length = len(candidate.token_ids)
        records.append(
            {
                "dataset": benchmark,
                "problem_id": str(row.get("id", row_index)),
                "sample_index": sample_index,
                "problem": row["problem"],
                "answer": reference,
                "response": candidate.text,
                "correct": verify_math_official(candidate.text, reference),
                "rule_based_correct": verify_math(candidate.text, reference),
                "finish_reason": candidate.finish_reason,
                "truncated": candidate.finish_reason == "length" or completion_length >= cap,
                "prompt_tokens": prompt_length,
                "completion_tokens": completion_length,
                "total_tokens": prompt_length + completion_length,
                "generation_cap": cap,
            }
        )
    return records


def read_problem_part(path: Path, *, benchmark: str, problem_id: str, num_samples: int) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    if len(records) != num_samples:
        raise ValueError(f"{path}: expected {num_samples} records")
    if {record["dataset"] for record in records} != {benchmark}:
        raise ValueError(f"{path}: benchmark mismatch")
    if {str(record["problem_id"]) for record in records} != {problem_id}:
        raise ValueError(f"{path}: problem mismatch")
    if sorted(int(record["sample_index"]) for record in records) != list(range(num_samples)):
        raise ValueError(f"{path}: sample-index mismatch")
    return records


def write_problem_part(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically checkpoint one completed problem."""
    write_jsonl_atomic(path, records)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="dualopsd eval", description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="merged HF model directory or hub id")
    parser.add_argument("--output-dir", required=True, help="writes <output-dir>/<benchmark>/samples.jsonl")
    parser.add_argument("--method", default=None, help="label stored in summary.json (default: output dir name)")
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--bench-dir", default=os.environ.get("BENCH_DIR"), help="default: $BENCH_DIR")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--max-total-tokens", type=int, default=32768, help="prompt + completion budget")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42, help="generation seed")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--log-vllm-stats", action="store_true", help="periodic vLLM engine stats (KV-cache use)")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N problems")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="2 problems, 1 sample, 256 new tokens")
    args = parser.parse_args(argv)
    if args.smoke:
        args.limit, args.num_samples, args.max_new_tokens, args.max_total_tokens = 2, 1, 256, 4096
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not args.bench_dir:
        parser.error("--bench-dir is required when $BENCH_DIR is unset")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from dualopsd.prompts import render_chat

    output_root = Path(args.output_dir)
    method = args.method or output_root.name
    targets = {name: output_root / name / "samples.jsonl" for name in args.benchmarks}
    for name, path in targets.items():
        check_output(path, overwrite=args.overwrite)
        check_output(path.with_name("summary.json"), overwrite=args.overwrite)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        max_model_len=args.max_total_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        disable_log_stats=not args.log_vllm_stats,
    )
    # vLLM 0.11 has no public LLM.close(); register shutdown after EngineCore creation so
    # exceptional exits also stop it before multiprocessing joins it.
    atexit.register(shutdown_llm, llm)
    settings = {
        "thinking": False,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "max_total_tokens": args.max_total_tokens,
        "max_new_tokens": args.max_new_tokens,
        "max_num_seqs": args.max_num_seqs,
        "vllm_stats_enabled": args.log_vllm_stats,
        "seed": args.seed,
        "limit": args.limit,
    }
    for name in args.benchmarks:
        rows = read_jsonl(benchmark_path(args.bench_dir, name), args.limit)
        prompts = [
            render_chat(tokenizer, [{"role": "user", "content": f"{row['problem']}{PROMPT_SUFFIX}"}], thinking=False)
            for row in rows
        ]
        prompt_lengths = [len(tokenizer.encode(prompt)) for prompt in prompts]
        caps = [args.max_total_tokens - length for length in prompt_lengths]
        if args.max_new_tokens is not None:
            caps = [min(cap, args.max_new_tokens) for cap in caps]
        if min(caps) < 1:
            raise ValueError(f"{name} has a prompt over the total-token limit")
        sampling = [
            SamplingParams(
                n=args.num_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=0.0,
                presence_penalty=0.0,
                max_tokens=cap,
                seed=args.seed,
            )
            for cap in caps
        ]
        parts_dir = output_root / name / ".parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        records_by_row: dict[int, list[dict[str, Any]]] = {}
        pending = []
        for row_index, row in enumerate(rows):
            part = parts_dir / f"{row_index:04d}.jsonl"
            if part.is_file():
                records_by_row[row_index] = read_problem_part(
                    part, benchmark=name, problem_id=str(row.get("id", row_index)), num_samples=args.num_samples
                )
            else:
                pending.append(row_index)

        def checkpoint(offset: int, request: Any) -> None:
            row_index = pending[offset]
            records = records_for_request(
                benchmark=name,
                row_index=row_index,
                row=rows[row_index],
                request=request,
                prompt_length=prompt_lengths[row_index],
                cap=caps[row_index],
            )
            if len(records) != args.num_samples:
                raise RuntimeError(f"{name}/{row_index}: expected {args.num_samples} samples, found {len(records)}")
            write_problem_part(parts_dir / f"{row_index:04d}.jsonl", records)
            records_by_row[row_index] = records

        generate_incrementally(llm, [prompts[i] for i in pending], [sampling[i] for i in pending], checkpoint)
        if set(records_by_row) != set(range(len(rows))):
            raise RuntimeError(f"{name}: incomplete problem checkpoints")
        records = [record for row_index in range(len(rows)) for record in records_by_row[row_index]]
        write_jsonl_atomic(targets[name], records)
        summary = {
            "method": method,
            "model": args.model,
            "settings": settings,
            "datasets": {name: summarize(records)},
        }
        write_json(targets[name].with_name("summary.json"), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    shutdown_llm(llm)
    atexit.unregister(shutdown_llm)
    print("vLLM EngineCore shutdown complete", flush=True)


if __name__ == "__main__":
    main()
