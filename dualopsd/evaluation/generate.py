"""Generate one teacher or student response per problem of the held-out set with a LoRA adapter.

The teacher sees the privileged reference solution, the student only the problem (the
same prompts as in training). Problem ``i`` is sampled with seed ``seed + i`` for both
policies. Correctness is assigned afterwards by ``dualopsd judge``.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
from statistics import mean

from dualopsd.evaluation import check_output, read_jsonl, shutdown_llm, write_json, write_jsonl_atomic

TEMPERATURE, TOP_P, TOP_K = 1.0, 0.95, 20


def default_dataset() -> str | None:
    data_dir = os.environ.get("DATA_DIR")
    return str(Path(data_dir) / "heldout_500.jsonl") if data_dir else None


def build_prompts(tokenizer, rows: list[dict], policy: str) -> list[str]:
    from dualopsd.prompts import render_chat, upstream_student_messages, upstream_teacher_messages

    if policy == "teacher":
        messages = [upstream_teacher_messages(row["problem"], row["reference"]) for row in rows]
    else:
        messages = [upstream_student_messages(row["problem"]) for row in rows]
    return [render_chat(tokenizer, message, thinking=False) for message in messages]


def summary_path(output: Path) -> Path:
    return output.with_name(output.stem + "_summary.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="dualopsd generate", description=__doc__.splitlines()[0])
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True, help="checkpoint-N/{teacher,student} adapter dir")
    parser.add_argument("--policy", choices=("teacher", "student"), required=True)
    parser.add_argument("--dataset", default=default_dataset(), help="default: $DATA_DIR/heldout_500.jsonl")
    parser.add_argument("--output", required=True, help="e.g. <generations>/<run>_step100/student.jsonl")
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="2 problems, 256 new tokens")
    args = parser.parse_args(argv)
    if args.smoke:
        args.limit, args.max_completion_tokens = 2, 256
    if not args.dataset:
        parser.error("--dataset is required when $DATA_DIR is unset")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    output = check_output(args.output, overwrite=args.overwrite)
    check_output(summary_path(output), overwrite=args.overwrite)
    rows = read_jsonl(args.dataset, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
    prompts = build_prompts(tokenizer, rows, args.policy)
    prompt_lengths = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts]
    if max(prompt_lengths) > args.max_model_len - args.max_completion_tokens:
        raise ValueError("a rendered prompt exceeds max_model_len - max_completion_tokens")

    llm = LLM(
        model=args.base_model,
        tokenizer=args.base_model,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_loras=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        seed=args.seed,
    )
    atexit.register(shutdown_llm, llm)
    sampling = [
        SamplingParams(
            n=1,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_tokens=args.max_completion_tokens,
            seed=args.seed + index,
        )
        for index in range(len(rows))
    ]
    lora = LoRARequest(args.policy, 1, str(Path(args.adapter).resolve()))
    outputs = llm.generate(prompts, sampling_params=sampling, lora_request=lora, use_tqdm=True)
    records = []
    for index, (row, prompt_length, request) in enumerate(zip(rows, prompt_lengths, outputs)):
        candidate = request.outputs[0]
        completion_tokens = len(candidate.token_ids)
        records.append(
            {
                "id": row["id"],
                "policy": args.policy,
                "problem": row["problem"],
                "reference": row["reference"],
                "answer": row["answer"],
                "response": candidate.text,
                "has_boxed_answer": "\\boxed{" in candidate.text,
                "finish_reason": candidate.finish_reason,
                "truncated": candidate.finish_reason == "length" or completion_tokens >= args.max_completion_tokens,
                "prompt_tokens": prompt_length,
                "completion_tokens": completion_tokens,
                "sampling_seed": args.seed + index,
            }
        )
    write_jsonl_atomic(output, records)
    summary = {
        "policy": args.policy,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "dataset": args.dataset,
        "num_samples": len(records),
        "accuracy": None,
        "accuracy_status": "pending_judge",
        "boxed_answer_rate": mean(record["has_boxed_answer"] for record in records),
        "truncation_rate": mean(record["truncated"] for record in records),
        "mean_completion_tokens": mean(record["completion_tokens"] for record in records),
        "settings": {
            "thinking": False,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "max_completion_tokens": args.max_completion_tokens,
            "max_model_len": args.max_model_len,
            "seed": args.seed,
            "per_prompt_seed": "seed + row_index, shared across policies",
        },
    }
    write_json(summary_path(output), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    shutdown_llm(llm)
    atexit.unregister(shutdown_llm)


if __name__ == "__main__":
    main()
