"""Grade held-out generations with a model judge on the terminal answer (default: Qwen3-8B).

For each response the last boxed answer (or the trailing sentences) is extracted and the
judge, decoding greedily under a structured choice, answers ``\\boxed{yes}`` or
``\\boxed{no}`` on equivalence with the reference answer. ``<dir>/<name>.jsonl`` is
written to ``<output-dir or dir>/<name>_judged.jsonl`` plus ``<name>_judged_summary.json``.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from dualopsd.evaluation import check_output, read_jsonl, shutdown_llm, write_json, write_jsonl_atomic
from dualopsd.evaluation.terminal_answer import TerminalAnswer, extract_terminal_answer, parse_boxed_yes_no

JUDGE_TEMPLATE = """Determine whether the candidate's final answer is equivalent to the standard answer.

Rules:
- Decide whether the standard answer is explicitly present in the candidate final text and is not replaced or contradicted by a different final answer. A concluding sentence or a longer answer may contain extra explanation; do not require the candidate text to consist only of the answer.
- Equivalent notation, simplified forms, algebraically rearranged equations, and numerically consistent exact/approximate forms count as equal.
- Ignore a unit that appears only in the standard answer when the candidate has the same numeric value. Leading zeros are also irrelevant.
- If the standard is a multiple-choice label, the same label followed by the option text or value counts as equal.
- Extra compatible statements do not make an answer wrong. Return no only when the candidate gives a clearly different answer, contradicts the standard answer, or contains no identifiable answer.
- Return exactly one verdict and no other text: \\boxed{{yes}} or \\boxed{{no}}.

Examples that must be judged yes:
- Standard `4 \\text{{ feet}}`; candidate `4`.
- Standard `053`; candidate `53`.
- Standard `y=-\\frac{{2}}{{5}}x+\\frac{{12}}{{5}}`; candidate `y-2=-\\frac{{2}}{{5}}(x-1)`.
- Standard `360^\\circ`; candidate `Therefore, the sum is 360 degrees.`
- Standard `\\text{{C}}`; candidate `(C) 0, 3`.

Standard answer:
{standard_answer}

Candidate final text:
{candidate_answer}
"""

ALLOWED_OUTPUT = [r"\boxed{yes}", r"\boxed{no}"]
JUDGE_MAX_TOKENS = 16


def judge_prompt(tokenizer: Any, standard_answer: str, candidate_answer: str) -> str:
    content = JUDGE_TEMPLATE.format(standard_answer=standard_answer, candidate_answer=candidate_answer)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def build_requests(
    records: list[dict[str, Any]], *, tokenizer: Any, min_answer_chars: int = 10
) -> tuple[list[TerminalAnswer], dict[int, str]]:
    """Extract terminal answers and build judge prompts (keyed by record index).

    Records whose response has no terminal text get no prompt and are judged ``no``.
    """
    terminals = [extract_terminal_answer(str(r["response"]), min_chars=min_answer_chars) for r in records]
    prompts = {
        index: judge_prompt(tokenizer, str(records[index]["answer"]), terminal.text)
        for index, terminal in enumerate(terminals)
        if terminal.text.strip()
    }
    return terminals, prompts


def attach_verdicts(
    records: list[dict[str, Any]], terminals: list[TerminalAnswer], responses: dict[int, str]
) -> list[dict[str, Any]]:
    judged = []
    for index, (record, terminal) in enumerate(zip(records, terminals, strict=True)):
        judge_response = responses.get(index, r"\boxed{no}")
        verdict = parse_boxed_yes_no(judge_response)
        updated = {key: value for key, value in record.items() if key not in {"correct", "rule_based_correct"}}
        updated.update(
            {
                "terminal_answer": terminal.text,
                "terminal_extraction_mode": terminal.mode,
                "judge_response": judge_response,
                "judge_answer": verdict,
                "judge_parse_valid": verdict is not None,
                "correct": verdict == "yes",
            }
        )
        judged.append(updated)
    return judged


def summarize(judged: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_samples": len(judged),
        "accuracy": mean(bool(record["correct"]) for record in judged),
        "judge_parse_failure_rate": mean(not bool(record["judge_parse_valid"]) for record in judged),
        "boxed_answer_rate": mean(record["terminal_extraction_mode"] == "last_boxed" for record in judged),
        "truncation_rate": mean(bool(record["truncated"]) for record in judged),
        "mean_completion_tokens": mean(int(record["completion_tokens"]) for record in judged),
    }


def judged_path(source: Path, output_dir: Path | None) -> Path:
    return (output_dir or source.parent) / f"{source.stem}_judged.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="dualopsd judge", description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="generation jsonl files")
    parser.add_argument("--output-dir", type=Path, default=None, help="default: next to each input")
    parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL"), help="default: $JUDGE_MODEL")
    parser.add_argument("--min-answer-chars", type=int, default=10)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="judge the first 2 records of each input")
    args = parser.parse_args(argv)
    if args.smoke:
        args.limit = 2
    if not args.model:
        parser.error("--model is required when $JUDGE_MODEL is unset")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    sources = []
    for path in args.inputs:
        records = read_jsonl(path, args.limit)
        if not records:
            raise ValueError(f"empty input: {path}")
        destination = check_output(judged_path(path, args.output_dir), overwrite=args.overwrite)
        check_output(destination.with_name(destination.stem + "_summary.json"), overwrite=args.overwrite)
        sources.append((path, destination, records))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        seed=42,
    )
    atexit.register(shutdown_llm, llm)
    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=JUDGE_MAX_TOKENS,
        seed=42,
        structured_outputs=StructuredOutputsParams(choice=ALLOWED_OUTPUT, disable_fallback=True),
    )

    # All inputs are judged in one vLLM batch.
    plans = [build_requests(records, tokenizer=tokenizer, min_answer_chars=args.min_answer_chars)
             for _, _, records in sources]
    prompts = [prompt for _, requests in plans for prompt in requests.values()]
    outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=True)
    if len(outputs) != len(prompts):
        raise RuntimeError("judge output count mismatch")
    texts = iter(output.outputs[0].text.strip() for output in outputs)

    for (path, destination, records), (terminals, requests) in zip(sources, plans):
        judged = attach_verdicts(records, terminals, {index: next(texts) for index in requests})
        write_jsonl_atomic(destination, judged)
        summary = {
            "source": str(path),
            "output": str(destination),
            "judge_model": args.model,
            **summarize(judged),
            "settings": {
                "min_answer_chars": args.min_answer_chars,
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs,
                "temperature": 0.0,
                "max_tokens": JUDGE_MAX_TOKENS,
                "thinking": False,
                "allowed_output": ALLOWED_OUTPUT,
                "structured_choice": True,
            },
        }
        write_json(destination.with_name(destination.stem + "_summary.json"), summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    shutdown_llm(llm)
    atexit.unregister(shutdown_llm)
    print("vLLM EngineCore shutdown complete", flush=True)


if __name__ == "__main__":
    main()
