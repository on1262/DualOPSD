"""GRPO baseline with a binary math_verify correctness reward (TRL, colocated vLLM rollouts).

Run with ``accelerate launch --config_file configs/accelerate.yaml -m dualopsd grpo <TRL args>``.
The paper runs use ``--use_peft true --lora_r 64 --lora_alpha 128 --learning_rate 5e-6
--per_device_train_batch_size 1 --gradient_accumulation_steps 32 --num_train_epochs 2 --max_steps 200
--num_generations 8 --num_iterations 2 --max_prompt_length 2048 --max_completion_length 8192
--temperature 1.0 --beta 0.0 --loss_type grpo --scale_rewards group --use_vllm true --vllm_mode colocate``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from dualopsd.data.prepare import OPENTHOUGHTS_REPO, OPENTHOUGHTS_REVISION


@dataclass
class GRPOScriptArguments:
    dataset_name: str = field(default=OPENTHOUGHTS_REPO)
    dataset_revision: str = field(default=OPENTHOUGHTS_REVISION)
    stop_after_step: int | None = field(
        default=None, metadata={"help": "save and exit at this step (multiple of 25); rerun to resume"}
    )
    smoke: bool = field(default=False, metadata={"help": "1 optimizer step with 128-token completions"})


def extract_boxed_answer(text: str) -> str | None:
    """Content of the first ``\\boxed{}`` (after ``</think>`` if present), with nested braces."""
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>") :] if think_end != -1 else text
    idx = search_text.find(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(search_text) and depth > 0:
        if search_text[i] == "{":
            depth += 1
        elif search_text[i] == "}":
            depth -= 1
        i += 1
    return search_text[start : i - 1].strip() if depth == 0 else None


def _preprocess_for_parse(answer: str | None) -> str | None:
    """Rewrite ratio notation ``a:b`` as ``\\frac{a}{b}`` so math_verify can parse it."""
    if answer is None:
        return None
    ratio = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*", answer)
    if ratio:
        return rf"\frac{{{ratio.group(1)}}}{{{ratio.group(2)}}}"
    return answer


def reward_correctness(completions, Answer, **kwargs):
    """1.0 if the boxed answer matches the reference (math_verify, else exact string), else 0.0."""
    from math_verify import parse, verify

    rewards = []
    for completion, ground_truth in zip(completions, Answer):
        pred_answer = extract_boxed_answer(completion)
        reward = 0.0
        gold_parsed = parse(ground_truth)
        pred_parsed = parse(_preprocess_for_parse(pred_answer))
        if gold_parsed is not None and pred_parsed is not None:
            try:
                reward = 1.0 if verify(gold_parsed, pred_parsed) else 0.0
            except Exception:
                pass
        if reward == 0.0:
            pred_norm = re.sub(r"\s+", "", pred_answer or "").lower()
            gt_norm = re.sub(r"\s+", "", ground_truth or "").lower()
            if pred_norm and pred_norm == gt_norm:
                reward = 1.0
        rewards.append(reward)
    return rewards


def make_format_prompt(tokenizer):
    def format_prompt(example):
        messages = [
            {
                "role": "user",
                "content": f"Problem: {example['Question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "Answer": example["Answer"]}

    return format_prompt


def main(argv: list[str] | None = None) -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl import (
        GRPOConfig,
        GRPOTrainer,
        ModelConfig,
        TrlParser,
        get_kbit_device_map,
        get_peft_config,
        get_quantization_config,
    )

    from dualopsd.baselines.trl_checkpointing import latest_optimizer_checkpoint, resumable_callbacks

    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config(args=argv)
    if script_args.smoke:
        training_args.max_steps, training_args.max_completion_length = 1, 128

    dtype = model_args.dtype if model_args.dtype not in (None, "auto") else "bfloat16"
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=getattr(torch, dtype),
        use_cache=not training_args.gradient_checkpointing,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config
    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(script_args.dataset_name, revision=script_args.dataset_revision)["train"]
    dataset = dataset.map(make_format_prompt(tokenizer), remove_columns=dataset.column_names)
    split = dataset.train_test_split(test_size=0.007, seed=42)

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_correctness,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=resumable_callbacks(optimizer_interval=25, stop_after_step=script_args.stop_after_step),
    )
    resume_from_checkpoint = latest_optimizer_checkpoint(training_args.output_dir)
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print(f"CUDA_PEAK_ALLOCATED_MIB={torch.cuda.max_memory_allocated() / 2**20:.1f}", flush=True)
    if trainer.state.global_step >= trainer.state.max_steps:  # not after a --stop_after_step segment
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
