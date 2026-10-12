"""SFT baseline: LoRA fine-tuning on the OpenThoughts reference solutions with TRL.

Run with ``accelerate launch --config_file configs/accelerate.yaml -m dualopsd sft <TRL args>``.
All ``SFTConfig`` / ``ModelConfig`` flags are accepted; e.g. the paper runs use
``--use_peft true --lora_r 64 --lora_alpha 128 --learning_rate 5e-6 --per_device_train_batch_size 2
--gradient_accumulation_steps 32 --num_train_epochs 4 --max_steps 100 --max_length 16000``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dualopsd.data.prepare import OPENTHOUGHTS_REPO, OPENTHOUGHTS_REVISION

PROMPT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


@dataclass
class SFTScriptArguments:
    dataset_name: str = field(default=OPENTHOUGHTS_REPO)
    dataset_revision: str = field(default=OPENTHOUGHTS_REVISION)
    stop_after_step: int | None = field(
        default=None, metadata={"help": "save and exit at this step (multiple of 25); rerun to resume"}
    )
    smoke: bool = field(default=False, metadata={"help": "1 optimizer step with tiny batch and length"})


def make_format_fn(tokenizer):
    """Chat-formatted (problem, solution) pair; the user turn matches the evaluation prompt."""

    def format_example(example):
        messages = [
            {"role": "user", "content": f"{example['problem']}{PROMPT_SUFFIX}"},
            {"role": "assistant", "content": example["solution"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    return format_example


def main(argv: list[str] | None = None) -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl import (
        ModelConfig,
        SFTConfig,
        SFTTrainer,
        TrlParser,
        get_kbit_device_map,
        get_peft_config,
        get_quantization_config,
    )

    from dualopsd.baselines.trl_checkpointing import latest_optimizer_checkpoint, resumable_callbacks

    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config(args=argv)
    if script_args.smoke:
        training_args.max_steps, training_args.gradient_accumulation_steps, training_args.max_length = 1, 2, 2048

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        dtype=torch.bfloat16,
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
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(script_args.dataset_name, revision=script_args.dataset_revision)["train"]
    split = dataset.map(make_format_fn(tokenizer)).train_test_split(test_size=0.01, seed=42)

    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
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
