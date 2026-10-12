"""``python -m dualopsd train``: OPSD / DualOPSD distillation with auto-resume.

Example::

    python -m dualopsd train --config configs/distill/base.yaml \\
        --config configs/models/qwen3-4b.yaml --config configs/distill/dualopsd.yaml \\
        --set seed=43 --output-dir $DUALOPSD_SCRATCH/runs/main/qwen3-4b/dualopsd_seed43

``--resume auto`` (default) continues from the newest checkpoint in the output
directory. ``--resume <run>/checkpoint-N`` first stages that checkpoint of
another run into the (new) output directory and then continues it, e.g. the
compute-matched OPSD run: ``--config .../opsd_matched.yaml --resume
<opsd run>/checkpoint-100``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from .checkpoint import (
    align_resume_artifacts,
    archive_checkpoints_after,
    archive_incomplete_run_artifacts,
    check_output_dir,
    checkpoint_step,
    find_latest_checkpoint,
    load_checkpoint,
    stage_continuation,
)
from .config import SMOKE_OVERRIDES, DistillConfig, load_config
from .types import TrainingExample

LOGGER = logging.getLogger("dualopsd.train")


class JsonlDataset(Dataset):
    def __init__(self, path: str | Path):
        with Path(path).open(encoding="utf-8") as handle:
            self.examples = [TrainingExample.from_mapping(json.loads(line)) for line in handle if line.strip()]
        if not self.examples:
            raise ValueError(f"training file contains no examples: {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self.examples[index]


def load_batches(config: DistillConfig) -> DataLoader:
    """File-order batches (no shuffling): step k always sees rows [k*B, (k+1)*B)."""

    return DataLoader(
        JsonlDataset(config.data.train_file), batch_size=config.data.batch_size, shuffle=False, collate_fn=lambda rows: rows
    )


def jsonl_metric_sink(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def write(metrics):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")

    return write


def resolve_resume(output_dir: Path) -> Path | None:
    """Newest resumable checkpoint in ``output_dir``; tidy partial artifacts around it."""

    latest = find_latest_checkpoint(output_dir)
    if latest is None:
        archived = archive_incomplete_run_artifacts(output_dir)
        LOGGER.info("auto-resume: no complete checkpoint; archived=%s; starting from step 0", archived)
        return None
    step = checkpoint_step(latest)
    aligned = align_resume_artifacts(output_dir, step)
    stale = archive_checkpoints_after(output_dir, step)
    LOGGER.info("auto-resume: using %s; aligned=%s; archived=%s", latest, aligned, stale)
    return latest


def _dtype(name: str) -> torch.dtype:
    values = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in values:
        raise ValueError(f"unsupported model dtype: {name}")
    return values[name]


def build_trainer(config: DistillConfig, output_dir: Path, *, resume_from: Path | None, adapter_root: Path | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, get_constant_schedule_with_warmup

    from .adapters import DualAdapterManager, attach_dual_lora
    from .rollout import VLLMRolloutEngine
    from .trainer import DistillTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("training requires a CUDA device")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    generation_eos = GenerationConfig.from_pretrained(config.model.name).eos_token_id
    if not isinstance(generation_eos, (list, tuple)):
        generation_eos = [generation_eos]
    tokenizer._generation_eos_token_ids = tuple(
        dict.fromkeys(int(value) for value in [*generation_eos, tokenizer.eos_token_id] if value is not None)
    )
    base = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        dtype=_dtype(config.model.dtype),
        attn_implementation="flash_attention_2",
        device_map={"": torch.cuda.current_device()},
    )
    model = attach_dual_lora(base, config.lora)
    model.config.use_cache = False
    if config.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    adapters = DualAdapterManager(model)
    optim = config.optim
    student_optimizer, teacher_optimizer = adapters.build_optimizers(
        student_lr=optim.learning_rate,
        teacher_lr=optim.learning_rate,
        student_weight_decay=optim.weight_decay,
        teacher_weight_decay=optim.weight_decay,
    )
    student_scheduler = get_constant_schedule_with_warmup(student_optimizer, num_warmup_steps=optim.warmup_steps)
    teacher_scheduler = get_constant_schedule_with_warmup(teacher_optimizer, num_warmup_steps=optim.warmup_steps)
    rollout_engine = VLLMRolloutEngine(
        model,
        tokenizer,
        adapters,
        model_name=config.model.name,
        dtype=config.model.dtype,
        lora_rank=config.lora.r,
        gpu_memory_utilization=config.rollout.gpu_memory_utilization,
        max_model_len=config.rollout.max_model_len,
        max_num_seqs=config.rollout.max_num_seqs,
        seed=config.seed,
        sleep_level=config.rollout.sleep_level,
        adapter_root=adapter_root,
    )
    trainer = DistillTrainer(
        model=model,
        tokenizer=tokenizer,
        adapters=adapters,
        config=config,
        student_optimizer=student_optimizer,
        teacher_optimizer=teacher_optimizer,
        student_scheduler=student_scheduler,
        teacher_scheduler=teacher_scheduler,
        rollout_engine=rollout_engine,
        metric_sink=jsonl_metric_sink(output_dir / "metrics.jsonl"),
        rollout_path=output_dir / "rollouts.jsonl",
    )
    if resume_from is not None:
        trainer.state = load_checkpoint(
            resume_from,
            model=model,
            student_optimizer=student_optimizer,
            teacher_optimizer=teacher_optimizer,
            student_scheduler=student_scheduler,
            teacher_scheduler=teacher_scheduler,
        )
    return trainer


def _write_run_config(config: DistillConfig, output_dir: Path) -> None:
    """Record the effective config; refuse to resume a run with a different setup.

    Only ``schedule.max_steps`` may change between launches (continuations).
    """

    path = output_dir / "config.yaml"
    current = config.to_dict()
    if path.is_file():
        previous = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        comparable = [dict(item, schedule={**item.get("schedule", {}), "max_steps": None}) for item in (previous, current)]
        if comparable[0] != comparable[1]:
            raise ValueError(f"{path} differs from the requested config; use a new --output-dir")
    path.write_text(yaml.safe_dump(current, sort_keys=False), encoding="utf-8")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="dualopsd train", description="OPSD / DualOPSD distillation with auto-resume.")
    parser.add_argument("--config", action="append", required=True, help="YAML file; repeat to merge in order")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. teacher.residual_beta=0.7")
    parser.add_argument("--output-dir", help="run directory (default: output.directory)")
    parser.add_argument("--resume", default="auto", help="'auto' or a checkpoint-N directory of another run to continue")
    parser.add_argument("--smoke", action="store_true", help="1 step, 2 problems, 64 new tokens, small vLLM memory")
    parser.add_argument("--dry-run", action="store_true", help="print the effective config and check the data, then exit")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    overrides = list(args.set) + (list(SMOKE_OVERRIDES) if args.smoke else [])
    config = load_config(args.config, overrides)
    if args.smoke and args.resume != "auto":
        # A smoke continuation trains one step past the staged checkpoint.
        config.schedule.max_steps = checkpoint_step(Path(args.resume)) + 1
    output = args.output_dir or config.output.directory
    if not output:
        raise SystemExit("set --output-dir (or output.directory)")
    output_dir = check_output_dir(output)
    config.output.directory = str(output_dir)
    if args.dry_run:
        print(yaml.safe_dump(config.to_dict(), sort_keys=False), end="")
        first = next(iter(load_batches(config)))
        print(f"# data ok: {len(first)} examples in the first batch of {config.data.train_file}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume != "auto":
        source = Path(args.resume).absolute()
        if source.parent != output_dir and not any(output_dir.glob("checkpoint-*")):
            stage_continuation(source, output_dir)
    _write_run_config(config, output_dir)
    adapter_root = None
    if args.smoke:
        # Keep every smoke artifact inside the run directory.
        adapter_root = output_dir / ".lora_cache"
        os.environ.setdefault("VLLM_CACHE_ROOT", str(output_dir / ".vllm_cache"))

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    batches = load_batches(config)
    resume_from = resolve_resume(output_dir)
    trainer = build_trainer(config, output_dir, resume_from=resume_from, adapter_root=adapter_root)
    trainer.fit(batches, output_dir=output_dir)


if __name__ == "__main__":
    main()
