"""Shared I/O and model helpers for the re-scoring diagnostics."""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def env_dir(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def check_output_path(path: Path, *, overwrite: bool = False) -> None:
    """Refuse to clobber results or to write through links leaving the scratch root.

    Outputs shared between experiments under ``$DUALOPSD_SCRATCH`` are linked,
    not copied; writing through such a link would modify the linked original.
    """

    path = Path(os.path.abspath(path))
    scratch = env_dir("DUALOPSD_SCRATCH")
    if scratch is not None:
        scratch = Path(os.path.abspath(scratch))
        real_scratch = scratch.resolve()
        for candidate in (path, *path.parents):
            if candidate == scratch or not _is_within(candidate, scratch):
                break
            if candidate.is_symlink() and not _is_within(
                candidate.resolve(), real_scratch
            ):
                raise PermissionError(
                    f"{candidate} is a symlink outside $DUALOPSD_SCRATCH; "
                    "choose a fresh output location"
                )
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")


def model_env_var(model_key: str) -> str:
    """``qwen3-1.7b`` -> ``MODEL_QWEN3_1P7B`` (the configs/user.env naming)."""

    return "MODEL_" + model_key.upper().replace("-", "_").replace(".", "P")


def resolve_base_model(base_model: Path | None, run_dir: Path) -> Path:
    """Use ``--base-model`` or the model env var named by ``runs/<exp>/<model>/<run>``."""

    if base_model is not None:
        return base_model
    variable = model_env_var(run_dir.parent.name)
    value = os.environ.get(variable)
    if not value:
        raise ValueError(f"pass --base-model or set {variable}")
    return Path(value)


def load_adapters(base_model: Path, adapters: dict[str, Path]) -> tuple[Any, Any]:
    """Load the base model once with every named LoRA adapter attached (eval mode)."""

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local = Path(base_model).is_dir()
    base_model = str(base_model)  # a local directory or a Hugging Face id
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=local)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    cuda = torch.cuda.is_available()
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if cuda else {"": "cpu"},
        trust_remote_code=False,
        attn_implementation="flash_attention_2" if cuda else "sdpa",
    )
    (first_name, first_path), *rest = adapters.items()
    model = PeftModel.from_pretrained(
        base, first_path, adapter_name=first_name, is_trainable=False
    )
    for name, path in rest:
        model.load_adapter(path, adapter_name=name, is_trainable=False)
    model.eval()
    return tokenizer, model


def forward_completion_logits(
    model: Any,
    prompt_ids: list[int],
    completion_ids: list[int],
    *,
    adapter: str | None,
) -> torch.Tensor:
    """Logits predicting each completion token; ``adapter=None`` uses the base model."""

    if not prompt_ids or not completion_ids:
        raise ValueError("prompt and completion must both contain tokens")
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    positions = torch.arange(
        len(prompt_ids) - 1, len(prompt_ids) - 1 + len(completion_ids), device=device
    )
    if adapter is None:
        context = model.disable_adapter()
    else:
        model.set_adapter(adapter)
        context = nullcontext()
    with context, torch.inference_mode():
        try:
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=positions,
                use_cache=False,
            ).logits
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits
    if logits.shape[1] != len(completion_ids):
        logits = logits[:, positions]
    return logits.squeeze(0)


def cuda_memory_stats(status: dict[str, Any]) -> None:
    """Track peak live-tensor and lower-bound device memory in ``status`` (MiB)."""

    if not torch.cuda.is_available():
        return
    status["live_tensor_peak_allocated_mib"] = max(
        float(status.get("live_tensor_peak_allocated_mib", 0)),
        torch.cuda.max_memory_allocated() / 2**20,
    )
    free, total = torch.cuda.mem_get_info()
    status["system_peak_used_mib_lower_bound"] = max(
        float(status.get("system_peak_used_mib_lower_bound", 0)),
        (total - free) / 2**20,
    )
