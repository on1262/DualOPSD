"""Atomic dual-adapter checkpoints, auto-resume and run-directory safety.

Layout of a run directory::

    checkpoint-<N>/{student,teacher}/adapter_model.safetensors   every save_steps (all kept)
    checkpoint-<N>/{trainer_state.json,config.yaml}
    training_state-<N>.pt   optimizer/scheduler/RNG, every optimizer_save_steps (latest only)
    metrics.jsonl  rollouts.jsonl
"""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file, save_file

from .types import TrainerState

ADAPTERS = ("student", "teacher")


def _training_state_path(checkpoint: Path, step: int) -> Path:
    return checkpoint.parent / f"training_state-{step}.pt"


def checkpoint_step(checkpoint: str | Path) -> int:
    return int(Path(checkpoint).name.rsplit("-", 1)[1])


def _is_complete(checkpoint: Path) -> bool:
    required = [checkpoint / "trainer_state.json"]
    required += [checkpoint / name / "adapter_model.safetensors" for name in ADAPTERS]
    return all(path.is_file() for path in required)


def find_latest_checkpoint(directory: str | Path) -> Path | None:
    """Newest complete checkpoint that has its resumable training-state file."""

    root = Path(directory)
    candidates: list[tuple[int, Path]] = []
    for state_path in root.glob("training_state-*.pt"):
        try:
            step = int(state_path.stem.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if _is_complete(root / f"checkpoint-{step}"):
            candidates.append((step, root / f"checkpoint-{step}"))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _archive_directory(root: Path, label: str) -> Path:
    archive_root = root / ".resume_archive"
    candidate = archive_root / label
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = archive_root / f"{label}-{suffix}"
    candidate.mkdir(parents=True)
    return candidate


def archive_incomplete_run_artifacts(directory: str | Path) -> list[str]:
    """Move a partial run without optimizer state aside before restarting from step 0."""

    root = Path(directory)
    candidates = [path for path in (root / "metrics.jsonl", root / "rollouts.jsonl") if path.exists()]
    candidates += sorted(root.glob("checkpoint-*")) + sorted(root.glob("training_state-*.pt"))
    if not candidates:
        return []
    archive = _archive_directory(root, "restart-from-zero")
    for source in candidates:
        os.replace(source, archive / source.name)
    return [source.name for source in candidates]


def archive_checkpoints_after(directory: str | Path, step: int) -> list[str]:
    """Move adapter-only checkpoints newer than the durable optimizer state aside."""

    root = Path(directory)
    stale = []
    for checkpoint in root.glob("checkpoint-*"):
        try:
            if checkpoint_step(checkpoint) > step:
                stale.append((checkpoint_step(checkpoint), checkpoint))
        except (IndexError, ValueError):
            continue
    if not stale:
        return []
    archive = _archive_directory(root, f"resume-from-{step}")
    for _, source in sorted(stale):
        os.replace(source, archive / source.name)
    return [source.name for _, source in sorted(stale)]


def align_resume_artifacts(directory: str | Path, step: int) -> dict[str, tuple[int, int]]:
    """Drop append-only rows produced after the checkpoint at ``step``."""

    root = Path(directory)
    rules = {
        "rollouts.jsonl": lambda row, _: int(row["step"]) < step,
        "metrics.jsonl": lambda row, index: int(row.get("train/global_step", index + 1)) <= step,
    }
    results: dict[str, tuple[int, int]] = {}
    for name, keep in rules.items():
        path = root / name
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        retained = [row for index, row in enumerate(rows) if keep(row, index)]
        temporary = path.with_name(f".{path.name}.resume-{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in retained:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        os.replace(temporary, path)
        results[name] = (len(rows), len(retained))
    return results


def stage_continuation(source_checkpoint: str | Path, directory: str | Path) -> Path:
    """Copy a finished run's checkpoint into a fresh run directory to continue it.

    Copies ``checkpoint-<N>``, ``training_state-<N>.pt`` and the metric/rollout
    logs up to step N, so the continuation keeps the full history (used for the
    compute-matched OPSD runs). Returns the staged checkpoint path.
    """

    source = Path(source_checkpoint)
    step = checkpoint_step(source)
    state_path = _training_state_path(source, step)
    if not _is_complete(source) or not state_path.is_file():
        raise FileNotFoundError(f"{source} is not a resumable checkpoint (needs {state_path.name})")
    root = Path(directory)
    if any(root.glob("checkpoint-*")):
        raise FileExistsError(f"refusing to stage into a run directory with checkpoints: {root}")
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, root / source.name)
    shutil.copy2(state_path, root / state_path.name)
    for name in ("metrics.jsonl", "rollouts.jsonl"):
        if (source.parent / name).is_file():
            shutil.copy2(source.parent / name, root / name)
    align_resume_artifacts(root, step)
    return root / source.name


def check_output_dir(directory: str | Path, scratch: str | None = None) -> Path:
    """Refuse output directories that would write through a symlink out of ``$DUALOPSD_SCRATCH``.

    Outputs shared between experiments are linked, not copied; appending
    through such a link would modify the linked original run.
    """

    path = Path(directory).absolute()
    scratch = scratch if scratch is not None else os.environ.get("DUALOPSD_SCRATCH")
    if not scratch:
        return path
    scratch_root = Path(scratch).absolute()
    scratch_real = scratch_root.resolve()
    try:
        relative = path.relative_to(scratch_root)
    except ValueError:
        return path
    current = scratch_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            target = current.resolve()
            if target != scratch_real and scratch_real not in target.parents:
                raise PermissionError(f"{current} links outside {scratch_root} ({target}); choose a new run name")
    return path


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _save_adapter(model, adapter_name: str, directory: Path) -> None:
    from peft import get_peft_model_state_dict

    directory.mkdir(parents=True, exist_ok=True)
    model.peft_config[adapter_name].save_pretrained(directory)
    state = get_peft_model_state_dict(model, adapter_name=adapter_name)
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in state.items()},
        directory / "adapter_model.safetensors",
    )


def _load_adapter(model, adapter_name: str, directory: Path) -> None:
    from peft import set_peft_model_state_dict

    result = set_peft_model_state_dict(model, load_file(directory / "adapter_model.safetensors"), adapter_name=adapter_name)
    unexpected = getattr(result, "unexpected_keys", [])
    if unexpected:
        raise ValueError(f"unexpected keys while loading {adapter_name}: {unexpected}")


def save_checkpoint(
    directory: str | Path,
    *,
    model,
    student_optimizer,
    teacher_optimizer,
    state: TrainerState,
    config,
    student_scheduler=None,
    teacher_scheduler=None,
    save_training_state: bool = True,
) -> Path:
    destination = Path(directory)
    if destination.exists():
        if _is_complete(destination):
            raise FileExistsError(f"refusing to overwrite complete checkpoint: {destination}")
        destination.rename(destination.with_name(f".{destination.name}.incomplete-{os.getpid()}"))
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary checkpoint already exists: {temporary}")
    temporary.mkdir(parents=True)
    for name in ADAPTERS:
        _save_adapter(model, name, temporary / name)
    metadata = asdict(state)
    metadata["student_checkpoint_step"] = state.global_step
    metadata["teacher_checkpoint_step"] = state.global_step
    (temporary / "trainer_state.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    payload = config.to_dict() if hasattr(config, "to_dict") else config
    (temporary / "config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.rename(destination)
    if not save_training_state:
        return destination

    state_path = _training_state_path(destination, state.global_step)
    training_state = {
        "global_step": state.global_step,
        "student_optimizer": student_optimizer.state_dict(),
        "teacher_optimizer": teacher_optimizer.state_dict(),
        "student_scheduler": student_scheduler.state_dict() if student_scheduler is not None else None,
        "teacher_scheduler": teacher_scheduler.state_dict() if teacher_scheduler is not None else None,
        "rng": _rng_state(),
    }
    state_temporary = state_path.with_name(f".{state_path.name}.tmp-{os.getpid()}")
    torch.save(training_state, state_temporary)
    os.replace(state_temporary, state_path)
    for previous in destination.parent.glob("training_state-*.pt"):
        if previous != state_path:
            previous.unlink()
    return destination


def load_checkpoint(
    directory: str | Path,
    *,
    model,
    student_optimizer=None,
    teacher_optimizer=None,
    student_scheduler=None,
    teacher_scheduler=None,
    map_location: str | torch.device = "cpu",
) -> TrainerState:
    source = Path(directory)
    metadata = json.loads((source / "trainer_state.json").read_text(encoding="utf-8"))
    step = int(metadata["student_checkpoint_step"])
    if step != int(metadata["teacher_checkpoint_step"]):
        raise ValueError("asynchronous student/teacher checkpoint is not supported")
    for name in ADAPTERS:
        _load_adapter(model, name, source / name)
    state_path = _training_state_path(source, step)
    if not state_path.is_file():
        raise FileNotFoundError(f"checkpoint has no resumable training state: {state_path}")
    training_state = torch.load(state_path, map_location=map_location, weights_only=False)
    if int(training_state["global_step"]) != step:
        raise ValueError("adapter checkpoint and optimizer training state have different steps")
    for component, key in (
        (student_optimizer, "student_optimizer"),
        (teacher_optimizer, "teacher_optimizer"),
        (student_scheduler, "student_scheduler"),
        (teacher_scheduler, "teacher_scheduler"),
    ):
        if component is not None and training_state[key] is not None:
            component.load_state_dict(training_state[key])
    _restore_rng_state(training_state["rng"])
    return TrainerState(
        **{name: metadata.get(name, field.default) for name, field in TrainerState.__dataclass_fields__.items()}
    )
