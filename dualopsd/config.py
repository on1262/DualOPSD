"""Distillation config: YAML files merged in order, ``${ENV:-default}`` expansion,
``--set section.key=value`` overrides and validation.

Teacher objectives (``teacher.objective``):

* ``frozen``     -- OPSD: the privileged teacher is never updated.
* ``direct``     -- DualOPSD w/o residual: minimize ``KL(sg[p_S] || p_T)``.
* ``residual``   -- DualOPSD: minimize ``KL(sg[q] || p_T)`` with
  ``q ∝ p_S * exp(beta * (log p0(.|x,r) - log p0(.|x)))``; ``teacher.clip``
  high/low gives the teacher clipping ablation (default none).

The student always minimizes the OPSD loss ``KL(sg[p_T] || p_S)`` with the
per-vocabulary upper clip ``student.kl_clip``. Teacher updates run after the
student step on the same student rollouts.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

OBJECTIVES = ("frozen", "direct", "residual")
CLIPS = ("none", "high", "low")


@dataclass
class ModelConfig:
    name: str = ""
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True


@dataclass
class LoRAConfig:
    r: int = 64
    alpha: int = 128
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class DataConfig:
    train_file: str = ""
    batch_size: int = 32


@dataclass
class RolloutConfig:
    backend: str = "vllm"
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int | None = 20
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.5
    max_num_seqs: int = 32
    sleep_level: int = 1


@dataclass
class OptimConfig:
    """Shared by the student and teacher AdamW optimizers (constant LR after warmup)."""

    learning_rate: float = 5e-6
    warmup_steps: int = 10
    weight_decay: float = 0.0
    max_grad_norm: float = 0.1
    loss_group_size: int = 4
    token_segment_size: int = 1024


@dataclass
class StudentConfig:
    kl_clip: float | None = 1e-6
    micro_batch_size: int = 2


@dataclass
class TeacherConfig:
    objective: str = "residual"
    residual_beta: float | None = None
    clip: str = "none"
    clip_threshold: float | None = None  # null: reuse student.kl_clip
    micro_batch_size: int = 2


@dataclass
class ScheduleConfig:
    max_steps: int = 100
    save_steps: int = 5
    optimizer_save_steps: int = 25


@dataclass
class OutputConfig:
    directory: str | None = None


@dataclass
class DistillConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 42

    @property
    def teacher_clip_threshold(self) -> float | None:
        """Effective threshold of the teacher clip (``None`` when unclipped)."""

        if self.teacher.clip == "none":
            return None
        return self.teacher.clip_threshold if self.teacher.clip_threshold is not None else self.student.kl_clip

    def validate(self) -> None:
        def require(condition: bool, message: str) -> None:
            if not condition:
                raise ValueError(message)

        require(bool(self.model.name), "model.name must be set (see configs/models)")
        require(bool(self.data.train_file), "data.train_file must be set")
        require(self.rollout.backend == "vllm", "rollout.backend must be vllm")
        require(self.teacher.objective in OBJECTIVES, f"teacher.objective must be one of {OBJECTIVES}")
        require(self.teacher.clip in CLIPS, f"teacher.clip must be one of {CLIPS}")
        if self.teacher.objective == "residual":
            beta = self.teacher.residual_beta
            require(beta is not None and beta >= 0, "teacher.objective=residual needs teacher.residual_beta >= 0")
        if self.teacher.objective != "residual":
            require(self.teacher.clip == "none", "teacher.clip is only defined for teacher.objective=residual")
        if self.teacher.clip != "none":
            threshold = self.teacher_clip_threshold
            require(threshold is not None and threshold > 0, "teacher clip needs a positive threshold")
        require(self.student.kl_clip is None or self.student.kl_clip > 0, "student.kl_clip must be positive or null")
        require(self.data.batch_size >= 1, "data.batch_size must be positive")
        group = self.optim.loss_group_size
        require(group >= 1 and self.data.batch_size % group == 0, "data.batch_size must be a multiple of optim.loss_group_size")
        for role in ("student", "teacher"):
            micro = getattr(self, role).micro_batch_size
            require(micro >= 1 and group % micro == 0, f"{role}.micro_batch_size must divide optim.loss_group_size")
        require(self.optim.warmup_steps >= 0, "optim.warmup_steps cannot be negative")
        require(self.optim.weight_decay >= 0, "optim.weight_decay cannot be negative")
        require(self.optim.max_grad_norm > 0, "optim.max_grad_norm must be positive")
        require(self.optim.token_segment_size >= 1, "optim.token_segment_size must be positive")
        require(self.rollout.top_k is None or self.rollout.top_k >= 1, "rollout.top_k must be null or positive")
        require(0.0 < self.rollout.gpu_memory_utilization < 1.0, "rollout.gpu_memory_utilization must be in (0, 1)")
        require(self.rollout.sleep_level in {0, 1, 2}, "rollout.sleep_level must be 0, 1 or 2")
        require(self.rollout.max_num_seqs >= 1, "rollout.max_num_seqs must be positive")
        require(self.rollout.max_model_len > self.rollout.max_new_tokens, "rollout.max_model_len must leave room for prompts")
        schedule = self.schedule
        require(min(schedule.max_steps, schedule.save_steps, schedule.optimizer_save_steps) >= 1, "schedule values must be positive")
        require(schedule.optimizer_save_steps % schedule.save_steps == 0, "optimizer_save_steps must be a multiple of save_steps")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ENV = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def expand_env(value: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}``; unset variables without default are errors."""

    def substitute(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        if os.environ.get(name):
            return os.environ[name]
        if default is None:
            raise ValueError(f"environment variable {name} is not set (source configs/user.env)")
        return default

    return _ENV.sub(substitute, value)


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return expand_env(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def merge(base: dict, update: Mapping) -> dict:
    """Recursively merge ``update`` into a copy of ``base``."""

    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


_FLOAT_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def _parse_override_value(text: str) -> Any:
    """Parse an override value as YAML, also accepting YAML-1.1-string floats such as ``1e-7``."""

    value = yaml.safe_load(text)
    if isinstance(value, str) and _FLOAT_RE.match(value.strip()):
        return float(value)
    return value


def apply_overrides(raw: dict, overrides: Iterable[str]) -> dict:
    """Apply ``section.key=value`` (or ``seed=value``) overrides; values are parsed as YAML."""

    raw = dict(raw)
    for override in overrides:
        key, sep, text = override.partition("=")
        if not sep or not key:
            raise ValueError(f"override must look like section.key=value: {override!r}")
        update: Any = _parse_override_value(text)
        for part in reversed(key.split(".")):
            update = {part: update}
        raw = merge(raw, update)
    return raw


def _construct(cls, values: Any, where: str):
    if not isinstance(values, Mapping):
        raise ValueError(f"config section {where} must be a mapping")
    known = {item.name: item for item in fields(cls)}
    unknown = set(values).difference(known)
    if unknown:
        raise ValueError(f"unknown keys in {where}: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        default = known[name].default_factory() if callable(known[name].default_factory) else known[name].default
        kwargs[name] = _construct(type(default), value, f"{where}.{name}") if is_dataclass(default) else value
    return cls(**kwargs)


def config_from_dict(raw: Mapping[str, Any]) -> DistillConfig:
    config = _construct(DistillConfig, _expand(dict(raw)), "config")
    config.validate()
    return config


def load_raw(paths: Iterable[str | Path], overrides: Iterable[str] = ()) -> dict:
    raw: dict = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config root must be a mapping: {path}")
        raw = merge(raw, loaded)
    return apply_overrides(raw, overrides)


def load_config(paths: Iterable[str | Path], overrides: Iterable[str] = ()) -> DistillConfig:
    """Merge YAML files in order, apply overrides, expand env vars and validate."""

    return config_from_dict(load_raw(paths, overrides))


SMOKE_OVERRIDES = (
    "schedule.max_steps=1",
    "schedule.save_steps=1",
    "schedule.optimizer_save_steps=1",
    "data.batch_size=2",
    "optim.loss_group_size=2",
    "student.micro_batch_size=1",
    "teacher.micro_batch_size=1",
    "rollout.max_new_tokens=64",
    "rollout.max_model_len=4096",
    "rollout.max_num_seqs=4",
    "rollout.gpu_memory_utilization=0.35",
)
