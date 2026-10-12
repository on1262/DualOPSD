"""Small data structures shared by the distillation trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class TrainingExample:
    id: str
    domain: str
    problem: str
    reference: str
    answer: Any
    verifier: str
    meta: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingExample":
        required = {"id", "domain", "problem", "reference", "answer", "verifier", "meta"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"example is missing required fields: {sorted(missing)}")
        return cls(**{key: value[key] for key in required})


@dataclass
class RolloutBatch:
    """Generated completions, ordered problem-major and generation-minor."""

    prompt_texts: list[str]
    completion_texts: list[str]
    prompt_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    prompt_lengths: torch.Tensor
    old_logprobs: torch.Tensor
    example_indices: torch.Tensor

    def __post_init__(self) -> None:
        rows = len(self.completion_texts)
        if rows == 0:
            raise ValueError("a rollout batch cannot be empty")
        tensors = (
            self.prompt_ids,
            self.prompt_attention_mask,
            self.completion_ids,
            self.completion_mask,
            self.old_logprobs,
        )
        if any(tensor.shape[0] != rows for tensor in tensors):
            raise ValueError("all rollout tensors must have one row per completion")
        if self.completion_ids.shape != self.completion_mask.shape:
            raise ValueError("completion_ids and completion_mask must have identical shapes")
        if self.old_logprobs.shape != self.completion_ids.shape:
            raise ValueError("old_logprobs must be token-aligned with completion_ids")

    @property
    def device(self) -> torch.device:
        return self.completion_ids.device

    def subset(self, indices: Sequence[int] | torch.Tensor) -> "RolloutBatch":
        selected = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        rows = selected.cpu().tolist()
        return RolloutBatch(
            prompt_texts=[self.prompt_texts[index] for index in rows],
            completion_texts=[self.completion_texts[index] for index in rows],
            prompt_ids=self.prompt_ids.index_select(0, selected),
            prompt_attention_mask=self.prompt_attention_mask.index_select(0, selected),
            completion_ids=self.completion_ids.index_select(0, selected),
            completion_mask=self.completion_mask.index_select(0, selected),
            prompt_lengths=self.prompt_lengths.index_select(0, selected),
            old_logprobs=self.old_logprobs.index_select(0, selected),
            example_indices=self.example_indices.index_select(0, selected),
        )


@dataclass
class TrainerState:
    global_step: int = 0
    teacher_steps: int = 0
    student_steps: int = 0
    generated_student_tokens: int = 0


BatchLike = Sequence[TrainingExample] | Sequence[Mapping[str, Any]]
