"""Append-only, token-faithful rollout records (``rollouts.jsonl``).

One JSON line per generated trace with fields: step, phase, example_id,
prompt_token_ids, completion_token_ids, input, output, resp_token, answer,
correctness, is_truncated. Text is decoded from the exact unpadded token ids
with special tokens kept; a trace is truncated iff its last token is not EOS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch

from .prompts import generation_eos_token_ids
from .types import RolloutBatch, TrainingExample


def _unpad(row: torch.Tensor, mask: torch.Tensor) -> list[int]:
    return row.detach().cpu()[mask.detach().cpu().bool()].tolist()


class RolloutDumper:
    def __init__(self, path: str | Path | None, tokenizer):
        self.path = Path(path) if path else None
        self.tokenizer = tokenizer
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        step: int,
        phase: str,
        examples: Sequence[TrainingExample],
        rollouts: RolloutBatch,
        correctness: Sequence[bool | None],
    ) -> None:
        if self.path is None:
            return
        if len(correctness) != len(rollouts.completion_texts):
            raise ValueError("correctness must contain one value per rollout")
        eos_ids = set(generation_eos_token_ids(self.tokenizer))
        with self.path.open("a", encoding="utf-8") as handle:
            for row in range(len(rollouts.completion_texts)):
                example = examples[int(rollouts.example_indices[row].detach().cpu().item())]
                prompt_ids = _unpad(rollouts.prompt_ids[row], rollouts.prompt_attention_mask[row])
                response_ids = _unpad(rollouts.completion_ids[row], rollouts.completion_mask[row])
                record = {
                    "step": step,
                    "phase": phase,
                    "example_id": example.id,
                    "prompt_token_ids": prompt_ids,
                    "completion_token_ids": response_ids,
                    "input": self.tokenizer.decode(prompt_ids, skip_special_tokens=False),
                    "output": self.tokenizer.decode(response_ids, skip_special_tokens=False),
                    "resp_token": len(response_ids),
                    "answer": example.answer,
                    "correctness": correctness[row],
                    "is_truncated": not response_ids or response_ids[-1] not in eos_ids,
                }
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
