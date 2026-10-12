"""Student and teacher LoRA adapters on one frozen backbone."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

STUDENT_ADAPTER = "student"
TEACHER_ADAPTER = "teacher"


def attach_dual_lora(base_model, lora_config):
    """Attach a student and a teacher adapter; the teacher starts as a copy of the student."""

    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

    config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=list(lora_config.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, config, adapter_name=STUDENT_ADAPTER)
    model.add_adapter(TEACHER_ADAPTER, config)
    student_state = get_peft_model_state_dict(model, adapter_name=STUDENT_ADAPTER)
    set_peft_model_state_dict(model, student_state, adapter_name=TEACHER_ADAPTER)
    return model


def _belongs_to_adapter(parameter_name: str, adapter_name: str) -> bool:
    pieces = parameter_name.split(".")
    return adapter_name in pieces and any(piece.startswith("lora_") for piece in pieces)


class DualAdapterManager:
    """Switch the active adapter and expose each adapter's trainable parameters."""

    def __init__(self, model, student: str = STUDENT_ADAPTER, teacher: str = TEACHER_ADAPTER):
        self.model = model
        self.student = student
        self.teacher = teacher
        missing = {student, teacher}.difference(getattr(model, "peft_config", {}))
        if missing:
            raise ValueError(f"model is missing named adapters: {sorted(missing)}")
        self.activate_student()

    def parameters_for(self, adapter_name: str) -> list[torch.nn.Parameter]:
        parameters = [
            parameter for name, parameter in self.model.named_parameters() if _belongs_to_adapter(name, adapter_name)
        ]
        if not parameters:
            raise ValueError(f"no parameters found for adapter {adapter_name!r}")
        return parameters

    def _activate(self, active: str) -> None:
        if active not in {self.student, self.teacher}:
            raise ValueError(f"unknown adapter: {active}")
        self.model.set_adapter(active)
        for name, parameter in self.model.named_parameters():
            if _belongs_to_adapter(name, self.student) or _belongs_to_adapter(name, self.teacher):
                parameter.requires_grad_(_belongs_to_adapter(name, active))

    def activate_student(self) -> None:
        self._activate(self.student)

    def activate_teacher(self) -> None:
        self._activate(self.teacher)

    @contextmanager
    def adapters_disabled(self) -> Iterator[None]:
        """Run the frozen initial model (all adapters bypassed)."""

        with self.model.disable_adapter():
            yield

    def build_optimizers(
        self,
        *,
        student_lr: float,
        teacher_lr: float,
        student_weight_decay: float = 0.0,
        teacher_weight_decay: float = 0.0,
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        student_optimizer = torch.optim.AdamW(
            self.parameters_for(self.student), lr=student_lr, weight_decay=student_weight_decay
        )
        teacher_optimizer = torch.optim.AdamW(
            self.parameters_for(self.teacher), lr=teacher_lr, weight_decay=teacher_weight_decay
        )
        student_ids = {id(p) for group in student_optimizer.param_groups for p in group["params"]}
        teacher_ids = {id(p) for group in teacher_optimizer.param_groups for p in group["params"]}
        if student_ids & teacher_ids:
            raise RuntimeError("student and teacher optimizers share parameters")
        return student_optimizer, teacher_optimizer
