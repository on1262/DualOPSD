"""OPSD / DualOPSD training loop: one student rollout per step, then a student
update and (unless the teacher is frozen) a teacher update on the same rollouts.

Loss normalization: the rollout batch is split into contiguous groups of
``optim.loss_group_size`` responses; each micro-batch token mean is weighted by
``micro tokens / group tokens / number of groups`` so the step loss is
``(1/G) * sum_g sum_{n in g} l_n / N_g``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from .adapters import DualAdapterManager
from .checkpoint import save_checkpoint
from .config import DistillConfig
from .losses import (
    js_divergence_values,
    masked_mean,
    opsd_forward_kl,
    opsd_forward_kl_values,
    privileged_residual_target_kl_values,
)
from .prompts import append_completions, build_student_prompt, build_teacher_prompt, completion_logits
from .rollout import model_device
from .rollout_dump import RolloutDumper
from .types import BatchLike, RolloutBatch, TrainerState, TrainingExample

LOGGER = logging.getLogger(__name__)


def verify_rollout(example: TrainingExample, completion: str) -> bool:
    """Rule-based answer check stored in ``rollouts.jsonl`` (never used in a loss)."""

    from .verifiers import verify_example

    return bool(verify_example(example, completion))


def _as_examples(batch: BatchLike) -> list[TrainingExample]:
    return [item if isinstance(item, TrainingExample) else TrainingExample.from_mapping(item) for item in batch]


def _grad_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    norms = [parameter.grad.detach().float().norm(2) for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(norms).norm(2).item()) if norms else 0.0


def _summaries(prefix: str, parts: list[torch.Tensor]) -> dict[str, float]:
    values = torch.cat(parts) if parts else torch.empty(0)
    if values.numel() == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_p90": 0.0}
    return {f"{prefix}_mean": float(values.mean().item()), f"{prefix}_p90": float(torch.quantile(values, 0.9).item())}


def _valid(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values.detach()[mask.bool()].float().cpu()


class DistillTrainer:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        adapters: DualAdapterManager,
        config: DistillConfig,
        student_optimizer: torch.optim.Optimizer,
        teacher_optimizer: torch.optim.Optimizer,
        rollout_engine,
        verifier: Callable[[TrainingExample, str], bool] = verify_rollout,
        student_scheduler=None,
        teacher_scheduler=None,
        metric_sink: Callable[[Mapping[str, float]], None] | None = None,
        rollout_path: str | Path | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.adapters = adapters
        self.config = config
        self.student_optimizer = student_optimizer
        self.teacher_optimizer = teacher_optimizer
        self.student_scheduler = student_scheduler
        self.teacher_scheduler = teacher_scheduler
        self.verifier = verifier
        self.rollout_engine = rollout_engine
        self.metric_sink = metric_sink
        self.rollout_dumper = RolloutDumper(rollout_path, tokenizer)
        self.state = TrainerState()
        self.last_metrics: dict[str, float] = {}
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prompts -------------------------------------------------------------

    def _render_student(self, example: TrainingExample) -> str:
        return build_student_prompt(self.tokenizer, example.problem)

    def _render_teacher(self, example: TrainingExample) -> str:
        max_prompt_tokens = self.config.rollout.max_model_len - self.config.rollout.max_new_tokens
        return build_teacher_prompt(self.tokenizer, example.problem, example.reference, max_tokens=max_prompt_tokens)

    # Phases and bookkeeping ----------------------------------------------

    def _run_phase(self, name: str, operation: Callable[[], Any]) -> tuple[Any, dict[str, float]]:
        """Run one phase, recording wall time and main-process CUDA memory.

        ``memory_allocated`` measures live tensors of the training process;
        ``mem_get_info`` adds the card view because vLLM runs in a separate
        EngineCore process.
        """

        cuda = torch.cuda.is_available()
        mib = 1024**2
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            allocated_start = torch.cuda.memory_allocated()
            reserved_start = torch.cuda.memory_reserved()
            free_start, _ = torch.cuda.mem_get_info()
        started = time.monotonic()
        try:
            result = operation()
        finally:
            if cuda:
                torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            metrics = {f"phase/{name}_seconds": float(elapsed)}
            if cuda:
                free_end, total = torch.cuda.mem_get_info()
                metrics.update(
                    {
                        f"memory/{name}_allocated_start_mib": allocated_start / mib,
                        f"memory/{name}_allocated_end_mib": torch.cuda.memory_allocated() / mib,
                        f"memory/{name}_peak_allocated_mib": torch.cuda.max_memory_allocated() / mib,
                        f"memory/{name}_reserved_start_mib": reserved_start / mib,
                        f"memory/{name}_reserved_end_mib": torch.cuda.memory_reserved() / mib,
                        f"memory/{name}_peak_reserved_mib": torch.cuda.max_memory_reserved() / mib,
                        f"memory/{name}_card_free_start_mib": free_start / mib,
                        f"memory/{name}_card_free_end_mib": free_end / mib,
                        f"memory/{name}_card_total_mib": total / mib,
                    }
                )
            LOGGER.info("phase=%s seconds=%.3f", name, elapsed)
        return result, metrics

    def _sleep_rollout_engine(self) -> None:
        sleep = getattr(self.rollout_engine, "sleep", None)
        if sleep is not None:
            sleep()

    @staticmethod
    def _release_memory() -> None:
        # vLLM shares the GPU in-process; hand cached blocks back after each step.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def dump_student_rollouts(self, batch: BatchLike, rollouts: RolloutBatch) -> None:
        examples = _as_examples(batch)
        correctness = [
            bool(self.verifier(examples[int(index.item())], completion))
            for index, completion in zip(rollouts.example_indices, rollouts.completion_texts)
        ]
        self.rollout_dumper.write(
            step=self.state.global_step, phase="student", examples=examples, rollouts=rollouts, correctness=correctness
        )

    # Scoring ---------------------------------------------------------------

    def _forward_completion(
        self,
        *,
        adapter_name: str | None,
        prompts: Sequence[str],
        rollouts: RolloutBatch,
        token_start: int = 0,
        token_stop: int | None = None,
    ) -> torch.Tensor:
        """Logits for completion tokens ``[token_start, token_stop)`` under ``prompts``.

        ``adapter_name=None`` scores with the frozen initial model.
        """

        if adapter_name is None:
            adapter_context = self.adapters.adapters_disabled()
        elif adapter_name == self.adapters.student:
            self.adapters.activate_student()
            adapter_context = None
        elif adapter_name == self.adapters.teacher:
            self.adapters.activate_teacher()
            adapter_context = None
        else:
            raise ValueError(f"unknown adapter: {adapter_name}")
        device = model_device(self.model)
        full_ids, full_mask, prompt_lengths = append_completions(
            [self.tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts],
            rollouts.completion_ids.to(device),
            rollouts.completion_mask.to(device),
            pad_token_id=self.tokenizer.pad_token_id,
            device=device,
        )
        completion_length = rollouts.completion_ids.shape[1]
        token_stop = completion_length if token_stop is None else min(token_stop, completion_length)
        if not 0 <= token_start < token_stop:
            raise ValueError("completion token segment must be non-empty and ordered")
        offset = int(prompt_lengths[0].item()) - 1
        positions = torch.arange(offset + token_start, offset + token_stop, device=device)

        def forward():
            try:
                return self.model(input_ids=full_ids, attention_mask=full_mask, logits_to_keep=positions)
            except TypeError as exc:
                if "logits_to_keep" not in str(exc):
                    raise
                return self.model(input_ids=full_ids, attention_mask=full_mask)

        if adapter_context is None:
            output = forward()
        else:
            with adapter_context:
                output = forward()
        if output.logits.shape[1] == token_stop - token_start:
            return output.logits
        return completion_logits(output.logits, prompt_lengths, completion_length)[:, token_start:token_stop]

    @staticmethod
    def _token_segments(length: int, size: int):
        for start in range(0, length, size):
            yield start, min(start + size, length)

    @staticmethod
    def _loss_weight(
        micro_mask: torch.Tensor, *, completion_lengths: Sequence[int], start: int, stop: int, group_size: int
    ) -> float:
        """Weight of a micro-batch token mean within its loss group (see module docstring)."""

        total_rows = len(completion_lengths)
        if total_rows % group_size:
            raise ValueError("rollout rows must be divisible by loss_group_size")
        group_start = (start // group_size) * group_size
        if stop > group_start + group_size:
            raise ValueError("a scoring micro-batch cannot cross a loss-group boundary")
        group_tokens = sum(completion_lengths[group_start : group_start + group_size])
        if group_tokens < 1:
            raise ValueError("a loss group must contain completion tokens")
        return float(micro_mask.sum().item()) / group_tokens / (total_rows // group_size)

    def _micro_segments(self, rollouts: RolloutBatch, micro_batch_size: int):
        """Yield ``(start, stop, micro, token_start, token_stop, mask)`` for non-empty segments."""

        device = model_device(self.model)
        for start in range(0, len(rollouts.completion_texts), micro_batch_size):
            stop = min(start + micro_batch_size, len(rollouts.completion_texts))
            micro = rollouts.subset(range(start, stop))
            for token_start, token_stop in self._token_segments(
                micro.completion_ids.shape[1], self.config.optim.token_segment_size
            ):
                mask = micro.completion_mask[:, token_start:token_stop].to(device)
                if mask.any():
                    yield start, stop, micro, token_start, token_stop, mask
            del micro

    def _step(self, optimizer, scheduler, adapter_name: str) -> float:
        parameters = self.adapters.parameters_for(adapter_name)
        grad_norm = _grad_norm(parameters)
        torch.nn.utils.clip_grad_norm_(parameters, self.config.optim.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        self.rollout_engine.sync_adapter(adapter_name)
        return grad_norm

    # Student ---------------------------------------------------------------

    def student_rollout(self, batch: BatchLike) -> RolloutBatch:
        examples = _as_examples(batch)
        rollout = self.config.rollout
        rollouts = self.rollout_engine.generate(
            adapter_name=self.adapters.student,
            prompts=[self._render_student(example) for example in examples],
            num_generations=1,
            temperature=rollout.temperature,
            top_p=rollout.top_p,
            top_k=rollout.top_k,
            max_new_tokens=rollout.max_new_tokens,
            seed=self.config.seed + 2 * self.state.global_step + 1,
        )
        self.state.generated_student_tokens += int(rollouts.completion_mask.sum().item())
        return rollouts

    def student_update(self, batch: BatchLike, rollouts: RolloutBatch) -> dict[str, float]:
        """One AdamW step on the clipped OPSD loss ``KL(sg[p_T(.|x,r)] || p_S(.|x))``."""

        examples = _as_examples(batch)
        teacher_prompts = [self._render_teacher(examples[int(index.item())]) for index in rollouts.example_indices]
        student_prompts = rollouts.prompt_texts
        self.student_optimizer.zero_grad(set_to_none=True)
        device = model_device(self.model)
        full_mask = rollouts.completion_mask.to(device)
        completion_lengths = full_mask.sum(1).long().cpu().tolist()
        loss_value = 0.0
        kl_parts = []
        for start, stop, micro, token_start, token_stop, mask in self._micro_segments(
            rollouts, self.config.student.micro_batch_size
        ):
            segment = dict(rollouts=micro, token_start=token_start, token_stop=token_stop)
            with torch.no_grad():
                teacher_scores = self._forward_completion(
                    adapter_name=self.adapters.teacher, prompts=teacher_prompts[start:stop], **segment
                )
            student_scores = self._forward_completion(
                adapter_name=self.adapters.student, prompts=student_prompts[start:stop], **segment
            )
            micro_loss, per_token = opsd_forward_kl(
                student_scores, teacher_scores, mask, pointwise_clip=self.config.student.kl_clip
            )
            weight = self._loss_weight(
                mask,
                completion_lengths=completion_lengths,
                start=start,
                stop=stop,
                group_size=self.config.optim.loss_group_size,
            )
            weighted_loss = micro_loss * weight
            weighted_loss.backward()
            loss_value += float(weighted_loss.detach().item())
            kl_parts.append(per_token[mask.bool()].float().cpu())
            del teacher_scores, student_scores, micro_loss, per_token, weighted_loss, mask
        grad_norm = self._step(self.student_optimizer, self.student_scheduler, self.adapters.student)
        self.state.student_steps += 1
        metrics = {
            "student/opsd_loss": loss_value,
            "student/completion_length": float(full_mask.sum(1).float().mean().item()),
            "student/grad_norm": grad_norm,
        }
        metrics.update(_summaries("student/teacher_forward_kl", kl_parts))
        return metrics

    # Teacher ---------------------------------------------------------------

    def teacher_update(self, batch: BatchLike, rollouts: RolloutBatch) -> dict[str, float]:
        """One AdamW step on the teacher objective, scored on the student rollouts.

        Student logits are recomputed after the student step (``sg[p_S]``).
        The residual objective also scores the frozen initial model with and
        without the reference.
        """

        objective = self.config.teacher.objective
        examples = _as_examples(batch)
        teacher_prompts = [self._render_teacher(examples[int(index.item())]) for index in rollouts.example_indices]
        unprivileged_prompts = [self._render_student(examples[int(index.item())]) for index in rollouts.example_indices]
        clip = self.config.teacher.clip
        clip_threshold = self.config.teacher_clip_threshold
        device = model_device(self.model)
        completion_lengths = rollouts.completion_mask.to(device).sum(1).long().cpu().tolist()
        self.teacher_optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        kl_parts: list[torch.Tensor] = []
        target_student_parts: list[torch.Tensor] = []
        current_js_parts: list[torch.Tensor] = []
        initial_js_parts: list[torch.Tensor] = []
        for start, stop, micro, token_start, token_stop, mask in self._micro_segments(
            rollouts, self.config.teacher.micro_batch_size
        ):
            segment = dict(rollouts=micro, token_start=token_start, token_stop=token_stop)
            with torch.no_grad():
                student_logits = self._forward_completion(
                    adapter_name=self.adapters.student, prompts=micro.prompt_texts, **segment
                )
            if objective == "residual":
                with torch.no_grad():
                    initial_privileged = self._forward_completion(
                        adapter_name=None, prompts=teacher_prompts[start:stop], **segment
                    )
                    initial_unprivileged = self._forward_completion(
                        adapter_name=None, prompts=unprivileged_prompts[start:stop], **segment
                    )
                    current_unprivileged = self._forward_completion(
                        adapter_name=self.adapters.teacher, prompts=unprivileged_prompts[start:stop], **segment
                    )
                teacher_logits = self._forward_completion(
                    adapter_name=self.adapters.teacher, prompts=teacher_prompts[start:stop], **segment
                )
                kl_values, target_to_student = privileged_residual_target_kl_values(
                    teacher_logits,
                    student_logits,
                    initial_privileged,
                    initial_unprivileged,
                    mask,
                    beta=self.config.teacher.residual_beta,
                    pointwise_clip=clip_threshold,
                    clip_region="low" if clip == "low" else "high",
                )
                with torch.no_grad():
                    target_student_parts.append(_valid(target_to_student, mask))
                    current_js_parts.append(
                        _valid(js_divergence_values(teacher_logits.detach(), current_unprivileged, mask), mask)
                    )
                    initial_js_parts.append(
                        _valid(js_divergence_values(initial_privileged, initial_unprivileged, mask), mask)
                    )
                del initial_privileged, initial_unprivileged, current_unprivileged, target_to_student
            else:
                teacher_logits = self._forward_completion(
                    adapter_name=self.adapters.teacher, prompts=teacher_prompts[start:stop], **segment
                )
                kl_values = opsd_forward_kl_values(teacher_logits, student_logits, mask)
            micro_loss = masked_mean(kl_values, mask)
            weight = self._loss_weight(
                mask,
                completion_lengths=completion_lengths,
                start=start,
                stop=stop,
                group_size=self.config.optim.loss_group_size,
            )
            weighted_loss = micro_loss * weight
            weighted_loss.backward()
            loss_value += float(weighted_loss.detach().item())
            kl_parts.append(_valid(kl_values, mask))
            del student_logits, teacher_logits, kl_values, micro_loss, weighted_loss, mask
        grad_norm = self._step(self.teacher_optimizer, self.teacher_scheduler, self.adapters.teacher)
        self.state.teacher_steps += 1

        metrics = {
            "teacher/loss": loss_value,
            "teacher/opd_loss": loss_value,
            "teacher/grad_norm": grad_norm,
            "teacher/learning_rate": float(self.teacher_optimizer.param_groups[0]["lr"]),
            "teacher/privileged_residual_beta": float(
                self.config.teacher.residual_beta if objective == "residual" else 0.0
            ),
        }
        if objective == "residual":
            # Clipped value under the teacher clipping ablation.
            metrics.update(_summaries("teacher/privileged_target_forward_kl", kl_parts))
            metrics.update(_summaries("teacher/target_to_student_forward_kl", target_student_parts))
            metrics.update(_summaries("teacher/current_privileged_js", current_js_parts))
            metrics.update(_summaries("teacher/initial_privileged_js", initial_js_parts))
            initial = metrics["teacher/initial_privileged_js_mean"]
            metrics["teacher/privileged_js_retention"] = (
                metrics["teacher/current_privileged_js_mean"] / initial if initial > 0 else 0.0
            )
        else:
            metrics.update(_summaries("teacher/student_reverse_clipped_kl", kl_parts))
        return metrics

    # Loop --------------------------------------------------------------------

    def train_step(self, batch: BatchLike) -> dict[str, float]:
        started = time.monotonic()
        tokens_before = self.state.generated_student_tokens
        metrics: dict[str, float] = {}

        def phase(name: str, operation: Callable[[], Any]) -> Any:
            result, phase_metrics = self._run_phase(name, operation)
            if isinstance(result, dict):
                metrics.update(result)
            metrics.update(phase_metrics)
            return result

        rollouts = phase("student_rollout", lambda: self.student_rollout(batch))
        phase("student_rollout_sleep", self._sleep_rollout_engine)
        phase("student_verifier_dump", lambda: self.dump_student_rollouts(batch, rollouts))
        phase("student_update", lambda: self.student_update(batch, rollouts))
        if self.config.teacher.objective != "frozen":
            phase("dual_teacher_update", lambda: self.teacher_update(batch, rollouts))
        del rollouts
        phase("student_release", self._release_memory)
        return self._finalize_train_step(started=started, tokens_before=tokens_before, metrics=metrics)

    def _finalize_train_step(self, *, started: float, tokens_before: int, metrics: dict[str, float]) -> dict[str, float]:
        self.state.global_step += 1
        elapsed = max(time.monotonic() - started, 1e-9)
        metrics.update(
            {
                "rollout/student_tokens": float(self.state.generated_student_tokens),
                "system/tokens_per_second": float((self.state.generated_student_tokens - tokens_before) / elapsed),
                "train/global_step": float(self.state.global_step),
            }
        )
        engine_metrics = getattr(self.rollout_engine, "runtime_metrics", None)
        if engine_metrics is not None:
            metrics.update(engine_metrics())
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            mib = 1024**2
            metrics.update(
                {
                    "system/cuda_free_mib": free_bytes / mib,
                    "system/cuda_total_mib": total_bytes / mib,
                    "system/cuda_peak_allocated_mib": max(
                        (value for key, value in metrics.items() if key.endswith("_peak_allocated_mib")),
                        default=torch.cuda.max_memory_allocated() / mib,
                    ),
                    "system/cuda_peak_reserved_mib": max(
                        (value for key, value in metrics.items() if key.endswith("_peak_reserved_mib")),
                        default=torch.cuda.max_memory_reserved() / mib,
                    ),
                }
            )
        self.last_metrics = metrics
        if self.metric_sink is not None:
            self.metric_sink(metrics)
        return metrics

    def fit(self, batches: Iterable[BatchLike], output_dir: str | Path | None = None) -> TrainerState:
        """Train until ``schedule.max_steps``; one data batch per step, resumable by step."""

        schedule = self.config.schedule
        iterator = _infinite(batches)
        for _ in range(self.state.global_step):
            next(iterator)
        while self.state.global_step < schedule.max_steps:
            metrics = self.train_step(next(iterator))
            LOGGER.info("step=%d metrics=%s", self.state.global_step, metrics)
            if output_dir is not None and self.state.global_step % schedule.save_steps == 0:
                save_checkpoint(
                    Path(output_dir) / f"checkpoint-{self.state.global_step}",
                    model=self.model,
                    student_optimizer=self.student_optimizer,
                    teacher_optimizer=self.teacher_optimizer,
                    student_scheduler=self.student_scheduler,
                    teacher_scheduler=self.teacher_scheduler,
                    state=self.state,
                    config=self.config,
                    save_training_state=self.state.global_step % schedule.optimizer_save_steps == 0,
                )
        return self.state


def _infinite(batches: Iterable[BatchLike]) -> Iterator[BatchLike]:
    while True:
        consumed = False
        for batch in batches:
            consumed = True
            yield batch
        if not consumed:
            raise ValueError("training batch iterable is empty")
