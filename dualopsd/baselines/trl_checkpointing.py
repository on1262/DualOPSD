"""Sparse optimizer-state retention and chained-resume support for TRL trainers.

Adapters are saved every ``save_steps``; optimizer/scheduler/RNG state is kept only for
the latest multiple of ``interval``. ``--stop-after-step`` ends a job at an optimizer
checkpoint while keeping the full ``max_steps`` schedule, so a follow-up job resumes
with optimizer and learning-rate state identical to an uninterrupted run.
"""

from __future__ import annotations

from pathlib import Path

from transformers import TrainerCallback

_STATE_PATTERNS = ("optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state.pth", "rng_state_*.pth")


def _remove_optimizer_state(checkpoint: Path) -> None:
    for pattern in _STATE_PATTERNS:
        for path in checkpoint.glob(pattern):
            path.unlink(missing_ok=True)


def latest_optimizer_checkpoint(output_dir: str) -> str | None:
    """Latest ``checkpoint-N`` under ``output_dir`` that still holds optimizer state."""
    checkpoints = sorted(Path(output_dir).glob("checkpoint-*"), key=lambda path: int(path.name.rsplit("-", 1)[-1]))
    valid = [path for path in checkpoints if (path / "optimizer.pt").is_file()]
    return str(valid[-1]) if valid else None


class SparseOptimizerCheckpointCallback(TrainerCallback):
    def __init__(self, interval: int = 25):
        self.interval = interval

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        current = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if state.global_step % self.interval:
            _remove_optimizer_state(current)
        else:
            for checkpoint in Path(args.output_dir).glob("checkpoint-*"):
                if checkpoint != current:
                    _remove_optimizer_state(checkpoint)
        return control


class StopAfterStepCallback(TrainerCallback):
    """Save and stop at an absolute step without changing ``max_steps``."""

    def __init__(self, stop_step: int):
        if stop_step < 1:
            raise ValueError("stop_step must be positive")
        self.stop_step = stop_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_step and state.global_step < state.max_steps:
            control.should_save = True
            control.should_training_stop = True
        return control


def resumable_callbacks(*, optimizer_interval: int = 25, stop_after_step: int | None = None) -> list[TrainerCallback]:
    callbacks: list[TrainerCallback] = [SparseOptimizerCheckpointCallback(interval=optimizer_interval)]
    if stop_after_step:
        if stop_after_step % optimizer_interval:
            raise ValueError(
                f"stop_after_step={stop_after_step} must be divisible by the optimizer interval {optimizer_interval}"
            )
        callbacks.append(StopAfterStepCallback(stop_after_step))
    return callbacks
