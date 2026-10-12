"""Token-level KL objectives for the student and teacher updates.

All functions take logits shaped ``[batch, tokens, vocab]`` and return per-token
values shaped ``[batch, tokens]``. Clipping acts on each vocabulary
contribution *before* the vocabulary sum, as in the public OPSD code.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def clip_kl_contributions(contributions: torch.Tensor, threshold: float, *, region: str) -> torch.Tensor:
    """Constantize one side of the per-vocabulary KL contributions.

    ``high`` caps contributions at ``threshold`` (no gradient above it); ``low``
    floors them at ``threshold`` (no gradient below it).
    """

    if threshold <= 0:
        raise ValueError("KL contribution clip threshold must be positive")
    if region == "high":
        return contributions.clamp(max=float(threshold))
    if region == "low":
        return contributions.clamp(min=float(threshold))
    raise ValueError("KL contribution clip region must be 'high' or 'low'")


def _check_aligned(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> None:
    if first.shape != second.shape:
        raise ValueError("logits must be aligned")
    if first.shape[:-1] != mask.shape:
        raise ValueError("completion mask must align with the logits' token dimensions")


def opsd_forward_kl_values(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    pointwise_clip: float | None = None,
) -> torch.Tensor:
    """Per-token ``KL(sg[p_teacher] || p_student)`` with gradients to ``student_logits``.

    With ``pointwise_clip`` this is the public OPSD student loss: each
    vocabulary contribution is upper-clipped before the sum, computed in the
    logits dtype. Without a clip the log-probabilities are computed in FP32.
    The teacher update of DualOPSD w/o residual calls this with the roles
    swapped, i.e. ``KL(sg[p_S] || p_T)``.
    """

    _check_aligned(student_logits, teacher_logits, completion_mask)
    if pointwise_clip is None:
        teacher_logprobs = F.log_softmax(teacher_logits.detach(), dim=-1, dtype=torch.float32)
        student_logprobs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
    else:
        teacher_logprobs = F.log_softmax(teacher_logits.detach(), dim=-1)
        student_logprobs = F.log_softmax(student_logits, dim=-1)
    pointwise = F.kl_div(student_logprobs, teacher_logprobs, reduction="none", log_target=True)
    if pointwise_clip is not None:
        pointwise = clip_kl_contributions(pointwise, float(pointwise_clip), region="high")
    return pointwise.sum(dim=-1)


def opsd_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    pointwise_clip: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the masked token mean of :func:`opsd_forward_kl_values` and detached values."""

    per_token = opsd_forward_kl_values(
        student_logits, teacher_logits, completion_mask, pointwise_clip=pointwise_clip
    )
    return masked_mean(per_token, completion_mask), per_token.detach()


def privileged_residual_target_kl_values(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    initial_privileged_logits: torch.Tensor,
    initial_unprivileged_logits: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    beta: float,
    pointwise_clip: float | None = None,
    clip_region: str = "high",
) -> tuple[torch.Tensor, torch.Tensor]:
    """DualOPSD teacher target: ``KL(sg[q] || p_teacher)`` with
    ``q ∝ p_student * exp(beta * (log p0(.|x,r) - log p0(.|x)))``.

    ``p0`` is the frozen initial model with (privileged) and without
    (unprivileged) the reference. Everything is FP32 and only
    ``teacher_logits`` receives gradients. Also returns ``KL(q || p_student)``.
    With ``pointwise_clip`` (teacher clipping ablation) each vocabulary
    contribution of ``KL(q || p_teacher)`` is clipped on the ``clip_region``
    side before the vocabulary sum, as in :func:`clip_kl_contributions`.
    """

    tensors = (student_logits, initial_privileged_logits, initial_unprivileged_logits)
    if any(tensor.shape != teacher_logits.shape for tensor in tensors):
        raise ValueError("all privileged-residual logits must be aligned")
    if teacher_logits.shape[:-1] != completion_mask.shape:
        raise ValueError("completion mask must align with privileged-residual logits")
    if beta < 0:
        raise ValueError("privileged residual beta cannot be negative")
    if pointwise_clip is not None and pointwise_clip <= 0:
        raise ValueError("pointwise_clip must be positive when enabled")

    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    with torch.no_grad():
        student_logprobs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
        privileged = F.log_softmax(initial_privileged_logits, dim=-1, dtype=torch.float32)
        unprivileged = F.log_softmax(initial_unprivileged_logits, dim=-1, dtype=torch.float32)
        target_logprobs = F.log_softmax(student_logprobs + float(beta) * (privileged - unprivileged), dim=-1)
        target_probs = target_logprobs.exp()
        target_to_student = (target_probs * (target_logprobs - student_logprobs)).sum(dim=-1)
    pointwise = target_probs * (target_logprobs - teacher_logprobs)
    if pointwise_clip is not None:
        pointwise = clip_kl_contributions(pointwise, float(pointwise_clip), region=clip_region)
    return pointwise.sum(dim=-1), target_to_student


def js_divergence_values(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token Jensen-Shannon divergence in FP32."""

    _check_aligned(first_logits, second_logits, completion_mask)
    first_logprobs = F.log_softmax(first_logits, dim=-1, dtype=torch.float32)
    second_logprobs = F.log_softmax(second_logits, dim=-1, dtype=torch.float32)
    mixture = torch.logaddexp(first_logprobs, second_logprobs) - torch.log(first_logprobs.new_tensor(2.0))
    first_kl = (first_logprobs.exp() * (first_logprobs - mixture)).sum(dim=-1)
    second_kl = (second_logprobs.exp() * (second_logprobs - mixture)).sum(dim=-1)
    return 0.5 * (first_kl + second_kl)
