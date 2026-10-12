"""Public OPSD prompt templates and completion-logit alignment.

The student sees the problem only; the privileged teacher additionally sees the
reference solution. Both are rendered with the model chat template in
non-thinking mode.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch

TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n"
)
REFERENCE_PREFIX = "Here is a reference solution to this problem:\n=== Reference Solution Begin ===\n"
REFERENCE_SUFFIX = (
    "\n=== Reference Solution End ===\n"
    f"{TRANSITION_PROMPT}\n"
    "Please reason step by step, and put your final answer within \\boxed{}."
)
STUDENT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def generation_eos_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Return all terminal token ids attached to ``tokenizer`` (falls back to EOS)."""

    values = getattr(tokenizer, "_generation_eos_token_ids", None)
    if values is None:
        value = getattr(tokenizer, "eos_token_id", None)
        values = [] if value is None else [value]
    return tuple(dict.fromkeys(int(value) for value in values if value is not None))


def upstream_student_messages(problem: str) -> list[dict[str, str]]:
    """The public OPSD student prompt (problem only)."""

    return [{"role": "user", "content": f"Problem: {problem}{STUDENT_SUFFIX}"}]


def upstream_teacher_messages(problem: str, reference: str) -> list[dict[str, str]]:
    """The public OPSD privileged-teacher prompt (problem + reference solution)."""

    return [
        {
            "role": "user",
            "content": f"Problem: {problem}\n\n{REFERENCE_PREFIX}{reference}{REFERENCE_SUFFIX}",
        }
    ]


def render_chat(tokenizer: Any, messages: Iterable[dict[str, str]], *, thinking: bool = False) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": thinking}
    try:
        return tokenizer.apply_chat_template(list(messages), **kwargs)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(list(messages), **kwargs)


def _fit_reference(tokenizer: Any, reference: str, render, max_tokens: int | None) -> str:
    """Render ``reference`` and shorten its middle until the prompt fits ``max_tokens``."""

    prompt = render(reference)
    if max_tokens is None or len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) <= max_tokens:
        return prompt
    fixed_tokens = len(tokenizer(render(""), add_special_tokens=False)["input_ids"])
    available = max(0, max_tokens - fixed_tokens - 8)
    reference_ids = tokenizer(str(reference), add_special_tokens=False)["input_ids"]

    def shortened(count: int) -> str:
        if count <= 0:
            return ""
        head = count // 2
        tail = count - head
        kept = reference_ids[:head] + reference_ids[-tail:] if tail else reference_ids[:head]
        decoded = tokenizer.decode(kept, skip_special_tokens=False)
        middle = len(decoded) // 2
        return decoded[:middle] + "\n...[reference truncated]...\n" + decoded[middle:]

    while True:
        prompt = render(shortened(available))
        length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if length <= max_tokens:
            return prompt
        if available == 0:
            raise ValueError("problem and chat template exceed the configured prompt token budget")
        available = max(0, available - (length - max_tokens) - 8)


def _check_template(upstream_prompt_template: bool) -> None:
    if not upstream_prompt_template:
        raise ValueError("only the public OPSD prompt template is supported")


def build_student_prompt(
    tokenizer: Any,
    problem: str,
    *,
    domain: str | None = None,
    thinking: bool = False,
    upstream_prompt_template: bool = True,
) -> str:
    """Render the student prompt. ``domain`` is accepted for call compatibility (math only)."""

    del domain
    _check_template(upstream_prompt_template)
    return render_chat(tokenizer, upstream_student_messages(problem), thinking=thinking)


def build_teacher_prompt(
    tokenizer: Any,
    problem: str,
    reference: str,
    *,
    domain: str | None = None,
    thinking: bool = False,
    max_tokens: int | None = None,
    upstream_prompt_template: bool = True,
) -> str:
    """Render the privileged prompt, truncating the reference to ``max_tokens`` if needed."""

    del domain
    _check_template(upstream_prompt_template)
    return _fit_reference(
        tokenizer,
        reference,
        lambda fitted: render_chat(tokenizer, upstream_teacher_messages(problem, fitted), thinking=thinking),
        max_tokens,
    )


def completion_logits(logits: torch.Tensor, prompt_lengths: torch.Tensor, completion_length: int) -> torch.Tensor:
    """Select the causal logits that predict each completion token.

    ``logits[:, j]`` predicts input token ``j + 1``; returns ``[batch, completion, vocab]``.
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if prompt_lengths.ndim != 1 or prompt_lengths.shape[0] != logits.shape[0]:
        raise ValueError("prompt_lengths must contain one value per logit row")
    if completion_length < 0:
        raise ValueError("completion_length cannot be negative")
    if torch.any(prompt_lengths < 1):
        raise ValueError("every prompt must contain at least one token")
    offsets = prompt_lengths.to(logits.device)[:, None] - 1
    positions = offsets + torch.arange(completion_length, device=logits.device)[None, :]
    if positions.numel() and int(positions.max()) >= logits.shape[1]:
        raise ValueError("logits do not cover the requested completion span")
    batch = torch.arange(logits.shape[0], device=logits.device)[:, None]
    return logits[batch, positions]


def append_completions(
    prompt_ids: list[list[int]],
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    pad_token_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-pad prompts so every completion starts at one shared position."""

    prompt_width = max(map(len, prompt_ids))
    inputs = torch.full(
        (len(prompt_ids), prompt_width + completion_ids.shape[1]), pad_token_id, dtype=torch.long, device=device
    )
    attention = torch.zeros_like(inputs)
    for row, prompt in enumerate(prompt_ids):
        start = prompt_width - len(prompt)
        inputs[row, start:prompt_width] = torch.tensor(prompt, dtype=torch.long, device=device)
        attention[row, start:prompt_width] = 1
        n_completion = int(completion_mask[row].sum().item())
        if n_completion:
            inputs[row, prompt_width : prompt_width + n_completion] = completion_ids[row, :n_completion]
            attention[row, prompt_width : prompt_width + n_completion] = 1
    prompt_lengths = torch.full((len(prompt_ids),), prompt_width, dtype=torch.long, device=device)
    return inputs, attention, prompt_lengths
