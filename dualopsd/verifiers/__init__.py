"""Math answer verifiers.

``verify_math_answer`` is the rule-based check used for rollout correctness logging
during distillation; benchmark evaluation uses ``verify_math_official``.
"""

from __future__ import annotations

import re
from typing import Any

from dualopsd.verifiers.math import verify_math

_TERMINAL_SPECIAL_TOKEN = re.compile(r"(?:<\|[^<>\n]+\|>|</?s>|<pad>)\s*$", flags=re.IGNORECASE)


def strip_terminal_special_tokens(completion: str) -> str:
    """Remove trailing model-control tokens (e.g. ``<|im_end|>``) before grading."""
    cleaned = completion.rstrip()
    while True:
        stripped = _TERMINAL_SPECIAL_TOKEN.sub("", cleaned).rstrip()
        if stripped == cleaned:
            return cleaned
        cleaned = stripped


def verify_math_answer(text: str, reference: Any) -> bool:
    """Rule-based correctness of a completion against a reference final answer."""
    return verify_math(strip_terminal_special_tokens(text), reference)


def verify_example(example: Any, completion: str) -> bool:
    """Verify a training example (any object with ``verifier`` and ``answer`` fields)."""
    if example.verifier != "math":
        raise ValueError(f"unsupported verifier: {example.verifier}")
    return verify_math_answer(completion, example.answer)


__all__ = ["strip_terminal_special_tokens", "verify_example", "verify_math", "verify_math_answer"]
