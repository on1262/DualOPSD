"""Terminal-answer extraction for the model-based answer judge."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class TerminalAnswer:
    text: str
    mode: str


def last_balanced_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    starts = [match.start() for match in re.finditer(re.escape(marker), text)]
    for position in reversed(starts):
        content_start = position + len(marker)
        depth = 1
        cursor = content_start
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            return re.sub(r"\s+", " ", text[content_start : cursor - 1]).strip()
    return None


def _terminal_sentences(text: str) -> list[str]:
    clean = text.strip()
    while clean.endswith("<|im_end|>"):
        clean = clean[: -len("<|im_end|>")].rstrip()
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    while lines and lines[-1] in {"$$", "\\]", "---"}:
        lines.pop()
    if not lines:
        return []
    flattened = re.sub(r"\s+", " ", " ".join(lines)).strip()
    return [
        piece.strip()
        # Do not split the conventional multiple-choice prefix ``A.``--``E.``.
        for piece in re.split(r"(?<![A-E]\.)(?<=[.!?。！？])\s+", flattened)
        if piece.strip()
    ]


def extract_terminal_answer(text: str, *, min_chars: int = 10) -> TerminalAnswer:
    """Extract the last box or enough trailing sentences to reach ``min_chars``.

    An explicit boxed answer is kept even when shorter than ``min_chars``. For
    prose, preceding sentences are prepended until the
    requested context length is reached. If the response has no terminal text,
    an empty value is returned and the downstream judge is instructed to reject
    it as lacking an obvious answer.
    """

    boxed = last_balanced_boxed(text)
    if boxed is not None:
        return TerminalAnswer(boxed, "last_boxed")
    sentences = _terminal_sentences(text)
    if not sentences:
        return TerminalAnswer("", "last_sentence")
    selected = [sentences[-1]]
    for sentence in reversed(sentences[:-1]):
        if len(" ".join(selected).strip()) >= min_chars:
            break
        selected.insert(0, sentence)
    return TerminalAnswer(" ".join(selected).strip(), "last_sentence")


def parse_boxed_yes_no(text: str) -> str | None:
    boxed = last_balanced_boxed(text)
    if boxed is None:
        return None
    value = boxed.strip().casefold()
    return value if value in {"yes", "no"} else None
