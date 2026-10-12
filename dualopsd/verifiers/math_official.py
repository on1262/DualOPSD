"""Benchmark grading with the official ``math_verify`` package and a rule-based fallback."""

import re

from math_verify import LatexExtractionConfig, parse, verify

from dualopsd.verifiers.math import extract_boxed_or_final_answer, verify_math


def extract_last_boxed(text: str) -> str:
    starts = [match.start() for match in re.finditer(r"\\boxed\{", text)]
    if not starts:
        return extract_boxed_or_final_answer(text) or text
    start = starts[-1] + len(r"\boxed{")
    depth = 1
    cursor = start
    while cursor < len(text) and depth:
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
        cursor += 1
    return (
        text[start : cursor - 1]
        if depth == 0
        else (extract_boxed_or_final_answer(text) or text)
    )


def verify_math_official(response: str, reference: str) -> bool:
    """Grade the last boxed answer with math_verify; fall back to the rule-based verifier."""
    prediction = extract_last_boxed(response)
    try:
        gold = parse(
            f"${reference}$",
            extraction_config=[LatexExtractionConfig()],
            fallback_mode="no_fallback",
        )
        answer = parse(
            f"${prediction}$",
            extraction_config=[LatexExtractionConfig()],
            fallback_mode="no_fallback",
        )
        if gold and answer:
            return bool(verify(gold, answer, timeout_seconds=5))
    except Exception:
        pass
    return verify_math(response, reference)
