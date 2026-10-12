"""Evaluation: benchmark avg@k, student-LoRA merging, held-out generation and answer judging.

Shared helpers for safe output writing live here; they import nothing heavy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows if limit is None else rows[:limit]


def write_jsonl_atomic(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def escapes_scratch(path: Path) -> bool:
    """True when ``path`` lies under $DUALOPSD_SCRATCH but resolves (via a symlink) outside it."""
    scratch = os.environ.get("DUALOPSD_SCRATCH")
    if not scratch:
        return False
    lexical_root = Path(os.path.abspath(scratch))
    real_root = Path(os.path.realpath(scratch))
    lexical = Path(os.path.abspath(path))
    if lexical != lexical_root and lexical_root not in lexical.parents:
        return False
    real = Path(os.path.realpath(lexical))
    return real != real_root and real_root not in real.parents


def check_output(path: str | Path, *, overwrite: bool = False) -> Path:
    """Validate an output file path and create its parent directory.

    Refuses to replace an existing file unless ``overwrite`` and refuses any path under
    ``$DUALOPSD_SCRATCH`` that resolves through a symlink to a location outside it.
    """
    path = Path(path)
    if escapes_scratch(path):
        raise PermissionError(f"refusing to write through a symlink leaving $DUALOPSD_SCRATCH: {path}")
    if (path.exists() or path.is_symlink()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path} (pass --overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def shutdown_llm(llm: Any) -> None:
    """Explicitly stop vLLM's EngineCore before multiprocessing's atexit hook joins it."""
    engine = getattr(llm, "llm_engine", None)
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()
        return
    engine_core = getattr(engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if callable(shutdown):
        shutdown()
