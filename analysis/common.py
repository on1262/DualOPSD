"""Shared paths, readers, statistics and table helpers for the analysis modules.

Every input is located through the scratch layout rooted at ``$DUALOPSD_SCRATCH``::

    runs/<exp>/<model>/<run>/                         training outputs
    eval/<exp>/<model>/<run>/<benchmark>/samples.jsonl
    generations/<exp>/<model>/<run>_step<N>/<policy>[_judged].jsonl
    diagnostics/<exp>/...                             post-hoc / recurrence outputs
    results/<exp>/                                    outputs written by this package
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Iterable, Sequence

MODELS = ("qwen3-1.7b", "qwen3-4b", "qwen3-8b")
MODEL_LABELS = {"qwen3-1.7b": "1.7B", "qwen3-4b": "4B", "qwen3-8b": "8B"}
SEEDS = (42, 43, 44)
BENCHMARKS = ("aime24", "aime25", "hmmt25")
CHECKPOINT_STEPS = (20, 40, 60, 80, 100)
# Residual weights selected on the held-out set (Appendix "Residual weight selection").
SELECTED_BETA = {"qwen3-1.7b": 0.8, "qwen3-4b": 0.7, "qwen3-8b": 0.5}

_SCRATCH_OVERRIDE: Path | None = None


# ----------------------------------------------------------------------------- paths


def set_scratch(path: str | Path | None) -> None:
    """Override ``$DUALOPSD_SCRATCH`` for this process (used by ``--scratch``)."""
    global _SCRATCH_OVERRIDE
    _SCRATCH_OVERRIDE = Path(path) if path else None


def scratch() -> Path:
    if _SCRATCH_OVERRIDE is not None:
        return _SCRATCH_OVERRIDE
    value = os.environ.get("DUALOPSD_SCRATCH")
    if not value:
        raise RuntimeError("DUALOPSD_SCRATCH is not set; source configs/user.env or pass --scratch")
    return Path(value)


def beta_tag(beta: float) -> str:
    """0.7 -> '0p7', 1.0 -> '1p0'."""
    return f"{beta:.1f}".replace(".", "p")


def run_name(method: str, seed: int, *, beta: float | None = None, step: int | None = None) -> str:
    name = method if beta is None else f"{method}_beta{beta_tag(beta)}"
    name = f"{name}_seed{seed}"
    return name if step is None else f"{name}_step{step}"


def base_run(exp: str, model: str, seed: int) -> str:
    """Base-model evaluation run for one generation seed (seed 42 may be stored as ``base``)."""
    name = f"base_genseed{seed}"
    if seed == 42 and not eval_dir(exp, model, name).exists():
        return "base"
    return name


def run_dir(exp: str, model: str, run: str) -> Path:
    return scratch() / "runs" / exp / model / run


def eval_dir(exp: str, model: str, run: str) -> Path:
    return scratch() / "eval" / exp / model / run


def eval_file(exp: str, model: str, run: str, benchmark: str) -> Path:
    return eval_dir(exp, model, run) / benchmark / "samples.jsonl"


def generation_file(
    exp: str, model: str, run: str, step: int, policy: str, *, judged: bool = False
) -> Path:
    suffix = "_judged" if judged else ""
    return scratch() / "generations" / exp / model / f"{run}_step{step}" / f"{policy}{suffix}.jsonl"


def diagnostics_dir(exp: str, *parts: str) -> Path:
    return scratch().joinpath("diagnostics", exp, *parts)


def results_dir(exp: str) -> Path:
    return scratch() / "results" / exp


def heldout_file() -> Path:
    """The 500-problem held-out set written by ``prepare-data`` (``$HELDOUT_FILE``)."""
    value = os.environ.get("HELDOUT_FILE") or (
        os.environ.get("DATA_DIR") and str(Path(os.environ["DATA_DIR"]) / "heldout_500.jsonl"))
    if not value:
        raise RuntimeError("HELDOUT_FILE is not set; source configs/user.env or pass --heldout-file")
    return Path(value)


def find_step_run(exp: str, model: str, prefix: str) -> tuple[str, int]:
    """Return the unique ``<prefix>_step<N>`` evaluation run (e.g. the matched OPSD checkpoint)."""
    matches = sorted((scratch() / "eval" / exp / model).glob(f"{prefix}_step*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {prefix}_step* under eval/{exp}/{model}, found {len(matches)}")
    return matches[0].name, int(matches[0].name.rsplit("_step", 1)[1])


# ------------------------------------------------------------------------------ I/O


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check_target(path: Path) -> None:
    try:
        root = scratch()
    except RuntimeError:
        return
    resolved_root = root.resolve()
    path = path.absolute()
    for candidate in (path, *path.parents):
        if candidate == root or root not in candidate.parents:
            break
        if candidate.is_symlink() and resolved_root not in candidate.resolve().parents:
            raise PermissionError(f"refusing to write through symlink leaving the scratch root: {candidate}")


def write_text(path: Path, text: str, *, overwrite: bool = False) -> Path:
    path = Path(path)
    _check_target(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> Path:
    return write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n", overwrite=overwrite)


# ------------------------------------------------------------------------ statistics


def mean_sd(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    return {"mean": mean(values), "sample_sd": stdev(values)}


def nearest_rank(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


@lru_cache(maxsize=None)
def summarize_samples(path: Path, num_problems: int = 30, num_samples: int = 12) -> dict[str, Any]:
    """avg@k / pass@k / length statistics of one benchmark evaluation file."""
    rows = read_jsonl(path)
    keys = {(str(row["problem_id"]), int(row["sample_index"])) for row in rows}
    expected = num_problems * num_samples
    if len(rows) != expected or len(keys) != expected:
        raise ValueError(f"expected {expected} unique samples: {path}")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["problem_id"]), []).append(row)
    if len(groups) != num_problems or any(len(group) != num_samples for group in groups.values()):
        raise ValueError(f"expected {num_problems} problems x {num_samples} samples: {path}")
    lengths = [int(row["completion_tokens"]) for row in rows]
    nontruncated = [int(row["completion_tokens"]) for row in rows if not bool(row["truncated"])]
    if not nontruncated:
        raise ValueError(f"all samples truncated: {path}")
    signature = hashlib.sha256(json.dumps(sorted(
        (str(row["problem_id"]), int(row["sample_index"]), str(row["problem"]), str(row["answer"]))
        for row in rows
    )).encode()).hexdigest()
    return {
        "source": str(path),
        "avg_at_12": mean(bool(row["correct"]) for row in rows),
        "pass_at_12": mean(any(bool(row["correct"]) for row in group) for group in groups.values()),
        "observed_mean_completion_tokens": mean(lengths),
        "nontruncated_mean_completion_tokens": mean(nontruncated),
        "p99_completion_tokens_nearest_rank": nearest_rank(lengths, 0.99),
        "truncation_rate": mean(bool(row["truncated"]) for row in rows),
        "_nontruncated_sum": sum(nontruncated),
        "_nontruncated_count": len(nontruncated),
        "_signature": signature,
    }


RUN_METRICS = (
    "macro_avg_at_12",
    "macro_pass_at_12",
    "observed_mean_completion_tokens",
    "nontruncated_mean_completion_tokens",
    "nontruncated_mean_completion_tokens_macro",
    "truncation_rate",
)
DATASET_METRICS = (
    "avg_at_12",
    "pass_at_12",
    "observed_mean_completion_tokens",
    "nontruncated_mean_completion_tokens",
    "truncation_rate",
)


def summarize_run(paths: dict[str, Path]) -> dict[str, Any]:
    """Combine per-benchmark summaries of one checkpoint.

    ``nontruncated_mean_completion_tokens`` pools all responses; the ``_macro`` variant averages
    the three per-benchmark untruncated means.  Observed tokens and truncation are identical
    under both conventions because every benchmark has the same number of responses.
    """
    datasets = {name: summarize_samples(Path(path)) for name, path in paths.items()}
    names = list(datasets)
    return {
        "datasets": {name: {k: v for k, v in row.items() if not k.startswith("_")} for name, row in datasets.items()},
        "macro_avg_at_12": mean(datasets[name]["avg_at_12"] for name in names),
        "macro_pass_at_12": mean(datasets[name]["pass_at_12"] for name in names),
        "observed_mean_completion_tokens": mean(datasets[name]["observed_mean_completion_tokens"] for name in names),
        "nontruncated_mean_completion_tokens": (
            sum(datasets[name]["_nontruncated_sum"] for name in names)
            / sum(datasets[name]["_nontruncated_count"] for name in names)
        ),
        "nontruncated_mean_completion_tokens_macro": mean(
            datasets[name]["nontruncated_mean_completion_tokens"] for name in names
        ),
        "truncation_rate": mean(datasets[name]["truncation_rate"] for name in names),
        "_signatures": {name: datasets[name]["_signature"] for name in names},
    }


def aggregate_seeds(path_for: Callable[[int, str], Path], seeds: Sequence[int] = SEEDS,
                    benchmarks: Sequence[str] = BENCHMARKS) -> dict[str, Any]:
    """Per-seed summaries and mean / sample SD across seeds for one method."""
    per_seed = {str(seed): summarize_run({b: path_for(seed, b) for b in benchmarks}) for seed in seeds}
    check_same_problems(per_seed.values())
    for row in per_seed.values():
        row.pop("_signatures")
    across = {metric: mean_sd(per_seed[str(seed)][metric] for seed in seeds) for metric in RUN_METRICS}
    across["datasets"] = {
        b: {m: mean_sd(per_seed[str(seed)]["datasets"][b][m] for seed in seeds) for m in DATASET_METRICS}
        for b in benchmarks
    }
    return {"per_seed": per_seed, "across_seeds": across}


def check_same_problems(runs: Iterable[dict[str, Any]]) -> None:
    reference: dict[str, str] = {}
    for run in runs:
        for name, signature in run.get("_signatures", {}).items():
            if reference.setdefault(name, signature) != signature:
                raise ValueError(f"benchmark problems differ across runs for {name}")


# ------------------------------------------------------------------------ table output


def pm(stats: dict[str, float], scale: float = 100.0, digits: int = 2) -> str:
    return f"{scale * stats['mean']:.{digits}f}±{scale * stats['sample_sd']:.{digits}f}"


def to_latex(cells: Sequence[str]) -> str:
    """Render normalized cells (``a±b``, ``1.2K``) as a LaTeX table row."""
    def render(cell: str) -> str:
        if "±" not in cell and not re.fullmatch(r"[+-][0-9.]+K?", cell):
            return cell
        return "$" + cell.replace("±", r"\pm").replace("K", r"\mathrm{K}") + "$"
    return " & ".join(render(cell) for cell in cells) + r" \\"


def print_tables(tables: dict[str, list[list[str]]]) -> None:
    for label, rows in tables.items():
        print(f"% {label}")
        for row in rows:
            print(to_latex(row))
        print()


def add_common_args(parser: argparse.ArgumentParser, exp: str, output_help: str | None = None) -> None:
    parser.add_argument("--scratch", type=Path, default=None, help="override $DUALOPSD_SCRATCH")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=output_help or f"default: $DUALOPSD_SCRATCH/results/{exp}")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    parser.add_argument("--print-tables", action="store_true", help="print LaTeX-ready paper table rows")


def output_dir(args: argparse.Namespace, exp: str) -> Path:
    set_scratch(args.scratch)
    return Path(args.output_dir) if args.output_dir else results_dir(exp)
