"""Build the training file, the 500-problem held-out set and the AIME24/AIME25/HMMT25 benchmarks.

    python -m dualopsd prepare-data all          # = train, heldout, benchmarks

Every source is pinned to a Hugging Face revision. Each written file is checked against
the sha256 recorded in the tracked ``data/manifest.json`` (``--update-manifest`` records
it instead).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

OPENTHOUGHTS_REPO = "siyanzhao/Openthoughts_math_30k_opsd"
OPENTHOUGHTS_REVISION = "1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62"
OPENTHOUGHTS_ROWS = 29_434
TRAIN_FILE = "train_openthoughts_30k.jsonl"
HELDOUT_FILE = "heldout_500.jsonl"

# Benchmark sources: (repo, revision, file in the repo, id rule, problem column)
BENCHMARK_SOURCES = {
    "aime24": ("HuggingFaceH4/aime_2024", "2fe88a2f1091d5048c0f36abc874fb997b3dd99a", "data/train-00000-of-00001.parquet", "id", "problem"),
    "aime25": ("MathArena/aime_2025", "87387eacfd8461d7b3bf33319653eaa6d3e3eba2", "data/train-00000-of-00001.parquet", "index", "problem"),
    "hmmt25": ("FlagEval/HMMT_2025", "0b2a3e6eba63082872a0daa6d1173a5067536c8f", "hmmt_25.jsonl", "id", "question"),
}


def repo_root() -> Path:
    return Path(os.environ.get("DUALOPSD_REPO") or Path(__file__).resolve().parents[2])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl_lines(rows: Iterable[dict[str, Any]]) -> Iterable[str]:
    return (json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


# ----------------------------------------------------------------------------- training data


def convert_openthoughts(record: dict, index: int) -> dict:
    return {
        "id": f"openthoughts_{index:05d}",
        "domain": "math",
        "problem": str(record["problem"]),
        "reference": str(record["solution"]),
        "answer": str(record.get("Answer") or "").strip() or None,
        "verifier": "math",
        "meta": {
            "source": "openthoughts_math_30k_opsd",
            "source_index": index,
            "source_subset": record.get("source"),
            "generated_token_count": record.get("generated_token_count"),
            "source_correct": record.get("correct"),
            "dataset_repo": OPENTHOUGHTS_REPO,
            "dataset_revision": OPENTHOUGHTS_REVISION,
        },
    }


def openthoughts_rows(shuffle_seed: int = 42) -> tuple[list[dict], dict[str, Any]]:
    """Deterministically shuffled OpenThoughts rows (training order; the loader does not reshuffle)."""
    from datasets import load_dataset

    dataset = load_dataset(OPENTHOUGHTS_REPO, revision=OPENTHOUGHTS_REVISION, split="train")
    if len(dataset) != OPENTHOUGHTS_ROWS:
        raise RuntimeError(f"expected {OPENTHOUGHTS_ROWS} rows at {OPENTHOUGHTS_REVISION}, found {len(dataset)}")
    indices = list(range(len(dataset)))
    random.Random(shuffle_seed).shuffle(indices)
    rows = [convert_openthoughts(dataset[index], index) for index in indices]
    info = {"source": OPENTHOUGHTS_REPO, "revision": OPENTHOUGHTS_REVISION, "shuffle_seed": shuffle_seed}
    return rows, info


# ----------------------------------------------------------------------------- held-out set


def select_heldout(rows: list[dict], *, tail_size: int = 20_000, count: int = 500, seed: int = 42) -> tuple[list[dict], list[int]]:
    """Sorted sample without replacement from the last ``tail_size`` rows (never reached by 100 x 32 training prompts)."""
    if len(rows) < tail_size:
        raise ValueError("training set is smaller than the requested tail")
    if count > tail_size:
        raise ValueError("sample count cannot exceed tail size")
    tail_start = len(rows) - tail_size
    indices = sorted(random.Random(seed).sample(range(tail_start, len(rows)), count))
    return [rows[index] for index in indices], indices


# ----------------------------------------------------------------------------- benchmarks


def benchmark_rows(name: str) -> tuple[list[dict], dict[str, Any]]:
    """Rows ``{"id", "problem", "answer"}``; ids match the ``problem_id`` of the paper's eval samples."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    repo, revision, filename, id_rule, problem_key = BENCHMARK_SOURCES[name]
    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    if filename.endswith(".parquet"):
        source = pd.read_parquet(path).to_dict("records")
    else:
        source = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for index, row in enumerate(source):
        row_id = index if id_rule == "index" else row["id"]
        row_id = int(row_id) if not isinstance(row_id, str) else row_id
        rows.append({"id": row_id, "problem": str(row[problem_key]), "answer": str(row["answer"])})
    return rows, {"source": repo, "revision": revision, "file": filename}


# ----------------------------------------------------------------------------- writing + manifest


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"files": {}}


def write_verified(
    output: Path, rows: list[dict], info: dict[str, Any], manifest: dict[str, Any], *, update: bool, overwrite: bool
) -> None:
    """Write ``rows`` to ``output`` and check (or record) its sha256 in ``manifest``."""
    expected = manifest["files"].get(output.name, {}).get("sha256")
    if output.exists() and not overwrite:
        if expected and sha256(output) == expected:
            print(f"{output}: up to date", flush=True)
            return
        raise FileExistsError(f"{output} exists and does not match the manifest (pass --overwrite)")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.writelines(jsonl_lines(rows))
    digest = sha256(temporary)
    if expected and digest != expected and not update:
        temporary.unlink()
        raise RuntimeError(f"{output.name}: sha256 {digest} does not match the manifest ({expected})")
    temporary.replace(output)
    manifest["files"][output.name] = {**info, "rows": len(rows), "sha256": digest}
    print(f"{output}: {len(rows)} rows, sha256 {digest}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="dualopsd prepare-data", description=__doc__.splitlines()[0])
    parser.add_argument("target", choices=("train", "heldout", "benchmarks", "all"))
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR"), help="default: $DATA_DIR")
    parser.add_argument("--bench-dir", default=os.environ.get("BENCH_DIR"), help="default: $BENCH_DIR")
    parser.add_argument("--manifest", type=Path, default=None, help="default: <repo>/data/manifest.json")
    parser.add_argument("--benchmarks", nargs="+", choices=sorted(BENCHMARK_SOURCES), default=sorted(BENCHMARK_SOURCES))
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--heldout-tail-size", type=int, default=20_000)
    parser.add_argument("--heldout-count", type=int, default=500)
    parser.add_argument("--heldout-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--update-manifest", action="store_true", help="record new hashes instead of verifying")
    args = parser.parse_args(argv)
    args.data_dir = Path(args.data_dir or repo_root() / "data" / "processed")
    args.bench_dir = Path(args.bench_dir or repo_root() / "data" / "benchmarks")
    args.manifest = args.manifest or repo_root() / "data" / "manifest.json"
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    common = dict(update=args.update_manifest, overwrite=args.overwrite)
    if args.target in ("train", "heldout", "all"):
        train_path = args.data_dir / TRAIN_FILE
        if args.target != "heldout" or not train_path.is_file():
            rows, info = openthoughts_rows(args.shuffle_seed)
            write_verified(train_path, rows, info, manifest, **common)
        if args.target in ("heldout", "all"):
            rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line]
            selected, indices = select_heldout(
                rows, tail_size=args.heldout_tail_size, count=args.heldout_count, seed=args.heldout_seed
            )
            info = {
                "source": TRAIN_FILE,
                "selection": "sorted random sample without replacement from the final tail_size rows",
                "tail_start_index": len(rows) - args.heldout_tail_size,
                "tail_size": args.heldout_tail_size,
                "seed": args.heldout_seed,
            }
            write_verified(args.data_dir / HELDOUT_FILE, selected, info, manifest, **common)
    if args.target in ("benchmarks", "all"):
        for name in args.benchmarks:
            rows, info = benchmark_rows(name)
            write_verified(args.bench_dir / f"{name}.jsonl", rows, info, manifest, **common)
    if args.update_manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
