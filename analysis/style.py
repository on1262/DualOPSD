"""Teacher--student formatting distance at step 100 (tab:style-alignment, App. style alignment).

Inputs: generations/style/<model>/{opsd,dualopsd}_seed<S>_step100/{student,teacher}.jsonl on the
500-problem diagnostic set.  Per scale and seed, only problems whose four responses (OPSD and
DualOPSD, teacher and student) are all untruncated are kept.  Nine formatting counts per 100
completion tokens are standardized by one pooled sample SD over every kept response of all
scales and seeds; the distance is the mean absolute standardized teacher - student difference.

Usage: python -m analysis.style [--print-tables]
"""

from __future__ import annotations

import argparse
import itertools
import re
from statistics import mean, stdev
from typing import Any

import numpy as np
from scipy import stats

from analysis.common import (
    MODELS,
    SEEDS,
    add_common_args,
    generation_file,
    output_dir,
    print_tables,
    read_jsonl,
    run_name,
    write_json,
)

EXP = "style"
STEP = 100
METHODS = ("opsd", "dualopsd")
POLICIES = ("student", "teacher")
FEATURES = (
    "newline_rate",
    "paragraph_rate",
    "heading_rate",
    "horizontal_rule_rate",
    "bullet_rate",
    "numbered_list_rate",
    "bold_rate",
    "inline_math_rate",
    "display_math_rate",
)
MODEL_NAMES = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B"}
EXAMPLE_MAX_TOKENS = 2400


def style_features(row: dict[str, Any]) -> np.ndarray:
    """Formatting counts per 100 completion tokens."""
    text = str(row["response"])
    per_100 = 100.0 / max(int(row.get("completion_tokens", 0)), 1)
    lines = text.splitlines()
    counts = (
        text.count("\n"),
        len(re.findall(r"\n\s*\n", text)),
        sum(bool(re.match(r"^\s{0,3}#{1,6}\s+", line)) for line in lines),
        sum(bool(re.match(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", line)) for line in lines),
        sum(bool(re.match(r"^\s*[-*+]\s+", line)) for line in lines),
        sum(bool(re.match(r"^\s*\d+[.)]\s+", line)) for line in lines),
        len(re.findall(r"\*\*[^*\n]+\*\*", text)),
        text.replace("$$", "").count("$") // 2 + text.count(r"\("),
        text.count("$$") // 2 + text.count(r"\["),
    )
    return np.asarray([count * per_100 for count in counts], dtype=np.float64)


def load(model: str, seed: int) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    rows = {}
    for method, policy in itertools.product(METHODS, POLICIES):
        path = generation_file(EXP, model, run_name(method, seed), STEP, policy)
        by_id = {str(row["id"]): row for row in read_jsonl(path)}
        if len(by_id) != 500:
            raise ValueError(f"expected 500 unique problems: {path}")
        rows[(method, policy)] = by_id
    if any(r.keys() != rows[("opsd", "student")].keys() for r in rows.values()):
        raise ValueError(f"problem IDs differ across generations: {model} seed {seed}")
    return rows


def exact_sign_flip_p(deltas: list[float]) -> float:
    values = np.asarray(deltas)
    observed = abs(values.mean())
    flips = [abs(np.mean(values * np.asarray(s))) for s in itertools.product((-1.0, 1.0), repeat=len(values))]
    return sum(v >= observed - 1e-15 for v in flips) / len(flips)


def build() -> dict[str, Any]:
    rows = {(model, seed): load(model, seed) for model in MODELS for seed in SEEDS}
    kept = {
        key: [i for i in sorted(data[("opsd", "student")])
              if not any(bool(data[mp][i]["truncated"]) for mp in data)]
        for key, data in rows.items()
    }
    features = {
        (key, mp, i): style_features(data[mp][i])
        for key, data in rows.items() for i in kept[key] for mp in data
    }
    scale = np.stack(list(features.values())).std(axis=0, ddof=1)
    if np.any(scale <= 1e-12):
        raise ValueError("constant formatting feature in the comparison set")
    distance = {
        (key, method, i): float(np.mean(np.abs((features[(key, (method, "teacher"), i)]
                                                 - features[(key, (method, "student"), i)]) / scale)))
        for key in rows for i in kept[key] for method in METHODS
    }
    scales: dict[str, Any] = {}
    examples: dict[str, Any] = {}
    for model in MODELS:
        seed_means = {m: [mean(distance[((model, s), m, i)] for i in kept[(model, s)]) for s in SEEDS] for m in METHODS}
        delta = list(np.asarray(seed_means["dualopsd"]) - np.asarray(seed_means["opsd"]))
        test = stats.ttest_rel(seed_means["dualopsd"], seed_means["opsd"])
        scales[model] = {
            "opsd": {"mean": mean(seed_means["opsd"]), "sample_sd": stdev(seed_means["opsd"]), "by_seed": seed_means["opsd"]},
            "dualopsd": {"mean": mean(seed_means["dualopsd"]), "sample_sd": stdev(seed_means["dualopsd"]),
                         "by_seed": seed_means["dualopsd"]},
            "delta_by_seed": [float(x) for x in delta],
            "relative_reduction": float(-np.mean(delta) / np.mean(seed_means["opsd"])),
            "paired_t_p_two_sided": float(test.pvalue),
            "exact_sign_flip_p_two_sided": exact_sign_flip_p(delta),
            "kept_problems_by_seed": {str(s): len(kept[(model, s)]) for s in SEEDS},
        }
        examples[model] = qualitative_candidates(model, rows[(model, 42)], kept[(model, 42)], distance)
    return {
        "metric": "mean |standardized teacher - student difference| over nine formatting features",
        "features": list(FEATURES),
        "feature_sample_sd": dict(zip(FEATURES, map(float, scale))),
        "scales": scales,
        "qualitative_candidates": examples,
    }


def qualitative_candidates(model: str, data: dict, kept: list[str], distance: dict, limit: int = 10) -> list[dict[str, Any]]:
    """Seed-42 problems with a positive reduction and short responses, nearest the median reduction."""
    key = (model, 42)
    positive = []
    for i in kept:
        improvement = distance[(key, "opsd", i)] - distance[(key, "dualopsd", i)]
        total = sum(int(data[mp][i]["completion_tokens"]) for mp in data)
        if improvement > 0 and total <= EXAMPLE_MAX_TOKENS:
            positive.append((improvement, total, i))
    median = float(np.median([p[0] for p in positive]))
    ordered = sorted(positive, key=lambda p: (abs(p[0] - median), p[1], p[2]))[:limit]
    return [
        {"id": i, "opsd_distance": distance[(key, "opsd", i)], "dualopsd_distance": distance[(key, "dualopsd", i)],
         "total_tokens": total, "problem": data[("opsd", "student")][i]["problem"],
         "responses": {f"{m}_{p}": data[(m, p)][i]["response"] for m, p in data}}
        for _, total, i in ordered
    ]


def tables(result: dict[str, Any]) -> dict[str, list[list[str]]]:
    rows = []
    for model in MODELS:
        row = result["scales"][model]
        rows.append([
            MODEL_NAMES[model],
            f"{row['opsd']['mean']:.3f}±{row['opsd']['sample_sd']:.3f}",
            f"{row['dualopsd']['mean']:.3f}±{row['dualopsd']['sample_sd']:.3f}",
            f"{100 * row['relative_reduction']:.1f}",
            " / ".join(str(row["kept_problems_by_seed"][str(s)]) for s in SEEDS),
        ])
    return {"tab:style-alignment": rows}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, EXP)
    args = parser.parse_args(argv)
    out = output_dir(args, EXP)
    result = build()
    print(write_json(out / "style.json", result, overwrite=args.overwrite))
    if args.print_tables:
        print_tables(tables(result))
    return result


if __name__ == "__main__":
    main()
