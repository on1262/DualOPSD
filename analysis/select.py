"""Locked selections made before benchmark evaluation.

``beta``:    pick the residual weight per model on the 500-problem held-out set (most correct,
             then lower truncation, fewer observed tokens, smaller beta).
``matched``: pick, per model and seed, the OPSD checkpoint on the continued OPSD trajectory whose
             cumulative training PFLOPs are closest to the step-100 DualOPSD run (for Qwen3-1.7B the
             paper locked the steps against the beta=1.0 run, ``dualopsd_beta1p0_seed<S>``).

Usage:
    python -m analysis.select beta
    python -m analysis.select matched [--model qwen3-4b] [--seed 42] [--target-run MODEL=dualopsd_beta1p0_seed{seed}]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from analysis import beta as beta_analysis
from analysis.common import MODELS, SEEDS, add_common_args, output_dir, run_dir, run_name, write_json
from analysis.compute import checkpoint_steps, pflops, run_vectors, token_counts

EXP = "main"
# DualOPSD run each model's matched OPSD step was locked against (default: run_name("dualopsd", seed)).
MATCHED_TARGET = {"qwen3-1.7b": "dualopsd_beta1p0_seed{seed}"}


def select_beta(models=MODELS, heldout: Path | None = None) -> dict[str, Any]:
    held_out = beta_analysis.validation(models, heldout)
    result = {}
    for model in models:
        ranking = beta_analysis.rank_betas(held_out[model])
        result[model] = {"selected_beta": held_out[model][ranking[0]]["beta"], "ranking": ranking,
                         "candidates": held_out[model]}
    return {"rule": ["most held-out correct", "lower truncation", "fewer observed tokens", "smaller beta"],
            "models": result}


def select_matched(model: str, seed: int, target_run: str | None = None, max_relative_error: float = 0.02,
                   train_file=None) -> dict[str, Any]:
    target_root = run_dir(EXP, model, target_run or run_name("dualopsd", seed))
    target_vectors, target_batch = run_vectors(target_root, train_file)
    target_counts = token_counts(target_vectors, 100 * target_batch)
    target = pflops("dualopsd", target_counts, model)
    opsd_root = run_dir(EXP, model, run_name("opsd_matched", seed))
    vectors, batch = run_vectors(opsd_root, train_file)
    candidates = []
    for step in checkpoint_steps(opsd_root):
        value = pflops("opsd", token_counts(vectors, step * batch), model)
        candidates.append({"step": step, "pflops": value, "relative_target_error": abs(value - target) / target})
    selected = min(candidates, key=lambda row: abs(row["pflops"] - target))
    crossed = min(r["pflops"] for r in candidates) <= target <= max(r["pflops"] for r in candidates)
    locked = crossed and selected["relative_target_error"] <= max_relative_error
    return {
        "status": "locked" if locked else "needs_more_training",
        "target": {"run": str(target_root), "step": 100, "pflops": target, "tokens": target_counts},
        "opsd_run": str(opsd_root),
        "selected": selected,
        "candidates": candidates,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("beta", "matched"))
    add_common_args(parser, EXP, output_help=f"default: $DUALOPSD_SCRATCH/results/{EXP} (matched) "
                                             "or $DUALOPSD_SCRATCH/results/beta (beta)")
    parser.add_argument("--model", choices=MODELS, nargs="+", default=list(MODELS))
    parser.add_argument("--seed", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--target-run", action="append", default=[], metavar="MODEL=TEMPLATE",
                        help="override the DualOPSD target run, e.g. qwen3-1.7b=dualopsd_beta1p0_seed{seed}")
    parser.add_argument("--max-relative-error", type=float, default=0.02)
    parser.add_argument("--heldout-file", type=Path, default=None, help="beta: held-out set (default: $HELDOUT_FILE)")
    args = parser.parse_args(argv)
    if args.what == "beta":
        out = output_dir(args, "beta")
        result = select_beta(args.model, args.heldout_file)
        print(write_json(out / "selected_beta.json", result, overwrite=args.overwrite))
        for model, row in result["models"].items():
            print(model, "beta =", row["selected_beta"])
        return result
    out = output_dir(args, EXP)
    targets = {**MATCHED_TARGET, **dict(item.split("=", 1) for item in args.target_run)}
    result = {"rule": "minimum |PFLOPs - target PFLOPs| over saved OPSD checkpoints", "models": {}}
    for model in args.model:
        result["models"][model] = {
            str(seed): select_matched(model, seed, targets[model].format(seed=seed) if model in targets else None,
                                      args.max_relative_error)
            for seed in args.seed
        }
        for seed, row in result["models"][model].items():
            print(model, seed, row["status"], "step", row["selected"]["step"], f"{row['selected']['pflops']:.2f}",
                  "target", f"{row['target']['pflops']:.2f}")
    print(write_json(out / "matched_selection.json", result, overwrite=args.overwrite))
    return result


if __name__ == "__main__":
    main()
