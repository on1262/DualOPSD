#!/bin/bash
# Checkpoint accuracy curves (fig:acc-curves): merge and evaluate the Qwen3-4B OPSD and
# DualOPSD (beta=0.7) students of the main experiment at steps 20/40/60/80/100, seeds 42/43/44.
# Inputs: $DUALOPSD_SCRATCH/runs/main/qwen3-4b/{opsd,dualopsd}_seed<S> (scripts/main.sh).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
EXP=curves
MODEL=qwen3-4b
METHODS=(opsd dualopsd)
STEPS=(20 40 60 80 100)

usage() { echo "Usage: bash scripts/curves.sh [--method opsd,dualopsd] [--seed ...] [--steps 20,...,100] [options]"; }

parse_args "$@"
want model "$MODEL" || die "curves are defined for $MODEL only"
for method in "${METHODS[@]}"; do
    want method "$method" || continue
    for seed in "${SEEDS[@]}"; do
        want seed "$seed" || continue
        run="$(seed_run "$method" "$seed")"
        source_dir="$SRC_ROOT/runs/main/$MODEL/$run"
        for step in "${STEPS[@]}"; do
            want steps "$step" || continue
            chain_begin "$MODEL $run step $step" || continue
            adapter="$source_dir/checkpoint-$step/student"
            require_path "$adapter"
            merge_eval "$EXP" "$MODEL" "$(step_name "$run" "$step")" "$adapter" ""
        done
    done
done

analysis_job curves
finish
