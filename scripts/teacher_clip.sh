#!/bin/bash
# Teacher clipping ablation (tab:core-ablation, App. Teacher clipping ablations): the DualOPSD
# teacher objective KL(sg[q] || p_T) (selected beta) with a high / low per-vocabulary clip, for
# Qwen3-1.7B/4B/8B and seeds 42/43/44; each run is train -> merge -> eval. The unclipped arm is
# the main DualOPSD run (runs/main/<model>/dualopsd_seed<S>).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
EXP=teacher_clip
METHODS=(clip_high clip_low)

usage() { echo "Usage: bash scripts/teacher_clip.sh [options]   (methods: ${METHODS[*]})"; }

parse_args "$@"
for model in "${MODELS[@]}"; do
    want model "$model" || continue
    for method in "${METHODS[@]}"; do
        want method "$method" || continue
        for seed in "${SEEDS[@]}"; do
            want seed "$seed" || continue
            run="$(seed_run "$method" "$seed")"
            chain_begin "$model $run" || continue
            distill_train "$EXP" "$model" "$run" "$method" "$seed" 100 "" 1 ""
            merge_eval "$EXP" "$model" "$run" "$(run_dir "$EXP" "$model" "$run")/checkpoint-$((SMOKE ? 1 : 100))/student" "$JOB_ID"
        done
    done
done

analysis_job teacher_clip
finish
