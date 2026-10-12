#!/bin/bash
# Teacher adaptation diagnostics (fig:ablation-dynamics, App. Teacher adaptation diagnostics):
# teacher (with reference) and student generations on the held-out 500 problems for the
# Qwen3-4B DualOPSD w/o residual / DualOPSD runs of the main experiment at steps
# 20/40/60/80/100, seeds 42/43/44, each judged by Qwen3-8B. OPSD is generated at step 100
# only, unjudged, for the style-alignment analysis (scripts/style.sh). JS retention,
# target-fit KL and clipped mass come from the posthoc re-scoring of scripts/mismatch.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
EXP=teacher_diag
MODEL=qwen3-4b
METHODS=(opsd dualopsd_direct dualopsd)
STEPS=(20 40 60 80 100)
ROLES=(student teacher)

usage() { echo "Usage: bash scripts/teacher_diag.sh [--method ...] [--seed ...] [--steps ...] [options]"; }

parse_args "$@"
want model "$MODEL" || die "teacher diagnostics are defined for $MODEL only"
for method in "${METHODS[@]}"; do
    want method "$method" || continue
    for seed in "${SEEDS[@]}"; do
        want seed "$seed" || continue
        run="$(seed_run "$method" "$seed")"
        for step in "${STEPS[@]}"; do
            want steps "$step" || continue
            [[ "$method" == opsd && "$step" != 100 ]] && continue
            chain_begin "$MODEL $run step $step" || continue
            ckpt="$SRC_ROOT/runs/main/$MODEL/$run/checkpoint-$step"
            require_path "$ckpt"
            name="$(step_name "$run" "$step")"
            gens=() files=()
            for role in "${ROLES[@]}"; do
                generate_job "$EXP" "$MODEL" "$name" "$ckpt" "$role" "$HELDOUT_FILE" ""
                gens+=("$JOB_ID") files+=("$GEN_OUT")
            done
            [[ "$method" == opsd ]] || judge_job "$EXP" "$MODEL" "$name" "${gens[*]}" "${files[@]}"
        done
    done
done

analysis_job teacher_diag
finish
