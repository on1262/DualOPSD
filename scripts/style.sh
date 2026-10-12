#!/bin/bash
# Teacher-student style alignment (App. Teacher-student style alignment): step-100 teacher and
# student generations on the held-out 500 problems for OPSD and DualOPSD, seeds 42/43/44,
# then the formatting-feature analysis. Qwen3-4B reuses the scripts/teacher_diag.sh
# generations through a relative symlink (run that script first).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
EXP=style
METHODS=(opsd dualopsd)
ROLES=(student teacher)
STEP=100

usage() { echo "Usage: bash scripts/style.sh [--model ...] [--method ...] [--seed ...] [options]"; }

# Link generations/style/qwen3-4b/NAME to the teacher_diag generations of the same checkpoint.
link_teacher_diag() {
    local name="$1" link
    link="$(gen_dir "$EXP" qwen3-4b "$name")"
    [[ -e "$(gen_dir teacher_diag qwen3-4b "$name")" ]] || note "warning: $name not generated yet by scripts/teacher_diag.sh"
    link_shared "$link" "../../teacher_diag/qwen3-4b/$name"
}

parse_args "$@"
for model in "${MODELS[@]}"; do
    want model "$model" || continue
    for method in "${METHODS[@]}"; do
        want method "$method" || continue
        for seed in "${SEEDS[@]}"; do
            want seed "$seed" || continue
            run="$(seed_run "$method" "$seed")"
            chain_begin "$model $run" || continue
            if [[ "$model" == qwen3-4b ]]; then link_teacher_diag "$(step_name "$run" "$STEP")"; continue; fi
            ckpt="$SRC_ROOT/runs/main/$model/$run/checkpoint-$STEP"
            require_path "$ckpt"
            for role in "${ROLES[@]}"; do
                generate_job "$EXP" "$model" "$(step_name "$run" "$STEP")" "$ckpt" "$role" "$HELDOUT_FILE" ""
            done
        done
    done
done

analysis_job style
finish
