#!/bin/bash
# Residual weight selection and sensitivity (App. Residual weight selection):
#   validation   seed-42 DualOPSD for every beta of tab:beta-validation, step-100 student
#                generation on the held-out 500 problems, Qwen3-8B judge (fig:beta-validation)
#   sensitivity  Qwen3-8B seed-42 step-100 students on AIME24 (tab:beta); beta=0 is
#                DualOPSD w/o residual. beta=0.7 and 1.0 are also scored on AIME25/HMMT25
#                (tab:beta reports pooled truncation/length for 0.7).
# Shared points reuse the main-experiment run (and, for the sensitivity table, its evaluation)
# through links in runs/beta and eval/beta: the selected beta of each model, beta=0, and Qwen3-1.7B
# beta=1.0 (the compute-matched target trained by main.sh --method opsd_matched). Run main.sh
# first or pass --after with its job ids.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
EXP=beta
declare -A GRID=(
    [qwen3-1.7b]="0.1 0.2 0.5 0.7 0.8 0.9 1.0 1.1"
    [qwen3-4b]="0.1 0.2 0.5 0.7 1.0"
    [qwen3-8b]="0.1 0.2 0.5 0.7 1.0"
)
SENSITIVITY="0 0.1 0.2 0.5 0.7 1.0"
SEED=42

usage() {
    cat <<EOF
Usage: bash scripts/beta.sh [--only validation|sensitivity] [options]
Grid: 1.7B {${GRID[qwen3-1.7b]}}, 4B {${GRID[qwen3-4b]}}, 8B {${GRID[qwen3-8b]}}; seed $SEED.
Sensitivity (8B): beta in {$SENSITIVITY}.
EOF
}

parse_args "$@"
[[ -z "$ONLY" || "$ONLY" == validation || "$ONLY" == sensitivity ]] || die "--only validation|sensitivity"
want seed "$SEED" || die "the beta grid uses seed $SEED only"
declare -A TRAINED=()

# Main-experiment run that serves as MODEL BETA (empty if the point is trained here).
shared_main_run() {
    local model="$1" beta="$2"
    if [[ "$beta" == 0 ]]; then seed_run dualopsd_direct "$SEED"
    elif [[ "$beta" == "$(selected_beta "$model")" ]]; then seed_run dualopsd "$SEED"
    elif [[ "$model:$beta" == qwen3-1.7b:1.0 ]]; then seed_run dualopsd_beta1p0 "$SEED"
    fi
}

# Train (or link) the seed-42 run for MODEL BETA; sets RUN, RUN_PATH, MAIN_RUN, JOB_ID.
beta_run() {
    local model="$1" beta="$2" method=dualopsd extra=()
    if [[ "$beta" == 0 ]]; then
        method=dualopsd_direct RUN="$(seed_run dualopsd_direct "$SEED")"
    else
        RUN="$(seed_run "dualopsd_beta$(beta_tag "$beta")" "$SEED")"
        extra=(-- --set "$KEY_BETA=$beta")
    fi
    MAIN_RUN=""
    (( SMOKE )) || MAIN_RUN="$(shared_main_run "$model" "$beta")"
    if [[ -n "$MAIN_RUN" ]]; then
        RUN_PATH="$SRC_ROOT/runs/main/$model/$MAIN_RUN" JOB_ID=""
        require_path "$RUN_PATH/checkpoint-100/student/adapter_model.safetensors"
        link_shared "$(run_dir "$EXP" "$model" "$RUN")" "../../main/$model/$MAIN_RUN"
        return
    fi
    RUN_PATH="$(run_dir "$EXP" "$model" "$RUN")"
    if [[ -z "${TRAINED[$model:$beta]+x}" ]]; then
        distill_train "$EXP" "$model" "$RUN" "$method" "$SEED" 100 "" 1 "" "${extra[@]}"
        TRAINED[$model:$beta]="$JOB_ID"
    fi
    JOB_ID="${TRAINED[$model:$beta]}"
}

if [[ -z "$ONLY" || "$ONLY" == validation ]]; then
    for model in "${MODELS[@]}"; do
        want model "$model" || continue
        for beta in ${GRID[$model]}; do
            want beta "$beta" || continue
            chain_begin "validation $model beta=$beta" || continue
            beta_run "$model" "$beta"
            name="$(step_name "$RUN" 100)"
            generate_job "$EXP" "$model" "$name" "$RUN_PATH/checkpoint-$((SMOKE ? 1 : 100))" student \
                "$HELDOUT_FILE" "$JOB_ID" generate_long "$(max_num_seqs "$model")"
            judge_job "$EXP" "$model" "$name" "$JOB_ID" "$GEN_OUT"
        done
    done
fi

if [[ -z "$ONLY" || "$ONLY" == sensitivity ]] && want model qwen3-8b; then
    model=qwen3-8b
    for beta in $SENSITIVITY; do
        want beta "$beta" || continue
        chain_begin "sensitivity $model beta=$beta" || continue
        beta_run "$model" "$beta"
        benches=(aime24)
        [[ "$beta" == 0.7 || "$beta" == 1.0 ]] && benches=("${BENCHMARKS[@]}")
        if [[ -n "$MAIN_RUN" ]]; then
            # Shared with the main experiment: link its evaluation.
            require_path "$SRC_ROOT/eval/main/$model/$MAIN_RUN/aime24/samples.jsonl"
            link_shared "$(eval_dir "$EXP" "$model" "$RUN")" "../../main/$model/$MAIN_RUN"
            continue
        fi
        merge_eval "$EXP" "$model" "$RUN" "$RUN_PATH/checkpoint-$((SMOKE ? 1 : 100))/student" "$JOB_ID" "${benches[@]}"
    done
fi

analysis_job beta
finish
