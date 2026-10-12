#!/bin/bash
# Main results (tab:main, per-seed tables, compute/length analysis):
# Base, SFT, GRPO, OPSD, compute-matched OPSD, DualOPSD w/o residual and DualOPSD
# for Qwen3-1.7B/4B/8B and seeds 42/43/44. Every (model, method, seed) is an
# independent train -> merge -> eval chain; evaluations run one job per benchmark.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
EXP=main
METHODS=(base sft grpo opsd opsd_matched dualopsd_direct dualopsd)
MATCHED_MAX_STEPS=320
# The Qwen3-1.7B matched steps were locked against DualOPSD beta=1.0 (trained here, not evaluated).
declare -A MATCHED_TARGET_BETA=([qwen3-1.7b]=1.0)

usage() {
    cat <<EOF
Usage: bash scripts/main.sh [options]
Methods: ${METHODS[*]}
  base          evaluation only, generation seeds 42/43/44
  opsd_matched  continues the OPSD step-100 state to step $MATCHED_MAX_STEPS and evaluates the
                compute-matched checkpoint: --steps, else the locked selection of
                'python -m analysis.select matched'. Without a selection only training is submitted.
                Qwen3-1.7B also trains the beta=${MATCHED_TARGET_BETA[qwen3-1.7b]} DualOPSD selection target.
  dualopsd      residual weight per model (0.8/0.7/0.5); --beta overrides it
EOF
}

parse_args "$@"
if (( SMOKE )) && [[ -z "$FILTER_METHODS" ]]; then FILTER_METHODS=dualopsd; fi
declare -A OPSD_JOB=()

# Locked compute-matched OPSD step for MODEL SEED (empty if not selected yet): --steps, else
# `python -m analysis.select matched` (results/main/matched_selection.json).
matched_steps() {
    local model="$1" seed="$2"
    if [[ -n "$FILTER_STEPS" ]]; then echo "${FILTER_STEPS//,/ }"; return; fi
    if (( SMOKE )); then echo 2; return; fi
    python3 - "$model" "$seed" "$DUALOPSD_SCRATCH" <<'PY'
import json, sys
from pathlib import Path
model, seed, root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
selection = root / "results/main/matched_selection.json"
if selection.is_file():
    row = json.loads(selection.read_text()).get("models", {}).get(model, {}).get(seed)
    if row and row.get("status") == "locked":
        print(row["selected"]["step"])
PY
}

# Submit OPSD training of MODEL SEED (once per invocation); sets JOB_ID.
opsd_train() {
    local model="$1" seed="$2" key="$1:$2"
    if [[ -z "${OPSD_JOB[$key]+x}" ]]; then
        distill_train "$EXP" "$model" "$(seed_run opsd "$seed")" opsd "$seed" 100 "" 1 ""
        OPSD_JOB[$key]="$JOB_ID"
    fi
    JOB_ID="${OPSD_JOB[$key]}"
}

for model in "${MODELS[@]}"; do
    want model "$model" || continue
    for method in "${METHODS[@]}"; do
        want method "$method" || continue
        for seed in "${SEEDS[@]}"; do
            want seed "$seed" || continue
            run="$(seed_run "$method" "$seed")"
            case "$method" in
                base)
                    chain_begin "$model base genseed$seed" || continue
                    eval_jobs "$EXP" "$model" "$(base_name "$seed")" "$(model_path "$model")" "$seed" ""
                    ;;
                sft|grpo)
                    chain_begin "$model $run" || continue
                    baseline_train "$EXP" "$model" "$run" "$method" "$seed" ""
                    merge_eval "$EXP" "$model" "$run" "$(run_dir "$EXP" "$model" "$run")" "$JOB_ID"
                    ;;
                opsd)
                    chain_begin "$model $run" || continue
                    opsd_train "$model" "$seed"
                    merge_eval "$EXP" "$model" "$run" "$(run_dir "$EXP" "$model" "$run")/checkpoint-$((SMOKE ? 1 : 100))/student" "$JOB_ID"
                    ;;
                opsd_matched)
                    chain_begin "$model $run" || continue
                    source_dir="$(run_dir "$EXP" "$model" "$(seed_run opsd "$seed")")"
                    distill_done "$source_dir" "$((SMOKE ? 1 : 100))" && JOB_ID="" || opsd_train "$model" "$seed"
                    distill_train "$EXP" "$model" "$run" opsd_matched "$seed" "$MATCHED_MAX_STEPS" "$JOB_ID" 2 "$source_dir"
                    train_job="$JOB_ID"
                    if [[ -n "${MATCHED_TARGET_BETA[$model]:-}" ]] && (( ! SMOKE )); then
                        target_beta="${MATCHED_TARGET_BETA[$model]}"
                        distill_train "$EXP" "$model" "$(seed_run "dualopsd_beta$(beta_tag "$target_beta")" "$seed")" \
                            dualopsd "$seed" 100 "" 1 "" -- --set "$KEY_BETA=$target_beta"
                    fi
                    steps="$(matched_steps "$model" "$seed")"
                    if [[ -z "$steps" ]]; then
                        note "no locked matched step for $model seed $seed: run 'python -m analysis.select matched' after training, then rerun with --method opsd_matched"
                        continue
                    fi
                    for step in $steps; do
                        merge_eval "$EXP" "$model" "$(step_name "$run" "$step")" \
                            "$(run_dir "$EXP" "$model" "$run")/checkpoint-$step/student" "$train_job"
                    done
                    ;;
                dualopsd_direct|dualopsd)
                    extra=()
                    if [[ "$method" == dualopsd ]]; then
                        beta="${FILTER_BETAS:-$(selected_beta "$model")}"
                        [[ "$beta" == *,* ]] && die "main.sh takes a single --beta"
                        [[ -n "$FILTER_BETAS" ]] && run="$(seed_run "dualopsd_beta$(beta_tag "$beta")" "$seed")"
                        extra=(--set "$KEY_BETA=$beta")
                    fi
                    chain_begin "$model $run" || continue
                    distill_train "$EXP" "$model" "$run" "$method" "$seed" 100 "" 1 "" -- "${extra[@]}"
                    merge_eval "$EXP" "$model" "$run" "$(run_dir "$EXP" "$model" "$run")/checkpoint-$((SMOKE ? 1 : 100))/student" "$JOB_ID"
                    ;;
            esac
        done
    done
done

analysis_job main_results compute
finish
