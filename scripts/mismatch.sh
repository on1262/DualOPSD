#!/bin/bash
# OPSD mismatch diagnostics (fig:diagnostic, App. OPSD mismatch diagnostics) on the
# Qwen3-4B OPSD / DualOPSD w/o residual / DualOPSD runs of the main experiment, seeds 42/43/44:
#   posthoc     KL in both directions and clipped fraction re-scored on each checkpoint's
#               own training rollouts (steps 20/40/60/80/100)
#   recurrence  R_k between steps 20 and 100 on fixed responses: the OPSD seed-42 step-20
#               student's generations on the held-out 500 problems (generated first).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
EXP=mismatch
MODEL=qwen3-4b
METHODS=(opsd dualopsd_direct dualopsd)
STEPS=(20 40 60 80 100)

usage() { echo "Usage: bash scripts/mismatch.sh [--only posthoc|recurrence] [--method ...] [--seed ...] [--steps ...] [options]"; }

parse_args "$@"
[[ -z "$ONLY" || "$ONLY" == posthoc || "$ONLY" == recurrence ]] || die "--only posthoc|recurrence"
want model "$MODEL" || die "mismatch diagnostics are defined for $MODEL only"
main_runs="$SRC_ROOT/runs/main/$MODEL"
beta="$(selected_beta "$MODEL")"

# Fixed responses for the recurrence statistic.
canonical_job=""
canonical="$(gen_dir "$EXP" "$MODEL" "$(step_name opsd_seed42 20)")/student.jsonl"
if [[ -z "$ONLY" || "$ONLY" == recurrence ]]; then
    require_path "$main_runs/opsd_seed42/checkpoint-20/student"
    generate_job "$EXP" "$MODEL" "$(step_name opsd_seed42 20)" "$main_runs/opsd_seed42/checkpoint-20" student "$HELDOUT_FILE" ""
    canonical_job="$JOB_ID" canonical="$GEN_OUT"
fi

for method in "${METHODS[@]}"; do
    want method "$method" || continue
    for seed in "${SEEDS[@]}"; do
        want seed "$seed" || continue
        run="$(seed_run "$method" "$seed")"
        chain_begin "$MODEL $run" || continue
        require_path "$main_runs/$run"
        if [[ -z "$ONLY" || "$ONLY" == posthoc ]]; then
            for step in "${STEPS[@]}"; do
                want steps "$step" || continue
                out="$(diag_dir "$EXP" posthoc)/$MODEL"   # writes $out/<run>_step<N>/
                result="$out/$(step_name "$run" "$step")"
                if [[ -f "$result/summary.json" ]] || protected "$result"; then skip "posthoc $run step $step" "$result"; continue; fi
                submit_job --name "${EXP}_${MODEL}_$(step_name "$run" "$step")_posthoc" --time "$(job_time posthoc)" \
                    diagnostics posthoc --run-dir "$main_runs/$run" --method "$method" --step "$step" \
                    --base-model "$(model_path "$MODEL")" \
                    --residual-beta "$beta" --output-dir "$out"
                if (( SMOKE )); then break; fi
            done
        fi
        if [[ -z "$ONLY" || "$ONLY" == recurrence ]]; then
            out="$(diag_dir "$EXP" recurrence)/$MODEL"   # writes $out/<run>.json
            if [[ -f "$out/$run.json" ]] || protected "$out/$run.json"; then
                skip "recurrence $run" "$out/$run.json"
            else
                submit_job --name "${EXP}_${MODEL}_${run}_recurrence" --time "$(job_time recurrence)" --after "$canonical_job" \
                    diagnostics recurrence --run-dir "$main_runs/$run" --method "$method" --base-model "$(model_path "$MODEL")" \
                    --canonical-responses "$canonical" --dataset "$HELDOUT_FILE" --output-dir "$out"
            fi
        fi
    done
done

analysis_job mismatch
finish
