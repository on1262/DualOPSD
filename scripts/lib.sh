#!/bin/bash
# Shared helpers for the experiment launchers (scripts/*.sh) and Slurm jobs (slurm/*.sbatch).
# Sourcing this file loads configs/user.env.

DUALOPSD_REPO="${DUALOPSD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DUALOPSD_REPO
# shellcheck source=../configs/user.env
source "$DUALOPSD_REPO/configs/user.env"

MODELS=(qwen3-1.7b qwen3-4b qwen3-8b)
SEEDS=(42 43 44)
BENCHMARKS=(aime24 aime25 hmmt25)

# Config keys the launchers override with `--set`.
KEY_MODEL=model.name
KEY_SEED=seed
KEY_TRAIN_FILE=data.train_file
KEY_BETA=teacher.residual_beta

die() { echo "error: $*" >&2; exit 1; }
note() { echo "# $*"; }

# ---------------------------------------------------------------- models

model_path() {
    case "$1" in
        qwen3-1.7b) echo "$MODEL_QWEN3_1P7B" ;;
        qwen3-4b) echo "$MODEL_QWEN3_4B" ;;
        qwen3-8b) echo "$MODEL_QWEN3_8B" ;;
        *) die "unknown model: $1" ;;
    esac
}

# Residual weight selected on the held-out set (tab:beta-validation).
selected_beta() {
    case "$1" in
        qwen3-1.7b) echo 0.8 ;;
        qwen3-4b) echo 0.7 ;;
        qwen3-8b) echo 0.5 ;;
        *) die "unknown model: $1" ;;
    esac
}

# 0.7 -> 0p7, 1 -> 1p0
beta_tag() {
    local b="$1"
    [[ "$b" == *.* ]] || b="$b.0"
    echo "${b/./p}"
}

# Distillation method -> configs/distill/<file>.yaml
method_config() {
    case "$1" in
        opsd|opsd_matched|dualopsd_direct|dualopsd|clip_high|clip_low) echo "$1" ;;
        *) die "not a distillation method: $1" ;;
    esac
}

# Extra `--set` overrides that reproduce the paper runs exactly: paper_overrides MODEL METHOD [BETA].
# They change only vLLM batching or gradient-accumulation order, not the objective.
paper_overrides() {
    local model="$1" method="$2" beta="${3:-}"
    case "$model:$method" in
        qwen3-1.7b:dualopsd|qwen3-1.7b:clip_*)
            case "${beta:-$(selected_beta "$model")}" in 0.8|0.9|1.1) echo "rollout.max_num_seqs=32" ;; esac ;;
        qwen3-1.7b:dualopsd_direct) echo "student.micro_batch_size=1 teacher.micro_batch_size=1" ;;
        qwen3-8b:opsd_matched) echo "student.micro_batch_size=1" ;;
    esac
}

# Wall time per (job kind, model). GRPO 1.7B/4B runs as a single 12 h job;
# everything else fits in <= 6 h.
job_time() {
    local kind="$1" model="${2:-}"
    if (( SMOKE )); then echo 00:30:00; return; fi
    case "$kind:$model" in
        distill:qwen3-1.7b) echo 02:00:00 ;;
        distill:qwen3-4b) echo 03:00:00 ;;
        distill:qwen3-8b) echo 04:00:00 ;;
        matched:qwen3-1.7b) echo 03:00:00 ;;
        matched:qwen3-4b) echo 04:00:00 ;;
        matched:qwen3-8b) echo 05:00:00 ;;
        sft:*) echo 04:00:00 ;;
        grpo:qwen3-8b) echo 04:00:00 ;;
        grpo:*) echo 12:00:00 ;;
        merge:*|posthoc:*|generate:*|analysis:*) echo 01:00:00 ;;
        generate_long:*) echo 02:00:00 ;;
        eval:*) echo 04:00:00 ;;
        judge:*|recurrence:*) echo 00:30:00 ;;
        *) die "no time for job kind $kind" ;;
    esac
}

# vLLM concurrency used for evaluation and held-out generation.
max_num_seqs() { [[ "$1" == qwen3-1.7b ]] && echo 128 || echo 64; }

# TRL arguments of the SFT / GRPO baselines (LoRA r64/alpha128, lr 5e-6, checkpoint every 5 steps).
BASELINE_ARGS=(--use_peft true --lora_r 64 --lora_alpha 128
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    --learning_rate 5e-6 --gradient_checkpointing true --save_strategy steps --save_steps 5
    --report_to none --data_seed 42 --bf16 true --fp16 false)
SFT_ARGS=(--per_device_train_batch_size 2 --gradient_accumulation_steps 32 --num_train_epochs 4
    --max_steps 100 --max_length 16000)
GRPO_ARGS=(--per_device_train_batch_size 1 --gradient_accumulation_steps 32 --num_train_epochs 2
    --max_steps 200 --num_generations 8 --num_iterations 2 --max_prompt_length 2048
    --max_completion_length 8192 --temperature 1.0 --beta 0.0 --loss_type grpo --scale_rewards group
    --use_vllm true --vllm_mode colocate)

# GRPO steps per job: 8B runs as 25-step segments resumed with --stop_after_step.
grpo_segment() { [[ "$1" == qwen3-8b ]] && echo 25 || echo 200; }

# ---------------------------------------------------------------- CLI

usage_common() {
    cat <<EOF
Common options:
  --model LIST     comma-separated models (qwen3-1.7b,qwen3-4b,qwen3-8b)
  --method LIST    comma-separated methods
  --seed LIST      comma-separated training seeds (42,43,44)
  --beta LIST      comma-separated residual weights
  --steps LIST     comma-separated checkpoint steps
  --only PART      run only one part of the experiment (script specific)
  --after IDS      make every submitted job depend on these job ids (afterok)
  --no-analysis    do not submit the final CPU analysis job
  --smoke          one chain, 1 step / tiny inputs, 30 min jobs, outputs under \$DUALOPSD_SCRATCH/smoke
  --dry-run        print sbatch commands without submitting
EOF
}

parse_args() {
    FILTER_MODELS="" FILTER_METHODS="" FILTER_SEEDS="" FILTER_BETAS="" FILTER_STEPS=""
    ONLY="" EXTERNAL_AFTER="" ANALYSIS=1 SMOKE=0 DRY_RUN=0
    while (( $# )); do
        case "$1" in
            --model) FILTER_MODELS="${2:?}"; shift 2 ;;
            --method) FILTER_METHODS="${2:?}"; shift 2 ;;
            --seed) FILTER_SEEDS="${2:?}"; shift 2 ;;
            --beta) FILTER_BETAS="${2:?}"; shift 2 ;;
            --steps) FILTER_STEPS="${2:?}"; shift 2 ;;
            --only) ONLY="${2:?}"; shift 2 ;;
            --after) EXTERNAL_AFTER="${2:?}"; shift 2 ;;
            --no-analysis) ANALYSIS=0; shift ;;
            --smoke) SMOKE=1; shift ;;
            --dry-run) DRY_RUN=1; shift ;;
            -h|--help) usage; usage_common; exit 0 ;;
            *) die "unknown option: $1 (see --help)" ;;
        esac
    done
    export SMOKE
    if (( SMOKE )); then ROOT="$DUALOPSD_SCRATCH/smoke"; else ROOT="$DUALOPSD_SCRATCH"; fi
    # Inputs produced by other experiments are always read from the full-scale tree.
    SRC_ROOT="$DUALOPSD_SCRATCH"
    LOG_DIR="$ROOT/logs/slurm"
    export DUALOPSD_LOG_DIR="$ROOT/logs"
    JOB_COUNT=0 CHAIN_COUNT=0 ALL_JOBS=()
    if (( SMOKE )); then note "smoke mode: outputs under $ROOT"; fi
}

# want KIND VALUE: true if VALUE passes the --model/--method/--seed/--beta/--steps filter.
want() {
    local list
    case "$1" in
        model) list="$FILTER_MODELS" ;;
        method) list="$FILTER_METHODS" ;;
        seed) list="$FILTER_SEEDS" ;;
        beta) list="$FILTER_BETAS" ;;
        steps) list="$FILTER_STEPS" ;;
    esac
    [[ -z "$list" || ",$list," == *",$2,"* ]]
}

# Start a new independent chain; in smoke mode only the first chain is submitted.
chain_begin() {
    if (( SMOKE && CHAIN_COUNT >= 1 )); then return 1; fi
    CHAIN_COUNT=$((CHAIN_COUNT + 1))
    note "chain $CHAIN_COUNT: $*"
}

# ---------------------------------------------------------------- paths

run_dir() { echo "$ROOT/runs/$1/$2/$3"; }
merged_dir() { echo "$ROOT/models/$1/$2/$3"; }
eval_dir() { echo "$ROOT/eval/$1/$2/$3"; }
gen_dir() { echo "$ROOT/generations/$1/$2/$3"; }
diag_dir() { echo "$ROOT/diagnostics/$1/$2"; }

seed_run() { echo "$1_seed$2"; }               # opsd 42 -> opsd_seed42
step_name() { echo "$1_step$2"; }              # opsd_seed42 20 -> opsd_seed42_step20
base_name() { [[ "$1" == 42 ]] && echo base || echo "base_genseed$1"; }

adapter_done() { [[ -f "$1/adapter_model.safetensors" ]]; }
distill_done() { adapter_done "$1/checkpoint-$2/student"; }
latest_checkpoint() {
    local d best=0 n
    for d in "$1"/checkpoint-*; do
        [[ -d "$d" ]] || continue
        n="${d##*-}"
        [[ "$n" =~ ^[0-9]+$ ]] && (( n > best )) && best="$n"
    done
    echo "$best"
}

# Fail (or warn in --dry-run / with --after) when an input from another experiment is missing.
require_path() {
    [[ -e "$1" ]] && return 0
    if (( DRY_RUN )) || [[ -n "$EXTERNAL_AFTER" ]]; then
        note "warning: missing input $1"
    else
        die "missing input $1 (run the producing experiment first, or pass --after JOBID)"
    fi
}

skip() { note "skip $1 (exists: $2)"; JOB_ID=""; }

# link_shared LINK TARGET: relative symlink LINK -> TARGET (e.g. ../../main/<model>/<run>) for an
# output shared with another experiment.
link_shared() {
    local link="$1" target="$2"
    if [[ -e "$link" || -L "$link" ]]; then note "exists: $link"; return; fi
    echo "+ ln -s $target $link"
    (( DRY_RUN )) || { mkdir -p "$(dirname "$link")"; ln -s "$target" "$link"; }
}

# True if PATH or a parent below $DUALOPSD_SCRATCH is a symlink (outputs shared
# between experiments are linked, and are never written to through the link).
protected() {
    local p="$1"
    while [[ "$p" == "$DUALOPSD_SCRATCH"/* ]]; do
        [[ -L "$p" ]] && return 0
        p="$(dirname "$p")"
    done
    return 1
}

# ---------------------------------------------------------------- submission

# "1 2,3:4" -> "1:2:3:4"
join_ids() { echo $1 | tr ', ' '::' | tr -s ':' | sed 's/^://; s/:$//'; }

# submit_job [--name N] [--time T] [--after IDS] [--afterany IDS] [--env K=V]... TEMPLATE [ARGS...]
# Sets JOB_ID. IDS are space- or colon-separated; empty values are ignored.
submit_job() {
    local name="" time="" after="" afterany="" envs=()
    while [[ "$1" == --* ]]; do
        case "$1" in
            --name) name="$2" ;;
            --time) time="$2" ;;
            --after) after="$after $2" ;;
            --afterany) afterany="$afterany $2" ;;
            --env) envs+=("$2") ;;
            *) die "submit_job: unknown option $1" ;;
        esac
        shift 2
    done
    local template="$1"; shift
    after="$after $EXTERNAL_AFTER"
    local deps=() ids
    ids="$(join_ids "$after")"; [[ -n "$ids" ]] && deps+=("afterok:$ids")
    ids="$(join_ids "$afterany")"; [[ -n "$ids" ]] && deps+=("afterany:$ids")
    local cmd=(sbatch --parsable)
    [[ -n "${SLURM_ACCOUNT:-}" ]] && cmd+=(--account="$SLURM_ACCOUNT")
    [[ -n "${SLURM_PARTITION:-}" ]] && cmd+=(--partition="$SLURM_PARTITION")
    cmd+=(--job-name="$name" --time="$time"
        --output="$LOG_DIR/%x_%j.out" --error="$LOG_DIR/%x_%j.err"
        --export=ALL)
    case "$template" in
        merge|analysis) ;;
        *) [[ -n "${SLURM_GPU_GRES:-}" ]] && cmd+=(--gres="$SLURM_GPU_GRES")
           [[ -n "${SLURM_GPU_CONSTRAINT:-}" ]] && cmd+=(--constraint="$SLURM_GPU_CONSTRAINT") ;;
    esac
    if (( ${#deps[@]} )); then cmd+=(--dependency="$(IFS=,; echo "${deps[*]}")"); fi
    cmd+=("$DUALOPSD_REPO/slurm/$template.sbatch" "$@")
    JOB_COUNT=$((JOB_COUNT + 1))
    if (( DRY_RUN )); then
        JOB_ID="dry$JOB_COUNT"
        if (( ${#envs[@]} )); then
            echo "[$JOB_ID] $(printf '%q ' env "${envs[@]}" "${cmd[@]}")"
        else
            echo "[$JOB_ID] $(printf '%q ' "${cmd[@]}")"
        fi
    else
        mkdir -p "$LOG_DIR"
        JOB_ID="$(env "${envs[@]}" "${cmd[@]}")" || die "sbatch failed for $name"
        JOB_ID="${JOB_ID%%;*}"
        echo "[$JOB_ID] $name"
        printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$JOB_ID" "$name" "${deps[*]:-}" >> "$ROOT/logs/jobs.tsv"
    fi
    ALL_JOBS+=("$JOB_ID")
}

# ---------------------------------------------------------------- job builders
# Each builder sets JOB_ID (empty when the output already exists and is skipped).

# distill_train EXP MODEL RUN METHOD SEED STEPS AFTER [SEGMENTS] [RESUME_RUN] [-- extra train args]
# STEPS is the final step of the method config (used to detect a finished run); RESUME_RUN
# starts the run from that run's step-100 state (compute-matched OPSD).
distill_train() {
    local exp="$1" model="$2" run="$3" method="$4" seed="$5" steps="$6" after="$7"
    local segments="${8:-1}" resume="${9:-}"
    shift 9 2>/dev/null || shift $#
    [[ "${1:-}" == -- ]] && shift
    local dir; dir="$(run_dir "$exp" "$model" "$run")"
    # Smoke: 1 step, or 1 step past the staged checkpoint-1 for a continuation.
    (( SMOKE )) && { steps=$((${#resume} ? 2 : 1)); segments=1; }
    if distill_done "$dir" "$steps" || protected "$dir"; then skip "train $run" "$dir"; return; fi
    local arg beta="" override
    for arg in "$@"; do [[ "$arg" == "$KEY_BETA="* ]] && beta="${arg#*=}"; done
    local args=(
        --config configs/distill/base.yaml
        --config "configs/models/$model.yaml"
        --config "configs/distill/$(method_config "$method").yaml"
        --set "$KEY_MODEL=$(model_path "$model")"
        --set "$KEY_SEED=$seed"
        --set "$KEY_TRAIN_FILE=$TRAIN_FILE"
    )
    for override in $(paper_overrides "$model" "$method" "$beta"); do args+=(--set "$override"); done
    args+=("$@" --output-dir "$dir")
    [[ -n "$resume" ]] && args+=(--resume "$resume/checkpoint-$((SMOKE ? 1 : 100))")
    local envs=(--env "RUN_DIR=$dir" --env "MAX_STEPS=$steps")
    local kind=distill; [[ "$method" == opsd_matched ]] && kind=matched
    local i prev="" first="$after"
    for (( i = 1; i <= segments; i++ )); do
        if (( i == 1 )); then
            submit_job --name "${exp}_${model}_${run}_train" --time "$(job_time "$kind" "$model")" \
                --after "$first" "${envs[@]}" train_distill "${args[@]}"
        else
            # Later segments auto-resume after a timeout or failure and exit at once when complete.
            submit_job --name "${exp}_${model}_${run}_train$i" --time "$(job_time "$kind" "$model")" \
                --afterany "$prev" "${envs[@]}" train_distill "${args[@]}"
        fi
        prev="$JOB_ID"
    done
}

# baseline_train EXP MODEL RUN sft|grpo SEED AFTER
baseline_train() {
    local exp="$1" model="$2" run="$3" method="$4" seed="$5" after="$6"
    local dir; dir="$(run_dir "$exp" "$model" "$run")"
    local max=100; [[ "$method" == grpo ]] && max=200; (( SMOKE )) && max=1
    # The final checkpoint marks a finished run (the root adapter is also written by partial segments).
    if adapter_done "$dir/checkpoint-$max" || protected "$dir"; then skip "train $run" "$dir"; return; fi
    local args=(--model_name_or_path "$(model_path "$model")" --output_dir "$dir" --seed "$seed" "${BASELINE_ARGS[@]}")
    if [[ "$method" == sft ]]; then args+=("${SFT_ARGS[@]}"); else args+=("${GRPO_ARGS[@]}"); fi
    if [[ "$method" == sft ]] || (( SMOKE )); then
        submit_job --name "${exp}_${model}_${run}_train" --time "$(job_time "$method" "$model")" \
            --after "$after" "train_$method" "${args[@]}"
        return
    fi
    local seg stop prev="$after" done_step
    seg="$(grpo_segment "$model")"
    done_step="$(latest_checkpoint "$dir")"
    for (( stop = seg; stop <= max; stop += seg )); do
        (( stop <= done_step )) && continue
        if (( seg < max )); then
            submit_job --name "${exp}_${model}_${run}_train_to$stop" --time "$(job_time grpo "$model")" \
                --after "$prev" train_grpo "${args[@]}" --stop_after_step "$stop"
        else
            submit_job --name "${exp}_${model}_${run}_train" --time "$(job_time grpo "$model")" \
                --after "$prev" train_grpo "${args[@]}"
        fi
        prev="$JOB_ID"
    done
}

# merge_job EXP MODEL NAME ADAPTER_DIR AFTER  -> MERGED (path), JOB_ID
merge_job() {
    local exp="$1" model="$2" name="$3" adapter="$4" after="$5"
    MERGED="$(merged_dir "$exp" "$model" "$name")"
    if [[ -f "$MERGED/config.json" ]] || protected "$MERGED"; then skip "merge $name" "$MERGED"; return; fi
    submit_job --name "${exp}_${model}_${name}_merge" --time "$(job_time merge)" --after "$after" \
        merge --base "$(model_path "$model")" --adapter "$adapter" --output "$MERGED"
}

# eval_jobs EXP MODEL NAME MODEL_DIR SEED AFTER [BENCHMARK...]  -> one job per benchmark
eval_jobs() {
    local exp="$1" model="$2" name="$3" path="$4" seed="$5" after="$6"; shift 6
    local benches=("$@") out bench
    (( ${#benches[@]} )) || benches=("${BENCHMARKS[@]}")
    (( SMOKE )) && benches=("${benches[0]}")
    out="$(eval_dir "$exp" "$model" "$name")"
    for bench in "${benches[@]}"; do
        if [[ -f "$out/$bench/samples.jsonl" ]] || protected "$out/$bench"; then skip "eval $name/$bench" "$out/$bench"; continue; fi
        submit_job --name "${exp}_${model}_${name}_${bench}" --time "$(job_time eval)" --after "$after" \
            eval --model "$path" --output-dir "$out" --benchmarks "$bench" --seed "$seed" --num-samples 12 \
            --max-num-seqs "$(max_num_seqs "$model")" --log-vllm-stats
    done
}

# merge_eval EXP MODEL NAME ADAPTER_DIR AFTER [BENCHMARK...]: merge a student adapter, then evaluate it (seed 42).
merge_eval() {
    local exp="$1" model="$2" name="$3" adapter="$4" after="$5"; shift 5
    merge_job "$exp" "$model" "$name" "$adapter" "$after"
    eval_jobs "$exp" "$model" "$name" "$MERGED" 42 "$JOB_ID" "$@"
}

# generate_job EXP MODEL NAME CHECKPOINT_DIR ROLE INPUT AFTER [TIME_KIND] [MAX_NUM_SEQS] -> GEN_OUT, JOB_ID
generate_job() {
    local exp="$1" model="$2" name="$3" ckpt="$4" role="$5" input="$6" after="$7" kind="${8:-generate}" seqs="${9:-64}"
    GEN_OUT="$(gen_dir "$exp" "$model" "$name")/$role.jsonl"
    if [[ -f "$GEN_OUT" ]] || protected "$GEN_OUT"; then skip "generate $name/$role" "$GEN_OUT"; return; fi
    submit_job --name "${exp}_${model}_${name}_${role}_gen" --time "$(job_time "$kind")" --after "$after" \
        generate --base-model "$(model_path "$model")" --adapter "$ckpt/$role" --policy "$role" \
        --dataset "$input" --output "$GEN_OUT" --max-num-seqs "$seqs"
}

# judge_job EXP MODEL NAME AFTER INPUT...: judge the terminal answers of generation files
# (written next to each input as <name>_judged.jsonl).
judge_job() {
    local exp="$1" model="$2" name="$3" after="$4"; shift 4
    local input todo=()
    for input in "$@"; do
        local output="${input%.jsonl}_judged.jsonl"
        if [[ -f "$output" ]] || protected "$output"; then skip "judge $name/$(basename "$input")" "$output"; else todo+=("$input"); fi
    done
    (( ${#todo[@]} )) || { JOB_ID=""; return; }
    submit_job --name "${exp}_${model}_${name}_judge" --time "$(job_time judge)" --after "$after" judge --inputs "${todo[@]}"
}

# analysis_job MODULE...: CPU summaries (regenerated under results/<exp>/) after every submitted job
# has finished; modules run in order, each after the previous one (later ones may read earlier outputs).
analysis_job() {
    (( ANALYSIS && ! SMOKE )) || return 0
    local deps="${ALL_JOBS[*]}" prev="" module
    for module in "$@"; do
        submit_job --name "analysis_$module" --time "$(job_time analysis)" --afterany "$deps" --after "$prev" \
            analysis "$module" --overwrite
        prev="$JOB_ID"
    done
}

finish() {
    note "$JOB_COUNT job(s) $( (( DRY_RUN )) && echo 'printed (dry run)' || echo submitted ), $CHAIN_COUNT chain(s)"
}

# ---------------------------------------------------------------- job side

# Environment for a Slurm job: modules, conda, caches on job-local disk, smoke flag.
job_setup() {
    set -euo pipefail
    if [[ -n "${CLUSTER_MODULES:-}" ]] && command -v module > /dev/null 2>&1; then
        module purge
        # shellcheck disable=SC2086
        module load $CLUSTER_MODULES
    fi
    if [[ -n "${CONDA_ENV:-}" ]] && command -v conda > /dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV"
    fi
    cd "$DUALOPSD_REPO"
    export PYTHONPATH="$DUALOPSD_REPO${PYTHONPATH:+:$PYTHONPATH}"
    local cache="${TMPDIR:-/tmp}/dualopsd_${SLURM_JOB_ID:-$$}"
    export VLLM_CACHE_ROOT="$cache/vllm" TRITON_CACHE_DIR="$cache/triton" TORCHINDUCTOR_CACHE_DIR="$cache/torchinductor"
    mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"
    SMOKE_ARGS=()
    [[ "${SMOKE:-0}" == 1 ]] && SMOKE_ARGS=(--smoke)
    echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date '+%F %T'); smoke=${SMOKE:-0}"
}

# Log GPU memory/utilisation every GPU_MONITOR_INTERVAL seconds until the job exits.
start_gpu_monitor() {
    command -v nvidia-smi > /dev/null 2>&1 || return 0
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    local dir="${DUALOPSD_LOG_DIR:-$DUALOPSD_SCRATCH/logs}/gpu_monitor"
    local log="$dir/${SLURM_JOB_NAME:-job}_${SLURM_JOB_ID:-$$}_gpu.csv"
    mkdir -p "$dir"
    echo "timestamp,index,memory.used MiB,memory.total MiB,utilization.gpu %" > "$log"
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits --loop="${GPU_MONITOR_INTERVAL:-5}" >> "$log" &
    GPU_MONITOR_PID=$!
    trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT
}
