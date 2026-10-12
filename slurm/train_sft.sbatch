#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
# SFT baseline: accelerate launch -m dualopsd sft ARGS (single process, bf16).
source "${DUALOPSD_REPO:-$SLURM_SUBMIT_DIR}/scripts/lib.sh"
job_setup
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000))
start_gpu_monitor
accelerate launch --config_file configs/accelerate.yaml --num_processes 1 --main_process_port "$MASTER_PORT" \
    -m dualopsd sft "$@" "${SMOKE_ARGS[@]}"
