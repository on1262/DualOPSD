#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# Distillation training (OPSD, DualOPSD, ablations): python -m dualopsd train ARGS.
# Submitted by scripts/lib.sh (account, partition, GPU and logs come from configs/user.env).
# Env: RUN_DIR and MAX_STEPS (exit at once when checkpoint-MAX_STEPS exists, so a chained
# segment after a finished run is a no-op). The trainer auto-resumes from RUN_DIR.
source "${DUALOPSD_REPO:-$SLURM_SUBMIT_DIR}/scripts/lib.sh"
job_setup
if distill_done "${RUN_DIR:?}" "${MAX_STEPS:?}"; then
    echo "already complete: $RUN_DIR/checkpoint-$MAX_STEPS"
    exit 0
fi
# vLLM sleep mode is incompatible with expandable segments.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
start_gpu_monitor
python -m dualopsd train "$@" "${SMOKE_ARGS[@]}"
