#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
# Terminal-answer judge (JUDGE_MODEL from configs/user.env): python -m dualopsd judge ARGS.
source "${DUALOPSD_REPO:-$SLURM_SUBMIT_DIR}/scripts/lib.sh"
job_setup
start_gpu_monitor
python -m dualopsd judge --model "$JUDGE_MODEL" "$@" "${SMOKE_ARGS[@]}"
