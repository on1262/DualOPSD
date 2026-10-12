#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# Benchmark avg@k evaluation (AIME24/AIME25/HMMT25): python -m dualopsd eval ARGS.
source "${DUALOPSD_REPO:-$SLURM_SUBMIT_DIR}/scripts/lib.sh"
job_setup
start_gpu_monitor
python -m dualopsd eval "$@" "${SMOKE_ARGS[@]}"
