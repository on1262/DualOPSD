#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# Teacher/student generation on the held-out 500-problem set: python -m dualopsd generate ARGS.
source "${DUALOPSD_REPO:-$SLURM_SUBMIT_DIR}/scripts/lib.sh"
job_setup
start_gpu_monitor
python -m dualopsd generate "$@" "${SMOKE_ARGS[@]}"
