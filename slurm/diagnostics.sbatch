#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
# GPU re-scoring diagnostics: python -m dualopsd posthoc|recurrence ARGS (first argument selects the command).
source "${DUALOPSD_REPO:-$SLURM_SUBMIT_DIR}/scripts/lib.sh"
job_setup
command="${1:?posthoc or recurrence}"; shift
case "$command" in posthoc|recurrence) ;; *) die "unknown diagnostic: $command" ;; esac
start_gpu_monitor
python -m dualopsd "$command" "$@" "${SMOKE_ARGS[@]}"
