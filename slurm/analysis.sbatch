#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# CPU summaries and tables: python -m analysis.MODULE ARGS (first argument is MODULE).
source "${DUALOPSD_REPO:-$SLURM_SUBMIT_DIR}/scripts/lib.sh"
job_setup
module_name="${1:?analysis module}"; shift
python -m "analysis.$module_name" "$@"
