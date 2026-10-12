#!/bin/bash
# Build the training file, the held-out 500-problem set and the AIME24/AIME25/HMMT25
# benchmark files under $DATA_DIR / $BENCH_DIR (configs/user.env). Runs locally (CPU).
#   --download-models  optional: pre-download Qwen3-1.7B/4B/8B to $DUALOPSD_SCRATCH/base_models
#                      (then point MODEL_QWEN3_* there); by default the Hugging Face ids
#                      in configs/user.env are fetched into $HF_HOME on first use
#   --dry-run          print the commands only
# Other arguments are passed to `python -m dualopsd prepare-data` (default target: all).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

dry_run=0 download=0 args=()
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1 ;;
        --download-models) download=1 ;;
        -h|--help) sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) args+=("$1") ;;
    esac
    shift
done

run() {
    echo "+ $(printf '%q ' "$@")"
    (( dry_run )) || "$@"
}

cd "$DUALOPSD_REPO"
run mkdir -p "$DATA_DIR" "$BENCH_DIR"
(( ${#args[@]} )) || args=(all)
run python -m dualopsd prepare-data "${args[@]}"
if (( download )); then
    for name in Qwen3-1.7B Qwen3-4B Qwen3-8B; do
        dest="$DUALOPSD_SCRATCH/base_models/$name"
        if [[ -e "$dest" ]]; then echo "skip $dest (exists)"; continue; fi
        run huggingface-cli download "Qwen/$name" --local-dir "$dest"
    done
fi
