#!/bin/bash

set -a; source "$(dirname "${BASH_SOURCE[0]}")/../.env"; set +a
export PYTHONPATH="$PYTHONPATH:$(dirname $(dirname $(realpath $0)))"

START=""
END=""
BATCH_SIZE=8
MAX_EVAL_N=1000
EVAL_SEED=42
INFERENCE_BATCH_SIZE=""
DEVICE="cuda:0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config-map-key) CONFIG_MAP_KEY=$2; shift 2 ;;
        --start) START=$2; shift 2 ;;
        --end) END=$2; shift 2 ;;
        --batch-size) BATCH_SIZE=$2; shift 2 ;;
        --max-eval-n) MAX_EVAL_N=$2; shift 2 ;;
        --eval-seed) EVAL_SEED=$2; shift 2 ;;
        --inference-batch-size) INFERENCE_BATCH_SIZE=$2; shift 2 ;;
        --device) DEVICE=$2; shift 2 ;;
        *) shift ;;
    esac
done

if [ -z "$CONFIG_MAP_KEY" ]; then
    echo "Error: --config-map-key (or CONFIG_MAP_KEY) required." >&2
    exit 1
fi
if [ -z "$START" ] || [ -z "$END" ]; then
    echo "Error: --start and --end required." >&2
    exit 1
fi

BENCH_SAVE_DIR="${QUANDA_ROOT_DIR:?set in .env}/eval_bench/${CONFIG_MAP_KEY%_linear_datamodeling}"

CONFIG_PATH=$(python -c "
from quanda.benchmarks.resources.config_map import config_map
print(str(config_map['${CONFIG_MAP_KEY}']))
")

log_dir="logs/${CONFIG_MAP_KEY}_subset_logits"
mkdir -p "$log_dir"

for i in $(seq "$START" "$((END - 1))"); do
    args=(
        --config-path "$CONFIG_PATH"
        --bench-save-dir "$BENCH_SAVE_DIR"
        --idx "$i"
        --device "$DEVICE"
        --batch-size "$BATCH_SIZE"
        --max-eval-n "$MAX_EVAL_N"
        --eval-seed "$EVAL_SEED"
    )
    [ -n "$INFERENCE_BATCH_SIZE" ] && \
        args+=(--inference-batch-size "$INFERENCE_BATCH_SIZE")

    python scripts/precompute_subset_logits_subset.py "${args[@]}" \
            > "${log_dir}/subset_${i}.log" 2>&1

done
wait
