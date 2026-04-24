#!/bin/bash
# Shared orchestrator: loops a range of LDS subset indices through the
# per-subset CLI. Source this from a per-bench shard that sets CONFIG_MAP_KEY
# (e.g. "qnli_linear_datamodeling") and CONFIG_MAP_PREFIX (e.g. "qnli"),
# and passes --start / --end plus any overrides.

export PYTHONPATH="$PYTHONPATH:$(dirname $(dirname $(realpath $0)))"

N_PARALLEL=1
START=""
END=""
BATCH_SIZE=8
MAX_EVAL_N=1000
EVAL_SEED=42
INFERENCE_BATCH_SIZE=""
DEVICE="cuda:0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --n-parallel) N_PARALLEL=$2; shift 2 ;;
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
    echo "Error: CONFIG_MAP_KEY must be set before sourcing this script." >&2
    exit 1
fi
if [ -z "$START" ] || [ -z "$END" ]; then
    echo "Error: --start and --end required." >&2
    exit 1
fi

if [ -d "/data/cluster/users/bareeva" ]; then
    BENCH_SAVE_DIR="/data/cluster/users/bareeva/quanda_output_new2/eval_bench/${CONFIG_MAP_PREFIX}"
else
    BENCH_SAVE_DIR="/data2/bareeva/Projects/quanda/cluster_output_new2/eval_bench/${CONFIG_MAP_PREFIX}"
fi

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

    if [ "$N_PARALLEL" -gt 1 ]; then
        while [ "$(jobs -rp | wc -l)" -ge "$N_PARALLEL" ]; do wait -n; done
        python scripts/precompute_subset_logits_subset.py "${args[@]}" \
            > "${log_dir}/subset_${i}.log" 2>&1 &
    else
        python scripts/precompute_subset_logits_subset.py "${args[@]}" \
            > "${log_dir}/subset_${i}.log" 2>&1
    fi
done
wait
