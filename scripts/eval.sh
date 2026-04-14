#!/bin/bash
# Shared benchmark evaluation logic.
# Dataset-specific scripts should set the following before sourcing this file:
#   - EVAL_CONFIG_NAME: Hydra config name (e.g. "mnist_lenet") from config/eval/
#   - RUN_SCRIPT:       path to run_bench_eval.py
#   - benchmarks:       array of benchmark ids (config_map keys)
#   - methods:          array of explainer group names (config/eval/explainer/*.yaml)
# and source their own eval_defs.sh (EXPL_SWEEP).

export PYTHONPATH="$PYTHONPATH:$(dirname $(dirname $(realpath $0)))"

PARALLEL=false
CACHE_DIR="${CACHE_DIR:-./tmp_bench}"
RESULTS_DIR="${RESULTS_DIR:-./eval_results}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel) PARALLEL=$2; shift 2 ;;
        --cache-dir) CACHE_DIR=$2; shift 2 ;;
        --results-dir) RESULTS_DIR=$2; shift 2 ;;
        --device) DEVICE=$2; shift 2 ;;
        --batch-size) BATCH_SIZE=$2; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

mkdir -p "$CACHE_DIR" "$RESULTS_DIR" logs

run_eval() {
    local bench=$1 method=$2 sweep=$3
    local multirun=""
    [ -n "$sweep" ] && multirun="--multirun"
    python "$RUN_SCRIPT" \
        --config-name "$EVAL_CONFIG_NAME" \
        bench="$bench" explainer="$method" \
        cache_dir="$CACHE_DIR" results_dir="$RESULTS_DIR" \
        device="$DEVICE" batch_size="$BATCH_SIZE" \
        $sweep $multirun "${EXTRA_ARGS[@]}"
}

for bench in "${benchmarks[@]}"; do
    for method in "${methods[@]}"; do
        sweep="${EXPL_SWEEP[$method]}"
        log="logs/${bench}__${method}.log"
        if [ "$PARALLEL" = true ]; then
            run_eval "$bench" "$method" "$sweep" > "$log" 2>&1 &
        else
            run_eval "$bench" "$method" "$sweep" > "$log" 2>&1
        fi
    done
done

wait
