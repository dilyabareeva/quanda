#!/bin/bash
# Shared benchmark evaluation logic.
# Dataset-specific scripts should set the following before sourcing this file:
#   - EVAL_CONFIG_NAME: Hydra config name (e.g. "mnist_lenet") from config/eval/
#   - benchmarks:       array of benchmark ids (config_map keys)
#   - methods:          array of explainer group names (config/eval/explainer/*.yaml)
# and source their own eval_defs.sh (EXPL_SWEEP).

export PYTHONPATH="$PYTHONPATH:$(dirname $(dirname $(realpath $0)))"

PARALLEL=true
CACHE_DIR="${CACHE_DIR:-}"
RESULTS_DIR="${RESULTS_DIR:-}"
DEVICE="${DEVICE:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
MAX_EVAL_N="${MAX_EVAL_N:-}"
EVAL_SEED="${EVAL_SEED:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel) PARALLEL=$2; shift 2 ;;
        --cache-dir) CACHE_DIR=$2; shift 2 ;;
        --results-dir) RESULTS_DIR=$2; shift 2 ;;
        --device) DEVICE=$2; shift 2 ;;
        --batch-size) BATCH_SIZE=$2; shift 2 ;;
        --max-eval-n) MAX_EVAL_N=$2; shift 2 ;;
        --eval-seed) EVAL_SEED=$2; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

mkdir -p logs
[ -n "$CACHE_DIR" ] && mkdir -p "$CACHE_DIR"
[ -n "$RESULTS_DIR" ] && mkdir -p "$RESULTS_DIR"

run_eval() {
    local bench=$1 method=$2 sweep=$3
    local multirun=""
    [ -n "$sweep" ] && multirun="--multirun"
    local overrides=(bench="$bench" explainer="$method")
    [ -n "$CACHE_DIR" ] && overrides+=(cache_dir="$CACHE_DIR")
    [ -n "$RESULTS_DIR" ] && overrides+=(results_dir="$RESULTS_DIR")
    [ -n "$DEVICE" ] && overrides+=(device="$DEVICE")
    [ -n "$BATCH_SIZE" ] && overrides+=(batch_size="$BATCH_SIZE")
    [ -n "$MAX_EVAL_N" ] && overrides+=(max_eval_n="$MAX_EVAL_N")
    [ -n "$EVAL_SEED" ] && overrides+=(eval_seed="$EVAL_SEED")
    python scripts/run_bench_eval.py \
        --config-name "$EVAL_CONFIG_NAME" \
        "${overrides[@]}" \
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
