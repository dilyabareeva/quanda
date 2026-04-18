#!/bin/bash
# Shared benchmark evaluation logic.
# Dataset-specific scripts should set the following before sourcing this file:
#   - EVAL_CONFIG_NAME: Hydra config name (e.g. "mnist_lenet") from config/eval/
#   - benchmarks:       array of benchmark ids (config_map keys)
#   - methods:          array of explainer group names (config/eval/explainer/*.yaml)
# and source their own eval_defs.sh (EXPL_SWEEP).

export PYTHONPATH="$PYTHONPATH:$(dirname $(dirname $(realpath $0)))"

PARALLEL="${PARALLEL:-true}"
EXTRA_ARGS=("$@")

LOG_DIR="logs_${EVAL_CONFIG_NAME}"
mkdir -p "$LOG_DIR"

run_eval() {
    local bench=$1 method=$2 sweep=$3
    local multirun=""
    [ -n "$sweep" ] && multirun="--multirun"
    python scripts/run_bench_eval.py \
        --config-name "$EVAL_CONFIG_NAME" \
        bench="$bench" explainer="$method" \
        $sweep $multirun "${EXTRA_ARGS[@]}"
}

for bench in "${benchmarks[@]}"; do
    for method in "${methods[@]}"; do
        sweep="${EXPL_SWEEP[$method]}"
        log="${LOG_DIR}/${bench}__${method}.log"
        if [ "$PARALLEL" = true ]; then
            run_eval "$bench" "$method" "$sweep" > "$log" 2>&1 &
        else
            run_eval "$bench" "$method" "$sweep" > "$log" 2>&1
        fi
    done
done

wait
