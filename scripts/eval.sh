#!/bin/bash
# Shared benchmark evaluation logic.

export PYTHONPATH="$PYTHONPATH:$(dirname $(dirname $(realpath $0)))"

set -o noglob

PARALLEL="${PARALLEL:-true}"

REGEN_OVERRIDE=""
[ "$1" = "--regenerate-explanations" ] && REGEN_OVERRIDE="+regenerate_explanations=true"

LOG_DIR="logs/${EVAL_CONFIG_NAME}"
mkdir -p "$LOG_DIR"

run_eval() {
    local bench=$1 method=$2 sweep=$3
    local multirun=""
    [ -n "$sweep" ] && multirun="--multirun"
    env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python scripts/run_bench_eval.py \
        --config-name "$EVAL_CONFIG_NAME" $multirun \
        bench="$bench" explainer="$method" \
        $sweep $REGEN_OVERRIDE
}

# Populate the local cache (metadata + ckpt) once per benchmark 
for bench in "${benchmarks[@]}"; do
    python scripts/prefetch_bench.py \
        --config-name "$EVAL_CONFIG_NAME" \
        bench="$bench" \
        >> "${LOG_DIR}/caching.log" 2>&1
done

for bench in "${benchmarks[@]}"; do
    for method in "${methods[@]}"; do
        sweep="${EXPL_SWEEP[$method]}"
        log="${LOG_DIR}/${bench}__${method}.log"
        run_eval "$bench" "$method" "$sweep" > "$log" 2>&1 &
    done
    [ "$PARALLEL" = true ] || wait
done

wait
