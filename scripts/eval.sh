#!/bin/bash
# Shared benchmark evaluation logic.
# Dataset-specific scripts should set the following before sourcing this file:
#   - EVAL_CONFIG_NAME: Hydra config name (e.g. "mnist_lenet") from config/eval/
#   - benchmarks:       array of benchmark ids (config_map keys)
#   - methods:          array of explainer group names (config/eval/explainer/*.yaml)
# and source their own eval_defs.sh (EXPL_SWEEP).

export PYTHONPATH="$PYTHONPATH:$(dirname $(dirname $(realpath $0)))"

# Disable pathname expansion so bracketed Hydra override values in
# ${EXPL_SWEEP[*]} (e.g. `key=[a,b],[c,d]`) are passed through to Python
# as-is instead of being globbed against cwd.
set -o noglob

PARALLEL="${PARALLEL:-true}"

REGENERATE_EXPLANATIONS=false
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --regenerate-explanations)
            REGENERATE_EXPLANATIONS=true
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done
if [ "$REGENERATE_EXPLANATIONS" = true ]; then
    EXTRA_ARGS+=("+regenerate_explanations=true")
fi

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
        $sweep "${EXTRA_ARGS[@]}"
}

# Populate the local cache (metadata + ckpt) once per benchmark so that
# parallel sweep jobs don't race on Hub downloads.
for bench in "${benchmarks[@]}"; do
    python scripts/prefetch_bench.py \
        --config-name "$EVAL_CONFIG_NAME" \
        bench="$bench" "${EXTRA_ARGS[@]}" \
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
