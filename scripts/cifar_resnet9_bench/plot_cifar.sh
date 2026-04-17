#!/bin/bash

DIR="$(dirname "$0")"
ROOT="$(realpath "$DIR/../..")"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/eval_results}"
OUT="${OUT:-$DIR/bar_rank.png}"

python "$DIR/../plot_results.py" \
    --results-dir "$RESULTS_DIR" \
    --config "$DIR/cifar_plot_config.json" \
    --out "$OUT" \
    "$@"
