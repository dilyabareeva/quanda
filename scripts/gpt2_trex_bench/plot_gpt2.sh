#!/bin/bash

DIR="$(dirname "$0")"
ROOT="$(realpath "$DIR/../..")"
RESULTS_DIR="/data2/bareeva/Projects/quanda/cluster_output_new2/eval_results/gpt2_trex"
OUT="${OUT:-$DIR/bar_rank.png}"

python "$DIR/../plot_results.py" \
    --results-dir "$RESULTS_DIR" \
    --config "$DIR/gpt2_plot_config.json" \
    --out "$OUT" \
    "$@"
