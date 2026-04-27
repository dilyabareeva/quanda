#!/bin/bash

source "$(dirname "$0")/eval_defs.sh"

EVAL_CONFIG_NAME="mnist_lenet"

benchmarks=(
    mnist_class_detection
    mnist_subclass_detection
    mnist_shortcut_detection
    mnist_mixed_datasets
)

methods=(
    similarity
    representer_points
    tracincpfast
    arnoldi
    trak
    random
)

source "$(dirname "$0")/../eval.sh" "$@"
