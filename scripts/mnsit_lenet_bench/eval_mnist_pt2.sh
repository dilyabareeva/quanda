#!/bin/bash

source "$(dirname "$0")/eval_defs.sh"

EVAL_CONFIG_NAME="mnist_lenet"

benchmarks=(
    mnist_top_k_cardinality
    mnist_model_randomization
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
