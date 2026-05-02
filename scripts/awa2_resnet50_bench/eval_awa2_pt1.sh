#!/bin/bash

source "$(dirname "$0")/eval_defs.sh"

EVAL_CONFIG_NAME="awa2_resnet50"

benchmarks=(
    awa2_mixed_datasets
)

methods=(
    arnoldi
)
PARALLEL=false

source "$(dirname "$0")/../eval.sh" "$@"