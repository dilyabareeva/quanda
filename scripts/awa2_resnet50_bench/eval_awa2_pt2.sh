#!/bin/bash

source "$(dirname "$0")/eval_defs.sh"

EVAL_CONFIG_NAME="awa2_resnet50"


benchmarks=(
    awa2_mixed_datasets
)

methods=(
    similarity
    representer_points
    random
)
PARALLEL=false

source "$(dirname "$0")/../eval.sh" "$@"


methods=(
    tracincpfast
)

source "$(dirname "$0")/../eval.sh" "$@"


methods=(
    trak
)

source "$(dirname "$0")/../eval.sh" "$@"