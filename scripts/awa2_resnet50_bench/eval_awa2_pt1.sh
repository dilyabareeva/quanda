#!/bin/bash

source "$(dirname "$0")/eval_defs.sh"

EVAL_CONFIG_NAME="awa2_resnet50"

benchmarks=(
    awa2_class_detection
    awa2_subclass_detection
    awa2_shortcut_detection
    awa2_mixed_datasets
)

methods=(
    similarity
    representer_points
    tracincpfast
    arnoldi
    trak
    random
)
PARALLEL=false

source "$(dirname "$0")/../eval.sh" "$@"