#!/bin/bash

source "$(dirname "$0")/eval_defs.sh"

EVAL_CONFIG_NAME="awa2_resnet50"


benchmarks=(
    awa2_mixed_datasets
    awa2_top_k_cardinality
    awa2_model_randomization
    awa2_mislabeling_detection
    awa2_linear_datamodeling
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


