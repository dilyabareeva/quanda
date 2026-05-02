#!/bin/bash

source "$(dirname "$0")/eval_defs.sh"

EVAL_CONFIG_NAME="awa2_resnet50"

benchmarks=(
    awa2_class_detection
    awa2_subclass_detection
    awa2_shortcut_detection
    awa2_mixed_datasets
    awa2_linear_datamodeling
    awa2_top_k_cardinality
    awa2_model_randomization
    awa2_mislabeling_detection
)

methods=(
    #similarity
    #representer_points
    trak
    #arnoldi
    #trak
    #random
)
PARALLEL=false

source "$(dirname "$0")/../eval.sh" "$@"
