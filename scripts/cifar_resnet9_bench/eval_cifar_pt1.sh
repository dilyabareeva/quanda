#!/bin/bash

source "$(dirname "$0")/eval_defs.sh"

EVAL_CONFIG_NAME="cifar_resnet9"

benchmarks=(
    cifar_class_detection
    cifar_subclass_detection
    cifar_shortcut_detection
    cifar_mixed_datasets
)

methods=(
    #similarity
    #representer_points
    #tracincpfast
    arnoldi
    #trak
    #random
)

source "$(dirname "$0")/../eval.sh" "$@"


benchmarks=(
    cifar_linear_datamodeling
    cifar_top_k_cardinality
    cifar_model_randomization
    cifar_mislabeling_detection
)

methods=(
    #similarity
    #representer_points
    #tracincpfast
    arnoldi
    #trak
    #random
)

source "$(dirname "$0")/../eval.sh" "$@"