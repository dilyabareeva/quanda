#!/bin/bash

source "$(dirname "$0")/eval_defs.sh"

EVAL_CONFIG_NAME="bert_qnli"

benchmarks=(
    qnli_class_detection
    qnli_mislabeling_detection
)

methods=(
    #similarity
    #kronfluence
    trak
    #random
    representer_points
)
HYDRA_FULL_ERROR=1
CUDA_LAUNCH_BLOCKING=1 
PARALLEL=false 

#source "$(dirname "$0")/../eval.sh" "$@"

benchmarks=(
    qnli_linear_datamodeling
    qnli_top_k_cardinality
    qnli_model_randomization
)

source "$(dirname "$0")/../eval.sh" "$@"