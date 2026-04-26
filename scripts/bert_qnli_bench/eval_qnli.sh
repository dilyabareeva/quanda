#!/bin/bash

source "$(dirname "$0")/eval_defs.sh"

EVAL_CONFIG_NAME="bert_qnli"

benchmarks=(
    qnli_class_detection
    #qnli_mislabeling_detection
    #qnli_mixed_datasets
    #qnli_top_k_cardinality
    #qnli_model_randomization
    #qnli_linear_datamodeling
)

methods=(
    #similarity
    #kronfluence
    #trak
    #random
    #representer_points
    dattri_if_datainf
)
HYDRA_FULL_ERROR=1
CUDA_LAUNCH_BLOCKING=1 
PARALLEL=false 


source "$(dirname "$0")/../eval.sh" "$@"