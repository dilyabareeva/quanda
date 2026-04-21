#!/bin/bash

source "$(dirname "$0")/eval_defs.sh"

EVAL_CONFIG_NAME="bert_qnli"

benchmarks=(
    qnli_class_detection
    qnli_mislabeling_detection
    qnli_linear_datamodeling
    #qnli_top_k_cardinality
    #qnli_model_randomization
)

methods=(
    similarity
    dattri_arnoldi
    dattri_ekfac
    dattri_tracin
    dattri_graddot
    dattri_gradcos
    trak
    random
)

PARALLEL=false
source "$(dirname "$0")/../eval.sh" --regenerate-explanations "$@"
