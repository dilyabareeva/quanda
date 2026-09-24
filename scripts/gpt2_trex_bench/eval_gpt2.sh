#!/bin/bash

source "$(dirname "$0")/eval_defs.sh"

EVAL_CONFIG_NAME="gpt2_trex"

benchmarks=(
    gpt2_trex_openwebtext_ft_tail_patch
    gpt2_trex_openwebtext_ft_mrr
    gpt2_trex_openwebtext_ft_recall_at_k
)

methods=(
    #random
    #kronfluence_gpt2
    dattri_if_datainf
    #dattri_trak
    #similarity
)

HYDRA_FULL_ERROR=1
CUDA_LAUNCH_BLOCKING=1
PARALLEL=false

source "$(dirname "$0")/../eval.sh" "$@"
