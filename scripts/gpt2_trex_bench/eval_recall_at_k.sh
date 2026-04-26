#!/bin/bash

source "$(dirname "$0")/eval_defs.sh"

EVAL_CONFIG_NAME="gpt2_trex"

benchmarks=(
    gpt2_trex_openwebtext_ft_recall_at_k
)

methods=(
    random
    kronfluence_gpt2
    dattri_if_datainf
    dattri_trak
    similarity
)

specified_methods=()
forwarded=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            specified_methods+=("$2")
            shift 2
            ;;
        *)
            forwarded+=("$1")
            shift
            ;;
    esac
done
set -- "${forwarded[@]}"

if [ "${#specified_methods[@]}" -gt 0 ]; then
    methods=("${specified_methods[@]}")
fi

HYDRA_FULL_ERROR=1
CUDA_LAUNCH_BLOCKING=1
PARALLEL=false

source "$(dirname "$0")/../eval.sh" "$@"
