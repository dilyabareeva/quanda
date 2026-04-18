#!/bin/bash

source "$(dirname "$0")/bench_defs.sh"

CONFIG_NAME="bert_qnli"
CONFIG_MAP_PREFIX="qnli"

source "$(dirname "$0")/../train_lds.sh" \
    --n-lds-parallel 10 \
    --hf-push-sleep 60 \
    --skip-main-train false \
    --gpu-split true \
    --start 0 \
    --end 1 \
    --train-only true \
