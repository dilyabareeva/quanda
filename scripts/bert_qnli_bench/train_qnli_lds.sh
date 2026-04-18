#!/bin/bash

source "$(dirname "$0")/bench_defs.sh"

CONFIG_NAME="bert_qnli"
CONFIG_MAP_PREFIX="qnli"

source "$(dirname "$0")/../train_lds.sh" \
    --n-lds-parallel 6 \
    --hf-push-sleep 60 \
    --skip-main-train true \
    --gpu-split true \
    --start 0 \
    --end 69 \
    --train-only true \
