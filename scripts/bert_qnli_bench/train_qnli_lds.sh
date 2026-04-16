#!/bin/bash

source "$(dirname "$0")/bench_defs.sh"

CONFIG_NAME="bert_qnli"

source "$(dirname "$0")/../train_lds.sh" \
    --n-lds-parallel 3 \
    --hf-push-sleep 60
