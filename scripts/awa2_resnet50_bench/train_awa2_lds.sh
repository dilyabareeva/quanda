#!/bin/bash

source "$(dirname "$0")/bench_defs.sh"

CONFIG_NAME="awa2_resnet50"
CONFIG_MAP_PREFIX="awa2"

source "$(dirname "$0")/../train_lds.sh" \
    --n-lds-parallel 4 \
    --hf-push-sleep 30 \
    --gpu-split false \
    "$@"
