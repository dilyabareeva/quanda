#!/bin/bash

source "$(dirname "$0")/bench_defs.sh"

CONFIG_NAME="cifar_resnet9"

source "$(dirname "$0")/../train_lds.sh" \
    --n-lds-parallel 20 \
    --hf-push-sleep 20
