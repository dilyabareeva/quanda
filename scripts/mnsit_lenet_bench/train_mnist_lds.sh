#!/bin/bash

source "$(dirname "$0")/bench_defs.sh"

CONFIG_NAME="mnist_lenet"

source "$(dirname "$0")/../train_lds.sh" \
    --n-lds-parallel 16 \
    --hf-push-sleep 60
