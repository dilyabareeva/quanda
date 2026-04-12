#!/bin/bash

source "$(dirname "$0")/bench_defs.sh"

CONFIG_NAME="bert_qnli"
CONFIG_MAP_PREFIX="qnli"

benchmarks=(
    #ClassDetection
    MislabelingDetection
    #LDS
)

source "$(dirname "$0")/../train.sh" "$@"
