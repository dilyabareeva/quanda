#!/bin/bash

source "$(dirname "$0")/bench_defs.sh"

CONFIG_NAME="mnist_lenet"
CONFIG_MAP_PREFIX="mnist"

benchmarks=(
    SubclassDetection
    MixedDatasets
)

source "$(dirname "$0")/../train.sh" "$@"
