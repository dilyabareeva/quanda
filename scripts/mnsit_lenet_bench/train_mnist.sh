#!/bin/bash

source "$(dirname "$0")/bench_defs.sh"

CONFIG_NAME="mnist_lenet"
CONFIG_MAP_PREFIX="mnist"

benchmarks=(
    ClassDetection
    SubclassDetection
    MixedDatasets
    ShortcutDetection
    MislabelingDetection
    LDS
)

source "$(dirname "$0")/../train.sh" "$@"
