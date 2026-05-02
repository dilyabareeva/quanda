#!/bin/bash

source "$(dirname "$0")/bench_defs.sh"

CONFIG_NAME="awa2_resnet50"
CONFIG_MAP_PREFIX="awa2"

benchmarks=(
    #ClassDetection
    #SubclassDetection
    MixedDatasets
    #ShortcutDetection
    #MislabelingDetection
)

source "$(dirname "$0")/../train.sh" "$@"
