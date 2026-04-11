#!/bin/bash

source "$(dirname "$0")/bench_defs.sh"

CONFIG_NAME="cifar_resnet9"
CONFIG_MAP_PREFIX="cifar"

benchmarks=(
    ClassDetection
)

source "$(dirname "$0")/../train.sh"
