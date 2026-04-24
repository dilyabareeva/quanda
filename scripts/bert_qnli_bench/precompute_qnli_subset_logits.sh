#!/bin/bash

CONFIG_MAP_KEY="qnli_linear_datamodeling"
CONFIG_MAP_PREFIX="qnli"

source "$(dirname "$0")/../precompute_subset_logits_range.sh" "$@"
