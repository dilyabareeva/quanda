#!/bin/bash

CONFIG_MAP_KEY="awa2_linear_datamodeling"
CONFIG_MAP_PREFIX="awa2"

source "$(dirname "$0")/../compute_lds_subset_logits.sh" "$@"
