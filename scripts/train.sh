#!/bin/bash
# Shared benchmark training logic.
# Dataset-specific scripts should set the following before sourcing this file:
#   - CONFIG_NAME:        Hydra config name (e.g. "mnist_lenet", "cifar_resnet9")
#   - CONFIG_MAP_PREFIX:  Prefix for config_map.py keys (e.g. "mnist", "cifar")
#   - benchmarks:         Array of benchmark names to run
# and source their own bench_defs.sh (BENCH_PARAMS / BENCH_SWEEP).

export PYTHONPATH="$PYTHONPATH:$(dirname $(dirname $(realpath $0)))"

PARALLEL=false
TRAIN_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel) PARALLEL=$2; shift 2 ;;
        --train-only) TRAIN_ONLY=$2; shift 2 ;;
        *) shift ;;
    esac
done

cfg_output_dir="quanda/benchmarks/resources/configs"
commit_tag=$(git rev-parse --short HEAD)
mkdir -p logs

# Map benchmark names to config_map.py keys
declare -A BENCH_CONFIG_MAP_KEY
BENCH_CONFIG_MAP_KEY[ClassDetection]="${CONFIG_MAP_PREFIX}_class_detection"
BENCH_CONFIG_MAP_KEY[SubclassDetection]="${CONFIG_MAP_PREFIX}_subclass_detection"
BENCH_CONFIG_MAP_KEY[MixedDatasets]="${CONFIG_MAP_PREFIX}_mixed_datasets"
BENCH_CONFIG_MAP_KEY[MislabelingDetection]="${CONFIG_MAP_PREFIX}_mislabeling_detection"
BENCH_CONFIG_MAP_KEY[ShortcutDetection]="${CONFIG_MAP_PREFIX}_shortcut_detection"
BENCH_CONFIG_MAP_KEY[LDS]="${CONFIG_MAP_PREFIX}_linear_datamodeling"

get_config_name_from_map() {
    local key=$1
    python -c "
from quanda.benchmarks.resources.config_map import config_map
import os
path = str(config_map['$key'])
print(os.path.splitext(os.path.basename(path))[0])
"
}

run_bench() {
    local bench=$1 params=$2 sweep=$3 id=$4
    if [ "$TRAIN_ONLY" = false ]; then
        python scripts/generate_config.py --config-name "$CONFIG_NAME" hydra.run.dir="hydra_logs" $params id=$id +cfg_file_name=$id +cfg_output_dir=$cfg_output_dir
        python scripts/train.py --config-name "$CONFIG_NAME" bench="$bench" $params $sweep id=$id +cfg_output_dir=$cfg_output_dir +cfg_file_name=$id --multirun
        python scripts/opt_results_to_cfg.py --config-name "$CONFIG_NAME" bench="$bench" $params id=$id +cfg_output_dir=$cfg_output_dir +cfg_file_name=$id
        python scripts/train_and_push_to_hub.py --config-name $id --config-dir $cfg_output_dir
    else
        local config_name
        config_name=$(get_config_name_from_map "${BENCH_CONFIG_MAP_KEY[$bench]}")
        python scripts/train_and_push_to_hub.py --config-name "$config_name" --config-dir $cfg_output_dir
    fi
}

for bench in "${benchmarks[@]}"; do
    params="${BENCH_PARAMS[$bench]}"
    sweep="${BENCH_SWEEP[$bench]}"
    id="${commit_tag}-default_${bench}"
    if [ "$PARALLEL" = true ]; then
        run_bench "$bench" "$params" "$sweep" "$id" > "logs/${bench}.log" 2>&1 &
    else
        run_bench "$bench" "$params" "$sweep" "$id" > "logs/${bench}.log" 2>&1
    fi
done

wait
