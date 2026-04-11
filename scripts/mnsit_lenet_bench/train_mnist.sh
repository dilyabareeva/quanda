#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(dirname $(dirname $(realpath $0)))"
source "$(dirname "$0")/bench_defs.sh"

# --- Benchmarks to run ---
benchmarks=(
    SubclassDetection
    MixedDatasets
)

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
BENCH_CONFIG_MAP_KEY[ClassDetection]="mnist_class_detection"
BENCH_CONFIG_MAP_KEY[SubclassDetection]="mnist_subclass_detection"
BENCH_CONFIG_MAP_KEY[MixedDatasets]="mnist_mixed_datasets"
BENCH_CONFIG_MAP_KEY[MislabelingDetection]="mnist_mislabeling_detection"
BENCH_CONFIG_MAP_KEY[ShortcutDetection]="mnist_shortcut_detection"
BENCH_CONFIG_MAP_KEY[LDS]="mnist_linear_datamodeling"

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
        python scripts/generate_config.py hydra.run.dir="hydra_logs" $params id=$id +cfg_file_name=$id +cfg_output_dir=$cfg_output_dir
        python scripts/train.py bench="$bench" $params $sweep id=$id +cfg_output_dir=$cfg_output_dir +cfg_file_name=$id --multirun
        python scripts/opt_results_to_cfg.py bench="$bench" $params id=$id +cfg_output_dir=$cfg_output_dir +cfg_file_name=$id
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
