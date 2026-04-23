#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(dirname $(dirname $(realpath $0)))"

PARALLEL=true
N_LDS_PARALLEL=16
HF_PUSH_SLEEP=60
TRAIN_ONLY=false
SKIP_MAIN_TRAIN=false
PUSH_ONLY=false
GPU_SPLIT=false
SUBSET_START=""
SUBSET_END=""
SUBSET_INDICES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel) PARALLEL=$2; shift 2 ;;
        --n-lds-parallel) N_LDS_PARALLEL=$2; shift 2 ;;
        --hf-push-sleep) HF_PUSH_SLEEP=$2; shift 2 ;;
        --train-only) TRAIN_ONLY=$2; shift 2 ;;
        --skip-main-train) SKIP_MAIN_TRAIN=$2; shift 2 ;;
        --push-only) PUSH_ONLY=$2; shift 2 ;;
        --gpu-split) GPU_SPLIT=$2; shift 2 ;;
        --start) SUBSET_START=$2; shift 2 ;;
        --end) SUBSET_END=$2; shift 2 ;;
        --indices) SUBSET_INDICES=$2; shift 2 ;;
        *) shift ;;
    esac
done

cfg_output_dir="quanda/benchmarks/resources/configs"
commit_tag=$(GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git rev-parse --short HEAD 2>/dev/null || echo "nogit")
mkdir -p logs

if [ -d "/data/cluster/users/bareeva" ]; then
    bench_save_dir_override="bench_save_dir=/data/cluster/users/bareeva/quanda_output_new/eval_bench/${CONFIG_MAP_PREFIX}"
else
    bench_save_dir_override="bench_save_dir=/data2/bareeva/Projects/quanda/cluster_output_new/eval_bench/${CONFIG_MAP_PREFIX}"
fi

declare -A BENCH_CONFIG_MAP_KEY
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
    if [ "$PUSH_ONLY" = true ]; then
        :
    elif [ "$SKIP_MAIN_TRAIN" = true ]; then
        python scripts/train_and_push_to_hub.py --config-name "$id" --config-dir $cfg_output_dir +skip_subsets=true +skip_main_train=true $bench_save_dir_override
    elif [ "$TRAIN_ONLY" = false ]; then
        python scripts/generate_config.py --config-name "$CONFIG_NAME" hydra.run.dir="hydra_logs" bench=LDS $params id=$id +cfg_file_name=$id +cfg_output_dir=$cfg_output_dir $bench_save_dir_override
        python scripts/train.py --config-name "$CONFIG_NAME" bench=LDS $params $sweep m=1 id=$id +cfg_output_dir=$cfg_output_dir +cfg_file_name=$id num_checkpoints=1 $bench_save_dir_override --multirun
        python scripts/opt_results_to_cfg.py --config-name "$CONFIG_NAME" bench=LDS $params id=$id +cfg_output_dir=$cfg_output_dir +cfg_file_name=$id $bench_save_dir_override
        python scripts/train_and_push_to_hub.py --config-name $id --config-dir $cfg_output_dir +skip_subsets=true $bench_save_dir_override
    else
        python scripts/train_and_push_to_hub.py --config-name "$id" --config-dir $cfg_output_dir +skip_subsets=true $bench_save_dir_override
    fi

    local M bench_save_dir
    M=$(python -c "import yaml; print(yaml.safe_load(open('${cfg_output_dir}/${id}.yaml'))['m'])")
    bench_save_dir="${bench_save_dir_override#bench_save_dir=}"
    ckpt_basename=$(python -c "import yaml, os; print(os.path.basename(yaml.safe_load(open('${cfg_output_dir}/${id}.yaml'))['subset_ckpt']))")

    # Hydrate non-pid metadata dir from HF Hub so parallel workers and
    # push_subset find subset_ids without hitting the pid-suffixed path.
    python -c "
from quanda.benchmarks.ground_truth import LinearDatamodeling
LinearDatamodeling.load_pretrained(
    bench_id='${cfg_output_dir}/${id}.yaml',
    cache_dir='${bench_save_dir}',
    offline=False,
)
"

    mkdir -p "logs/${id}"

    local index_list
    if [ -n "$SUBSET_INDICES" ]; then
        index_list=$(echo "$SUBSET_INDICES" | tr ',' ' ')
    else
        local start end
        start=${SUBSET_START:-0}
        end=${SUBSET_END:-$((M - 1))}
        if [ "$end" -gt "$((M - 1))" ]; then end=$((M - 1)); fi
        index_list=$(seq "$start" "$end")
    fi

    if [ "$PUSH_ONLY" = false ]; then
        for i in $index_list; do
            gpu_env=()
            device_args=()
            if [ "$GPU_SPLIT" = true ]; then
                gpu_env=(env "CUDA_VISIBLE_DEVICES=$((i % 2))")
                device_args=(--device "cuda:0")
            fi
            if [ "$PARALLEL" = true ]; then
                while [ "$(jobs -rp | wc -l)" -ge "$N_LDS_PARALLEL" ]; do
                    wait -n
                done
                "${gpu_env[@]}" python scripts/train_lds_subset.py \
                    --config-path "${cfg_output_dir}/${id}.yaml" --idx "$i" \
                    "${device_args[@]}" \
                    > "logs/${id}/subset_${i}.log" 2>&1 &
            else
                "${gpu_env[@]}" python scripts/train_lds_subset.py \
                    --config-path "${cfg_output_dir}/${id}.yaml" --idx "$i" \
                    "${device_args[@]}" \
                    > "logs/${id}/subset_${i}.log" 2>&1
            fi
        done
        wait
    fi

    missing_subsets=()
    for i in $index_list; do
        subset_dir="${bench_save_dir}/ckpt/${ckpt_basename}_lds_subset_${i}"
        echo "Pushing subset $i from $subset_dir to HF Hub..."
        if [ ! -d "$subset_dir" ]; then
            missing_subsets+=("$i")
            continue
        fi
        python scripts/train_lds_subset.py \
            --config-path "${cfg_output_dir}/${id}.yaml" \
            --idx "$i" --push-only
        sleep "$HF_PUSH_SLEEP"
    done
    if [ "${#missing_subsets[@]}" -gt 0 ]; then
        echo "WARNING: skipped push for missing subset ckpts: ${missing_subsets[*]}" >&2
    fi
}

bench="LDS"
params="${BENCH_PARAMS[$bench]}"
sweep="${BENCH_SWEEP[$bench]}"
if [ "$SKIP_MAIN_TRAIN" = true ] || [ "$TRAIN_ONLY" = true ] || [ "$PUSH_ONLY" = true ]; then
    id=$(get_config_name_from_map "${BENCH_CONFIG_MAP_KEY[$bench]}")
else
    id="${commit_tag}-${CONFIG_NAME}_${bench}"
fi
run_bench "$bench" "$params" "$sweep" "$id" > "logs/${id}.log" 2>&1 < /dev/null
