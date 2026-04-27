#!/bin/bash
# Train and push the M subset models for an LDS benchmark. Subset training
# fans out up to N_LDS_PARALLEL workers.
#
# Required from the caller's env:
#   CONFIG_MAP_PREFIX   key prefix in benchmarks/resources/config_map.py;
#                       used to resolve the registered LDS config id.

export PYTHONPATH="$PYTHONPATH:$(dirname $(dirname $(realpath $0)))"

# ---------- defaults ----------
PARALLEL=true
N_LDS_PARALLEL=1
HF_PUSH_SLEEP=60
PUSH_ONLY=false
GPU_SPLIT=false
SUBSET_START=""
SUBSET_END=""
SUBSET_INDICES=""

# ---------- CLI ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)         PARALLEL=$2;       shift 2 ;;
        --n-lds-parallel)   N_LDS_PARALLEL=$2; shift 2 ;;
        --hf-push-sleep)    HF_PUSH_SLEEP=$2;  shift 2 ;;
        --push-only)        PUSH_ONLY=$2;      shift 2 ;;
        --gpu-split)        GPU_SPLIT=$2;      shift 2 ;;
        --start)            SUBSET_START=$2;   shift 2 ;;
        --end)              SUBSET_END=$2;     shift 2 ;;
        --indices)          SUBSET_INDICES=$2; shift 2 ;;
        *) shift ;;
    esac
done

# Resolve the list of subset indices to process
resolve_indices() {
    local m=$1
    if [ -n "$SUBSET_INDICES" ]; then
        echo "$SUBSET_INDICES" | tr ',' ' '
    else
        local start=${SUBSET_START:-0}
        local end=${SUBSET_END:-$((m - 1))}
        (( end > m - 1 )) && end=$((m - 1))
        seq "$start" "$end"
    fi
}

# ---------- paths ----------
CFG_DIR="quanda/benchmarks/resources/configs"
mkdir -p logs

if [ -d "/data/cluster/users/bareeva" ]; then
    BENCH_SAVE_DIR="/data/cluster/users/bareeva/quanda_output_new2/eval_bench/${CONFIG_MAP_PREFIX}"
else
    BENCH_SAVE_DIR="/data2/bareeva/Projects/quanda/cluster_output_new2/eval_bench/${CONFIG_MAP_PREFIX}"
fi
SAVE_OVERRIDE="bench_save_dir=${BENCH_SAVE_DIR}"

# ---------- helpers ----------

# Look up a registered benchmark config name from config_map.py.
config_name_from_map() {
    python -c "
from quanda.benchmarks.resources.config_map import config_map
import os
print(os.path.splitext(os.path.basename(str(config_map['$1'])))[0])
"
}

# Read a top-level field from a generated config yaml.
yaml_field() {
    python -c "import yaml; print(yaml.safe_load(open('$1'))['$2'])"
}

# Cache metadata
hydrate_metadata() {
    python -c "
from quanda.benchmarks.ground_truth import LinearDatamodeling
LinearDatamodeling.load_pretrained(
    bench_id='${CFG_DIR}/$1.yaml',
    cache_dir='${BENCH_SAVE_DIR}',
    offline=False,
)
"
}

# Train one LDS subset model. Optionally pin to a GPU based on idx parity.
train_subset() {
    local id=$1 idx=$2
    local gpu_env=() device_args=()
    if [ "$GPU_SPLIT" = true ]; then
        gpu_env=(env "CUDA_VISIBLE_DEVICES=$((idx % 2))")
        device_args=(--device "cuda:0")
    fi
    "${gpu_env[@]}" python scripts/train_lds_subset.py \
        --config-path "${CFG_DIR}/${id}.yaml" --idx "$idx" \
        --bench-save-dir "$BENCH_SAVE_DIR" "${device_args[@]}"
}

# Push one already-trained subset checkpoint to HF Hub.
push_subset() {
    python scripts/train_lds_subset.py \
        --config-path "${CFG_DIR}/$1.yaml" --idx "$2" --push-only \
        --bench-save-dir "$BENCH_SAVE_DIR"
}


# ---------- run id ----------
ID=$(config_name_from_map "${CONFIG_MAP_PREFIX}_linear_datamodeling")

# ---------- main ----------
run_lds() {
    # 1. Read M and the subset ckpt basename from the resolved config.
    local cfg="${CFG_DIR}/${ID}.yaml"
    local M ckpt_base
    M=$(yaml_field "$cfg" "m")
    ckpt_base=$(python -c "import yaml, os; print(os.path.basename(yaml.safe_load(open('$cfg'))['subset_ckpt']))")

    # 2. Sync HF Hub metadata into the local cache.
    hydrate_metadata "$ID"

    # 3. Decide which subset indices to process.
    local index_list
    index_list=$(resolve_indices "$M")

    # 4. Train each subset (fan-out up to N_LDS_PARALLEL).
    local log_dir="logs/${ID}"
    mkdir -p "$log_dir"
    if [ "$PUSH_ONLY" = false ]; then
        for i in $index_list; do
            local log="${log_dir}/subset_${i}.log"
            if [ "$PARALLEL" = true ]; then
                while [ "$(jobs -rp | wc -l)" -ge "$N_LDS_PARALLEL" ]; do
                    wait -n
                done
                train_subset "$ID" "$i" > "$log" 2>&1 &
            else
                train_subset "$ID" "$i" > "$log" 2>&1
            fi
        done
        wait
    fi

    # 5. Push each existing subset to HF Hub; warn about any missing.
    local missing=()
    for i in $index_list; do
        local subset_dir="${BENCH_SAVE_DIR}/ckpt/${ckpt_base}_lds_subset_${i}"
        echo "Pushing subset $i from $subset_dir to HF Hub..."
        if [ ! -d "$subset_dir" ]; then
            missing+=("$i")
            continue
        fi
        push_subset "$ID" "$i"
        sleep "$HF_PUSH_SLEEP"
    done
    if (( ${#missing[@]} > 0 )); then
        echo "WARNING: skipped push for missing subset ckpts: ${missing[*]}" >&2
    fi
}

run_lds > "logs/${ID}.log" 2>&1 < /dev/null
