#!/bin/bash
# Chain three fan-outs with SLURM dependencies:
#   1. per-bench training (one job per benchmark)
#   2. LDS subset training (M/STRIDE jobs)  — runs after stage 1 succeeds
#   3. LDS subset logit computation         — runs after stage 2 succeeds
# Any failure in a stage cancels the dependent stages via --kill-on-invalid-dep.

set -euo pipefail

M=100
STRIDE=10

# ---- Stage 1: per-bench training -------------------------------------------
BENCHMARKS=(
    ClassDetection
    SubclassDetection
    MixedDatasets
    ShortcutDetection
    MislabelingDetection
    LDS
)

stage1_ids=()
for bench in "${BENCHMARKS[@]}"; do
    jid=$(sbatch --parsable slurm/slurm_job.sbatch \
        scripts/awa2_resnet50_bench/train_awa2_per_bench.sh \
        "$bench")
    stage1_ids+=("$jid")
done
dep1=$(IFS=:; echo "${stage1_ids[*]}")

# ---- Stage 2: LDS subset training ------------------------------------------
# train_awa2_lds.sh treats --end as INCLUSIVE; pass STRIDE-1 to avoid
# overlapping boundary indices across concurrent jobs.
stage2_ids=()
for start in $(seq 0 "$STRIDE" "$((M - STRIDE))"); do
    end=$((start + STRIDE - 1))
    jid=$(sbatch --parsable \
        --dependency=afterok:"$dep1" --kill-on-invalid-dep=yes \
        slurm/slurm_job.sbatch \
        scripts/awa2_resnet50_bench/train_awa2_lds.sh \
        --start "$start" --end "$end")
    stage2_ids+=("$jid")
done
dep2=$(IFS=:; echo "${stage2_ids[*]}")

# ---- Stage 3: compute LDS subset logits ------------------------------------
BATCH_SIZE=64
MAX_EVAL_N=1000
EVAL_SEED=42
INFERENCE_BATCH_SIZE=64
DEVICE=cuda:0

stage3_ids=()
for start in $(seq 0 "$STRIDE" "$((M - STRIDE))"); do
    end=$((start + STRIDE))
    jid=$(sbatch --parsable \
        --dependency=afterok:"$dep2" --kill-on-invalid-dep=yes \
        slurm/slurm_job.sbatch \
        scripts/awa2_resnet50_bench/compute_lds_subset_logits_awa2.sh \
        --start "$start" --end "$end" \
        --batch-size "$BATCH_SIZE" \
        --max-eval-n "$MAX_EVAL_N" \
        --eval-seed "$EVAL_SEED" \
        --inference-batch-size "$INFERENCE_BATCH_SIZE" \
        --device "$DEVICE")
    stage3_ids+=("$jid")
done

echo "Stage 1 (per-bench):     ${stage1_ids[*]}"
echo "Stage 2 (LDS train):     ${stage2_ids[*]}"
echo "Stage 3 (LDS logits):    ${stage3_ids[*]}"
