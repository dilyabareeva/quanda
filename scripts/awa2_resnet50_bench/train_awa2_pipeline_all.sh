#!/bin/bash
# Chain two fan-outs with SLURM dependencies:
#   1. LDS subset training (M/STRIDE jobs)
#   2. LDS subset logit computation         — runs after stage 1 succeeds
# Any failure in a stage cancels the dependent stages via --kill-on-invalid-dep.

set -euo pipefail

M=100
STRIDE=1

# ---- Stage 1: LDS subset training ------------------------------------------
stage2_ids=()
for start in $(seq 0 "$STRIDE" "$((M - STRIDE))"); do
    end=$((start + STRIDE - 1))
    jid=$(sbatch --parsable \
        slurm/slurm_job.sbatch \
        scripts/awa2_resnet50_bench/train_awa2_lds.sh \
        --start "$start" --end "$end")
    stage2_ids+=("$jid")
done
dep2=$(IFS=:; echo "${stage2_ids[*]}")

# ---- Stage 2: compute LDS subset logits ------------------------------------
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

echo "Stage 1 (LDS train):     ${stage2_ids[*]}"
echo "Stage 2 (LDS logits):    ${stage3_ids[*]}"
