#!/bin/bash

M=100
STRIDE=10

BATCH_SIZE=64
MAX_EVAL_N=1000
EVAL_SEED=42
INFERENCE_BATCH_SIZE=32
DEVICE=cuda:0

for start in $(seq 0 "$STRIDE" "$((M - STRIDE))"); do
    end=$((start + STRIDE))
    sbatch slurm/slurm_job.sbatch \
        scripts/bert_qnli_bench/precompute_qnli_subset_logits.sh \
        --start "$start" --end "$end" \
        --batch-size "$BATCH_SIZE" \
        --max-eval-n "$MAX_EVAL_N" \
        --eval-seed "$EVAL_SEED" \
        --inference-batch-size "$INFERENCE_BATCH_SIZE" \
        --device "$DEVICE"
done
