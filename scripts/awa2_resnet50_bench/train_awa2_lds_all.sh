#!/bin/bash

M=100
STRIDE=10

for start in $(seq 0 "$STRIDE" "$((M - STRIDE))"); do
    end=$((start + STRIDE))
    sbatch slurm/slurm_job.sbatch \
        scripts/awa2_resnet50_bench/train_awa2_lds.sh \
        --start "$start" --end "$end"
done
