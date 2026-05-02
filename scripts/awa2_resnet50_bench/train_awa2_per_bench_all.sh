#!/bin/bash

BENCHMARKS=(
    ClassDetection
    SubclassDetection
    MixedDatasets
    ShortcutDetection
    MislabelingDetection
)

for bench in "${BENCHMARKS[@]}"; do
    sbatch slurm/slurm_job.sbatch \
        scripts/awa2_resnet50_bench/train_awa2_per_bench.sh \
        "$bench"
done
 