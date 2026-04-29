#!/bin/bash
# Fan out one sbatch job per benchmark; each job runs the full sweep for
# that benchmark with hydra n_jobs sized to its grid cardinality.

BENCHMARKS=(
    ClassDetection
    SubclassDetection
    MixedDatasets
    ShortcutDetection
    MislabelingDetection
    LDS
)

for bench in "${BENCHMARKS[@]}"; do
    sbatch slurm/slurm_job.sbatch \
        scripts/awa2_resnet50_bench/train_awa2_per_bench.sh \
        "$bench"
done
