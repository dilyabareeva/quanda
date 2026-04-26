#!/bin/bash

methods=(
    random
    kronfluence_gpt2
    dattri_if_datainf
    dattri_trak
    similarity
)

for method in "${methods[@]}"; do
    sbatch slurm/slurm_job.sbatch \
        scripts/gpt2_trex_bench/eval_mrr.sh \
        --method "$method"
done
