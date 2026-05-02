#!/bin/bash
set -euo pipefail

# Worker mode: run a single (method, benchmark) pair.
if [ "${1:-}" = "--run" ]; then
    source "$(dirname "$0")/eval_defs.sh"
    EVAL_CONFIG_NAME="awa2_resnet50"
    PARALLEL=false
    methods=("$2")
    benchmarks=("$3")
    source "$(dirname "$0")/../eval.sh"
    exit
fi

# Submitter mode: one sbatch job per (method, benchmark);
# every pt2 job waits on all pt1 jobs.
methods=(
    arnoldi
)

bench_pt1=(
    awa2_class_detection
    #awa2_subclass_detection
    #awa2_shortcut_detection
    #awa2_mixed_datasets
    #awa2_mislabeling_detection
)

bench_pt2=(
    awa2_linear_datamodeling
    awa2_top_k_cardinality
    awa2_model_randomization
)

pt1_jids=()
for method in "${methods[@]}"; do
    for bench in "${bench_pt1[@]}"; do
        jid=$(sbatch --parsable slurm/slurm_job.sbatch "$0" --run "$method" "$bench")
        [[ -n $jid ]] || { echo "$method $bench pt1 submission failed"; exit 1; }
        pt1_jids+=("$jid")
    done
done

dep=$(IFS=:; echo "${pt1_jids[*]}")
for method in "${methods[@]}"; do
    for bench in "${bench_pt2[@]}"; do
        sbatch --dependency=afterok:$dep slurm/slurm_job.sbatch "$0" --run "$method" "$bench"
    done
done
