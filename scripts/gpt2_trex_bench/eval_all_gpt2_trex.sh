#!/bin/bash
set -euo pipefail

methods=(
    random
    kronfluence_gpt2
    dattri_if_datainf
    dattri_trak
    similarity
)

mrr_jids=()
for method in "${methods[@]}"; do
    jid=$(sbatch --parsable slurm/slurm_job.sbatch \
        scripts/gpt2_trex_bench/eval_mrr.sh \
        --method "$method")
    [[ -n $jid ]] || { echo "mrr submission failed for $method"; exit 1; }
    mrr_jids+=("$jid")
done

for i in "${!methods[@]}"; do
    method="${methods[$i]}"
    jid="${mrr_jids[$i]}"

    sbatch --dependency=afterok:$jid slurm/slurm_job.sbatch \
        scripts/gpt2_trex_bench/eval_recall_at_k.sh \
        --method "$method"

    sbatch --dependency=afterok:$jid slurm/slurm_job.sbatch \
        scripts/gpt2_trex_bench/eval_tail_patch.sh \
        --method "$method"
done
