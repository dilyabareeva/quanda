set -euo pipefail

jid1=$(sbatch --parsable slurm/slurm_job.sbatch scripts/awa2_resnet50_bench/eval_awa2_pt1.sh)
[[ -n $jid1 ]] || { echo "pt1 submission failed"; exit 1; }

sbatch --dependency=afterok:$jid1 slurm/slurm_job.sbatch scripts/awa2_resnet50_bench/eval_awa2_pt2.sh

