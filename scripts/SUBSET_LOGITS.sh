#!/bin/bash

cd "$(dirname "$0")/.."
mkdir -p log

declare -A BS=([mnist_lenet]="32" [cifar_resnet9]="32" [awa2_resnet50]="16 128 256 1000" [bert_qnli]="8 32 1000")
declare -A INF_BS=([awa2_resnet50]=64 [bert_qnli]=32)

logits() {
    local sizes=("${@:4}"); [ ${#sizes[@]} -eq 0 ] && sizes=(${BS[$3]})
    for bs in "${sizes[@]}"; do
        sbatch --job-name="$3__logits_b$bs" --partition="${PART[$2]}" slurm/slurm_job.sbatch scripts/compute_lds_subset_logits.sh "${@:1:3}" --config-map-key "$3" --start 0 --end 100 --batch-size "$bs" --max-eval-n 1000 --eval-seed 42 --inference-batch-size "${INF_BS[$2]}" --device cuda:0 "${ARGS[@]}"
    done
}

ARGS=("$@")


logits mnsit_lenet_bench   mnist_lenet   mnist_linear_datamodeling
logits cifar_resnet9_bench cifar_resnet9 cifar_linear_datamodeling
logits awa2_resnet50_bench awa2_resnet50 awa2_linear_datamodeling
for a in alpha075 alpha09 alpha095 alpha0999; do
    logits mnsit_lenet_bench   mnist_lenet   mnist_${a}_linear_datamodeling
    logits cifar_resnet9_bench cifar_resnet9 cifar_${a}_linear_datamodeling
done
logits awa2_resnet50_bench awa2_resnet50 awa2_alpha075_linear_datamodeling 16
logits bert_qnli_bench     bert_qnli     qnli_alpha075_linear_datamodeling 32
logits bert_qnli_bench     bert_qnli     qnli_linear_datamodeling