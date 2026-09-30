#!/bin/bash

cd "$(dirname "$0")/.."
mkdir -p log

declare -A PART=([mnist_lenet]=gpu1,gpu2 [cifar_resnet9]=gpu1,gpu2 [awa2_resnet50]=gpu3,gpu4 [bert_qnli]=gpu3,gpu4 [gpt2_trex]=gpu5)


eval() { sbatch --job-name="$3__$4" --partition="${PART[$2]}" slurm/slurm_job.sbatch scripts/eval_one.sh "$@" "${ARGS[@]}"; }

ARGS=("$@")


# AwA2 Mislabeling Detection Percentages
for m in similarity representer_points tracincpfast arnoldi trak random kronfluence; do
    for b in mislabeling_detection_p10; do
        eval awa2_resnet50_bench awa2_resnet50 awa2_$b $m
    done
done


