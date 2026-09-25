#!/bin/bash

cd "$(dirname "$0")/.."
mkdir -p log

declare -A PART=([mnist_lenet]=gpu1,gpu2 [cifar_resnet9]=gpu1,gpu2 [awa2_resnet50]=gpu3,gpu4 [bert_qnli]=gpu3,gpu4 [gpt2_trex]=gpu5)


eval() { sbatch --job-name="$3__$4" --partition="${PART[$2]}" slurm/slurm_job.sbatch scripts/eval_one.sh "$@" "${ARGS[@]}"; }

ARGS=("$@")

# MNIST/CIFAR STEP 1 Eval
for m in similarity representer_points tracincpfast arnoldi trak random; do
    for b in class_detection subclass_detection shortcut_detection mixed_datasets; do
        eval mnsit_lenet_bench   mnist_lenet   mnist_$b $m
        eval cifar_resnet9_bench cifar_resnet9 cifar_$b $m
    done
done

# MNIST/CIFAR STEP 2 Eval
for m in similarity representer_points tracincpfast arnoldi trak random; do
    for b in top_k_cardinality model_randomization mislabeling_detection linear_datamodeling; do
        eval mnsit_lenet_bench   mnist_lenet   mnist_$b $m
        eval cifar_resnet9_bench cifar_resnet9 cifar_$b $m
    done
done

# MNIST/CIFAR LDS ALPHA Eval
for m in similarity representer_points tracincpfast arnoldi trak random; do
    for b in alpha075_linear_datamodeling alpha09_linear_datamodeling alpha095_linear_datamodeling alpha0999_linear_datamodeling; do
        eval mnsit_lenet_bench   mnist_lenet   mnist_$b $m
        eval cifar_resnet9_bench cifar_resnet9 cifar_$b $m
    done
done


# AwA2 STEP 1 Eval
for m in similarity representer_points tracincpfast arnoldi trak random kronfluence; do
    for b in class_detection subclass_detection shortcut_detection mixed_datasets; do
        eval awa2_resnet50_bench awa2_resnet50 awa2_$b  $m
    done
done

# AwA2 STEP 2 Eval
for m in similarity representer_points tracincpfast arnoldi trak random kronfluence; do
    for b in top_k_cardinality model_randomization mislabeling_detection linear_datamodeling; do
        eval awa2_resnet50_bench awa2_resnet50 awa2_$b  $m
    done
done


# AwA2 LDS ALPHA Eval
for m in similarity representer_points tracincpfast arnoldi trak random kronfluence; do
    eval awa2_resnet50_bench awa2_resnet50 awa2_alpha075_linear_datamodeling  $m
done



# BERT STEP 1 Eval
for m in similarity kronfluence trak random representer_points; do
    for b in class_detection mislabeling_detection mixed_datasets; do
        eval bert_qnli_bench bert_qnli qnli_$b $m
    done
done


# BERT STEP 2 Eval
for m in similarity kronfluence trak random representer_points; do
    for b in top_k_cardinality model_randomization linear_datamodeling; do
        eval bert_qnli_bench bert_qnli qnli_$b $m
    done
done

# BERT LDS ALPHA Eval
for m in similarity kronfluence trak random representer_points; do
    eval bert_qnli_bench bert_qnli qnli_alpha075_linear_datamodeling  $m
done


