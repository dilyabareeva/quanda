#!/bin/bash
# The following script is not meant to be run and only serves as a representation of the sequence of steps performed to train and evaluate all benchmarks.
set -e

DIR="$(dirname "$0")"

# 1) Train benchmarks (with hyperparam sweep)
bash "$DIR/mnsit_lenet_bench/train_mnist.sh"
bash "$DIR/cifar_resnet9_bench/train_cifar.sh"
bash "$DIR/awa2_resnet50_bench/train_awa2.sh"
bash "$DIR/bert_qnli_bench/train_qnli.sh"

# 2) Train LDS models
bash "$DIR/mnsit_lenet_bench/train_mnist_lds.sh"
bash "$DIR/cifar_resnet9_bench/train_cifar_lds.sh"
bash "$DIR/awa2_resnet50_bench/train_awa2_lds.sh"
bash "$DIR/bert_qnli_bench/train_qnli_lds.sh"

# 3) Collect LDS submodel logits
bash "$DIR/awa2_resnet50_bench/compute_lds_subset_logits_awa2.sh" \
    --start 0 --end 100 \
    --batch-size 64 --max-eval-n 1000 --eval-seed 42 \
    --inference-batch-size 64 --device cuda:0
bash "$DIR/bert_qnli_bench/compute_lds_subset_logits_qnli.sh" \
    --start 0 --end 100 \
    --batch-size 8 --max-eval-n 1000 --eval-seed 42 \
    --inference-batch-size 32 --device cuda:0

# 4) Run eval
bash "$DIR/mnsit_lenet_bench/eval_mnist_pt1.sh"
bash "$DIR/mnsit_lenet_bench/eval_mnist_pt2.sh"
bash "$DIR/cifar_resnet9_bench/eval_cifar_pt1.sh"
bash "$DIR/cifar_resnet9_bench/eval_cifar_pt2.sh"
bash "$DIR/awa2_resnet50_bench/eval_awa2_pt1.sh"
bash "$DIR/awa2_resnet50_bench/eval_awa2_pt2.sh"
bash "$DIR/bert_qnli_bench/eval_qnli.sh"
bash "$DIR/gpt2_trex_bench/eval_mrr.sh"
bash "$DIR/gpt2_trex_bench/eval_recall_at_k.sh"
bash "$DIR/gpt2_trex_bench/eval_tail_patch.sh"
