#!/bin/bash

DIR="$(dirname "$0")"

bash "$DIR/mnsit_lenet_bench/plot_mnist.sh"
bash "$DIR/cifar_resnet9_bench/plot_cifar.sh"
bash "$DIR/bert_qnli_bench/plot_qnli.sh"
bash "$DIR/gpt2_trex_bench/plot_gpt2.sh"

