#!/bin/bash

DIR="$(dirname "$0")"

# standard settings
python "$DIR/plot_results.py" --results_config "$DIR/mnsit_lenet_bench/mnist_plot_config.json"
python "$DIR/plot_results.py" --results_config "$DIR/cifar_resnet9_bench/cifar_plot_config.json"
python "$DIR/plot_results.py" --results_config "$DIR/bert_qnli_bench/qnli_plot_config.json"
python "$DIR/plot_results.py" --results_config "$DIR/gpt2_trex_bench/gpt2_plot_config.json"
python "$DIR/plot_results.py" --results_config "$DIR/awa2_resnet50_bench/awa2_plot_config.json"

# LDS alpha settings
python "$DIR/plot_results.py" --results_config "$DIR/mnsit_lenet_bench/mnist_lds_alpha_plot_config.json"
python "$DIR/plot_results.py" --results_config "$DIR/cifar_resnet9_bench/cifar_lds_alpha_plot_config.json"
python "$DIR/plot_results.py" --results_config "$DIR/bert_qnli_bench/qnli_lds_alpha_plot_config.json"
python "$DIR/plot_results.py" --results_config "$DIR/awa2_resnet50_bench/awa2_lds_alpha_plot_config.json"

# Mislabeling Detection Percentages
python "$DIR/plot_results.py" --results_config "$DIR/mnsit_lenet_bench/mnist_mislabel_p_plot_config.json"
python "$DIR/plot_results.py" --results_config "$DIR/cifar_resnet9_bench/cifar_mislabel_p_plot_config.json"
python "$DIR/plot_results.py" --results_config "$DIR/awa2_resnet50_bench/awa2_mislabel_p_plot_config.json"

python "$DIR/plot_eval_sweeps.py"
