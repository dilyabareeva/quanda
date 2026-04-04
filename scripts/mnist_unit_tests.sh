#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(dirname $(dirname $(realpath $0)))"

bench_types=(
    "LDS"
    "MislabelingDetection"
    "ClassDetection"
    "SubclassDetection"
    "ShortcutDetection"
    "MixedDatasets"
)
bench_params=(
    "train_dataset=mnist_train train_dataset.dataset_split='train[:1%]' eval_dataset=mnist_test eval_dataset.dataset_split='test[:1%]'"
    "train_dataset=mnist_train_mislabeling train_dataset.dataset_split='train[:1%]' eval_dataset=mnist_test eval_dataset.dataset_split='test[:1%]'"
    "train_dataset=mnist_train train_dataset.dataset_split='train[:1%]' eval_dataset=mnist_test eval_dataset.dataset_split='test[:1%]'"
    "model=mnist_lenet_subclass train_dataset=mnist_train_subclass train_dataset.dataset_split='train[:1%]' eval_dataset=mnist_test_subclass eval_dataset.dataset_split='test[:1%]'"
    "train_dataset=mnist_train_shortcut train_dataset.dataset_split='train[:1%]' eval_dataset=mnist_test_shortcut eval_dataset.dataset_split='test[:1%]'"
    "train_dataset=mnist_train train_dataset.dataset_split='train[:1%]' eval_dataset=mnist_test eval_dataset.dataset_split='test[:1%]' +adv_dataset=fashion_mnist_unit"
)

# Define the output directory
cfg_output_dir="tests/assets/mnist_local_bench"

# Get the current git commit tag
commit_tag=$(git rev-parse --short HEAD)

# Iterate over each parameter dictionary
for i in "${!bench_types[@]}"; do
    bench="${bench_types[$i]}"
    params="${bench_params[$i]}"

    # Construct the output file name
    id="${commit_tag}-default_${bench}"
    cfg_file_name="${id}"

    # Construct and execute the command with Hydra overrides
    echo "Bench type: $bench"
    echo "Config file name: $cfg_file_name"
    echo "Running with parameters: $params"
    echo "Saving output to: $cfg_output_dir/$cfg_file_name"
    python scripts/generate_config.py hydra.run.dir="hydra_logs" $params id=$id +cfg_file_name=$cfg_file_name +cfg_output_dir=$cfg_output_dir
    # Hyperparameter sweep
    python scripts/train.py bench="$bench" $params id=$id +cfg_output_dir=$cfg_output_dir +cfg_file_name=$cfg_file_name --multirun
    # Saving the results to a config file
    python scripts/opt_results_to_cfg.py bench="$bench" $params id=$id +cfg_output_dir=$cfg_output_dir +cfg_file_name=$cfg_file_name
    # Training the model
    #python scripts/train_and_push_to_hub.py --config-name $cfg_file_name --config-dir $cfg_output_dir
    python scripts/train.py --config-name $cfg_file_name --config-dir $cfg_output_dir
    echo "Finished running with parameters: $params"
    echo "--------------------------------------"
done 