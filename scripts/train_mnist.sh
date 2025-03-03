#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(dirname $(dirname $(realpath $0)))"

bench_types=(
    #"MislabelingDetection"
    "ClassDetection"
    # "SubclassDetection"
    # "ShortcutDetection"
    # "MixedDatasets"
)
bench_params=(
    #"train_dataset=mnist_train_unit_mislabeling"
    "train_dataset=mnist_train_unit eval_dataset=mnist_test_unit"
    # "model=mnist_lenet_subclass train_dataset=mnist_train_unit_subclass eval_dataset=mnist_test_unit_subclass"
    # "train_dataset=mnist_train_unit_shortcut eval_dataset=mnist_test_unit_shortcut"
    # "train_dataset=mnist_train_unit eval_dataset=mnist_test_unit +adv_dataset=fashion_mnist_unit"
)
# Define the output directory
cfg_output_dir="quanda/benchmarks/resources/configs"

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
    echo "Running with parameters: $params"
    echo "Saving output to: $cfg_output_dir/$cfg_file_name"
    # Hyperparameter sweep
    #python scripts/train.py bench="$bench" $params id=$id +cfg_output_dir=$cfg_output_dir +cfg_file_name=$cfg_file_name --multirun
    # Saving the results to a config file
    #python scripts/opt_results_to_cfg.py bench="$bench" $params id=$id +cfg_output_dir=$cfg_output_dir +cfg_file_name=$cfg_file_name
    # Training the model
    python scripts/train_and_push_to_hub.py --config-name $cfg_file_name
    echo "Finished running with parameters: $params"
    echo "--------------------------------------"
done 