#!/bin/bash

# Define the parameter dictionaries
param_dicts=(
    "train_dataset=mnist_train_unit"
)

# Define the output directory
cfg_output_dir="tests/assets/mnist_test_suite_2"

# Get the current git commit tag
commit_tag=$(git rev-parse --short HEAD)

# Iterate over each parameter dictionary
for params in "${param_dicts[@]}"; do
    # Replace spaces with underscores in the params string
    params_underscored=$(echo $params | tr ' ' '_' | tr '=' '--')

    # Construct the output file name
    cfg_file_name="${commit_tag}-default-${params_underscored}.yaml"

    # Construct and execute the command with Hydra overrides
    echo "Running with parameters: $params"
    echo "Saving output to: $cfg_output_dir/$cfg_file_name"
    python bench_prep/run.py $params hydra.run.dir="hydra_logs" +cfg_file_name=$cfg_file_name +cfg_output_dir=$cfg_output_dir
    echo "Finished running with parameters: $params"
    echo "--------------------------------------"
done 