#!/bin/bash

# Define the parameter dictionaries
param_dicts=(
    "bench=ClassDetection train_dataset=mnist_train_unit eval_dataset=mnist_test_unit"
    "bench=MislabelingDetection model=mnist_lenet_mislabeling train_dataset=mnist_train_unit_mislabeling"
    "bench=SubclassDetection model=mnist_lenet_subclass train_dataset=mnist_train_unit_subclass eval_dataset=mnist_test_unit_subclass"
    "bench=ShortcutDetection train_dataset=mnist_train_unit_shortcut eval_dataset=mnist_test_unit_shortcut"
    "bench=MixedDatasets train_dataset=mnist_train_unit eval_dataset=mnist_test_unit +adv_dataset=fashion_mnist_unit"
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
    cfg_file_name="${commit_tag}-default"

    # Construct and execute the command with Hydra overrides
    echo "Running with parameters: $params"
    echo "Saving output to: $cfg_output_dir/$cfg_file_name"
    python bench_prep/run.py $params hydra.run.dir="hydra_logs" +cfg_file_name=$cfg_file_name +cfg_output_dir=$cfg_output_dir
    echo "Finished running with parameters: $params"
    echo "--------------------------------------"
done 