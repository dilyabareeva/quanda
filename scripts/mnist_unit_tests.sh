#!/bin/bash

# Define the parameter dictionaries
param_dicts=(
    "bench=ClassDetection train_dataset=mnist_train_unit train_dataset.dataset_split='train[:1%]' eval_dataset=mnist_test_unit eval_dataset.dataset_split='test[:1%]'"
    "bench=MislabelingDetection train_dataset=mnist_train_unit_mislabeling train_dataset.dataset_split='train[:1%]'"
    "bench=SubclassDetection model=mnist_lenet_subclass train_dataset=mnist_train_unit_subclass train_dataset.dataset_split='train[:1%]' eval_dataset=mnist_test_unit_subclass eval_dataset.dataset_split='test[:1%]'"
    "bench=ShortcutDetection train_dataset=mnist_train_unit_shortcut train_dataset.dataset_split='train[:1%]' eval_dataset=mnist_test_unit_shortcut eval_dataset.dataset_split='test[:1%]'"
    "bench=MixedDatasets train_dataset=mnist_train_unit train_dataset.dataset_split='train[:1%]' eval_dataset=mnist_test_unit eval_dataset.dataset_split='test[:1%]' +adv_dataset=fashion_mnist_unit"
)

# Define the output directory
cfg_output_dir="tests/assets/mnist_test_suite_2"

# Get the current git commit tag
commit_tag=$(git rev-parse --short HEAD)

# Iterate over each parameter dictionary
for params in "${param_dicts[@]}"; do

    # Construct the output file name
    id="${commit_tag}-default"
    cfg_file_name="${id}"


    # Construct and execute the command with Hydra overrides
    echo "Running with parameters: $params"
    echo "Saving config to: $cfg_output_dir"
    python scripts/run.py $params hydra.run.dir="hydra_logs" id=$id +cfg_file_name=$cfg_file_name +cfg_output_dir=$cfg_output_dir bench_save_dir=$cfg_output_dir
    echo "Finished running with parameters: $params"
    echo "--------------------------------------"
done 