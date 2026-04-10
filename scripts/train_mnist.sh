#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(dirname $(dirname $(realpath $0)))"

bench_types=(
    #"LDS"
    #"MislabelingDetection"
    #"ClassDetection"
    #"SubclassDetection"
    "ShortcutDetection"
    #"MixedDatasets"
)
bench_params=(
    #"train_dataset=mnist_train train_dataset.dataset_split='train' eval_dataset=mnist_test eval_dataset.dataset_split='test'"
    #"train_dataset=mnist_train_mislabeling train_dataset.dataset_split='train' eval_dataset=mnist_test eval_dataset.dataset_split='test'"
    #"train_dataset=mnist_train train_dataset.dataset_split='train' eval_dataset=mnist_test eval_dataset.dataset_split='test'"
    #"model=mnist_lenet_subclass train_dataset=mnist_train_subclass train_dataset.dataset_split='train' eval_dataset=mnist_test_subclass eval_dataset.dataset_split='test' val_dataset=mnist_val_subclass"
    "train_dataset=mnist_train_shortcut train_dataset.dataset_split='train' eval_dataset=mnist_test_shortcut eval_dataset.dataset_split='test' +filter_by_shortcut_pred=true +filter_by_non_shortcut=true val_dataset=mnist_val_shortcut "
    #"train_dataset=mnist_train train_dataset.dataset_split='train' eval_dataset=mnist_test_mixed_main eval_dataset.dataset_split='test' +adv_dataset=fashion_mnist"
)

sweep_params=(
    #""
    #"model.trainer.max_epochs=5,20,50"
    #""
    #""
    "model.trainer.max_epochs=100,500 train_dataset.wrapper.metadata.p=0.5,0.75,0.9"
    #""
)

# Define the output directory
cfg_output_dir="quanda/benchmarks/resources/configs"

# Get the current git commit tag
commit_tag=$(git rev-parse --short HEAD)

# Iterate over each parameter dictionary
for i in "${!bench_types[@]}"; do
    bench="${bench_types[$i]}"
    params="${bench_params[$i]}"
    sweep="${sweep_params[$i]}"

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
    python scripts/train.py bench="$bench" $params $sweep id=$id +cfg_output_dir=$cfg_output_dir +cfg_file_name=$cfg_file_name --multirun
    # Saving the results to a config file
    python scripts/opt_results_to_cfg.py bench="$bench" $params id=$id +cfg_output_dir=$cfg_output_dir +cfg_file_name=$cfg_file_name
    # Training the model
    python scripts/train_and_push_to_hub.py --config-name $cfg_file_name --config-dir $cfg_output_dir
    echo "Finished running with parameters: $params"
    echo "--------------------------------------"
done 