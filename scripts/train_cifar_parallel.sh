#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(dirname $(dirname $(realpath $0)))"

bench_types=(
    #"ClassDetection"
    "MislabelingDetection"
    #"SubclassDetection"
    "ShortcutDetection"
    #"MixedDatasets"
    #"LDS"
)
bench_params=(
    #"train_dataset=cifar10_train train_dataset.dataset_split='train' eval_dataset=cifar10_test eval_dataset.dataset_split='test'"
    "train_dataset=cifar10_train_mislabeling train_dataset.dataset_split='train' eval_dataset=cifar10_test eval_dataset.dataset_split='test'"
    #"model=cifar_resnet9_subclass train_dataset=cifar10_train_subclass train_dataset.dataset_split='train' eval_dataset=cifar10_test_subclass eval_dataset.dataset_split='test' val_dataset=cifar10_val_subclass"
    "train_dataset=cifar10_train_shortcut train_dataset.dataset_split='train' eval_dataset=cifar10_test_shortcut eval_dataset.dataset_split='test' +filter_by_shortcut_pred=true +filter_by_non_shortcut=true val_dataset=cifar10_val_shortcut"
    #"train_dataset=cifar10_train train_dataset.dataset_split='train' eval_dataset=cifar10_test_mixed_main eval_dataset.dataset_split='test' +adv_dataset=svhn"
    #"train_dataset=cifar10_train train_dataset.dataset_split='train' eval_dataset=cifar10_test eval_dataset.dataset_split='test'"
)

sweep_params=(
    #""
    "model.trainer.max_epochs=50,100 train_dataset.wrapper.metadata.p=0.2"
    #""
    "model.trainer.max_epochs=50,100 train_dataset.wrapper.metadata.p=0.75"
    #""
    #""
)


cfg_output_dir="quanda/benchmarks/resources/configs"
commit_tag=$(git rev-parse --short HEAD)

mkdir -p logs

for i in "${!bench_types[@]}"; do
    bench="${bench_types[$i]}"
    params="${bench_params[$i]}"
    sweep="${sweep_params[$i]}"
    id="${commit_tag}-default_${bench}"

    bash scripts/train_cifar_single.sh "$bench" "$params" "$sweep" "$id" "$cfg_output_dir" > "logs/${bench}.log" 2>&1 &
    echo "Launched $bench (PID $!, log: logs/${bench}.log)"
done

wait
echo "All benchmarks finished."
