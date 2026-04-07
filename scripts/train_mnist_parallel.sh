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
    "train_dataset=mnist_train train_dataset.dataset_split='train' eval_dataset=mnist_test eval_dataset.dataset_split='test'"
    "train_dataset=mnist_train_mislabeling train_dataset.dataset_split='train' eval_dataset=mnist_test eval_dataset.dataset_split='test'"
    "train_dataset=mnist_train train_dataset.dataset_split='train' eval_dataset=mnist_test eval_dataset.dataset_split='test'"
    "model=mnist_lenet_subclass train_dataset=mnist_train_subclass train_dataset.dataset_split='train' eval_dataset=mnist_test_subclass eval_dataset.dataset_split='test' val_dataset=mnist_val_subclass"
    "train_dataset=mnist_train_shortcut train_dataset.dataset_split='train' eval_dataset=mnist_test_shortcut eval_dataset.dataset_split='test' +filter_by_shortcut_pred=true +filter_by_non_shortcut=true model.trainer.max_epochs=5 val_dataset=mnist_val_shortcut"
    "train_dataset=mnist_train train_dataset.dataset_split='train' eval_dataset=mnist_test_mixed_main eval_dataset.dataset_split='test' +adv_dataset=fashion_mnist"
)

cfg_output_dir="quanda/benchmarks/resources/configs"
commit_tag=$(git rev-parse --short HEAD)

mkdir -p logs

for i in "${!bench_types[@]}"; do
    bench="${bench_types[$i]}"
    params="${bench_params[$i]}"
    id="${commit_tag}-default_${bench}"

    bash scripts/train_mnist_single.sh "$bench" "$params" "$id" "$cfg_output_dir" > "logs/${bench}.log" 2>&1 &
    echo "Launched $bench (PID $!, log: logs/${bench}.log)"
done

wait
echo "All benchmarks finished."
