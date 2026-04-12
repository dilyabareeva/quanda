#!/bin/bash
# Benchmark definitions: dataset params and sweep hyperparams.
# Source this file, then use: ${BENCH_PARAMS[Name]} and ${BENCH_SWEEP[Name]}

declare -A BENCH_PARAMS
declare -A BENCH_SWEEP


BENCH_PARAMS[ClassDetection]="train_dataset=mnist_train train_dataset.dataset_split='train' eval_dataset=mnist_test eval_dataset.dataset_split='test' +filter_by_prediction=true"
BENCH_SWEEP[ClassDetection]=""

BENCH_PARAMS[SubclassDetection]="model=mnist_lenet_subclass train_dataset=mnist_train_subclass train_dataset.dataset_split='train' eval_dataset=mnist_test_subclass eval_dataset.dataset_split='test' val_dataset=mnist_val_subclass +filter_by_prediction=true"
BENCH_SWEEP[SubclassDetection]=""

BENCH_PARAMS[MixedDatasets]="train_dataset=mnist_train train_dataset.dataset_split='train' eval_dataset=mnist_test_mixed_main eval_dataset.dataset_split='test' +adv_dataset=fashion_mnist +filter_by_prediction=true"
BENCH_SWEEP[MixedDatasets]=""

BENCH_PARAMS[ShortcutDetection]="train_dataset=mnist_train_shortcut train_dataset.dataset_split='train' eval_dataset=mnist_test_shortcut eval_dataset.dataset_split='test' +filter_by_shortcut_pred=true +filter_by_non_shortcut=true val_dataset=mnist_val_shortcut"
BENCH_SWEEP[ShortcutDetection]="model.trainer.lr=0.01,0.001,0.0001 model.trainer.optimizer=adam,sgd model.trainer.max_epochs=5,50,100 train_dataset.wrapper.metadata.p=0.75,0.9,0.95 hydra.sweeper.n_trials=27"

BENCH_PARAMS[MislabelingDetection]="train_dataset=mnist_train_mislabeling train_dataset.dataset_split='train' eval_dataset=mnist_test eval_dataset.dataset_split='test'"
BENCH_SWEEP[MislabelingDetection]="model.trainer.lr=0.001,0.0001 model.trainer.optimizer=adam,sgd model.trainer.max_epochs=50,100,200 train_dataset.wrapper.metadata.p=0.1,0.2 hydra.sweeper.n_trials=24"

BENCH_PARAMS[LDS]="train_dataset=mnist_train train_dataset.dataset_split='train' eval_dataset=mnist_test eval_dataset.dataset_split='test'"
BENCH_SWEEP[LDS]=""
