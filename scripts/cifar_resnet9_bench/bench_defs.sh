#!/bin/bash
# Benchmark definitions: dataset params and sweep hyperparams for CIFAR-10 / ResNet9.
# Source this file, then use: ${BENCH_PARAMS[Name]} and ${BENCH_SWEEP[Name]}

declare -A BENCH_PARAMS
declare -A BENCH_SWEEP


BENCH_PARAMS[ClassDetection]="train_dataset=cifar10_train train_dataset.dataset_split='train' eval_dataset=cifar10_test eval_dataset.dataset_split='test' +filter_by_prediction=true"
BENCH_SWEEP[ClassDetection]=""

BENCH_PARAMS[SubclassDetection]="model=cifar_resnet9_subclass train_dataset=cifar10_train_subclass train_dataset.dataset_split='train' eval_dataset=cifar10_test_subclass eval_dataset.dataset_split='test' val_dataset=cifar10_val_subclass +filter_by_prediction=true"
BENCH_SWEEP[SubclassDetection]=""

BENCH_PARAMS[MixedDatasets]="train_dataset=cifar10_train train_dataset.dataset_split='train' eval_dataset=cifar10_test_mixed_main eval_dataset.dataset_split='test' +adv_dataset=svhn +filter_by_prediction=true"
BENCH_SWEEP[MixedDatasets]=""

BENCH_PARAMS[ShortcutDetection]="train_dataset=cifar10_train_shortcut train_dataset.dataset_split='train' eval_dataset=cifar10_test_shortcut eval_dataset.dataset_split='test' +filter_by_shortcut_pred=true +filter_by_non_shortcut=true val_dataset=cifar10_val_shortcut"
BENCH_SWEEP[ShortcutDetection]="model.trainer.lr=0.4,0.1,0.01 model.trainer.optimizer=adam,sgd model.trainer.max_epochs=12,24 train_dataset.wrapper.metadata.p=0.75,0.9,0.95 hydra.sweeper.n_trials=27"

BENCH_PARAMS[MislabelingDetection]="train_dataset=cifar10_train_mislabeling train_dataset.dataset_split='train' eval_dataset=cifar10_test eval_dataset.dataset_split='test'"
BENCH_SWEEP[MislabelingDetection]="model.trainer.lr=0.4,0.1,0.01 model.trainer.optimizer=adam,sgd model.trainer.max_epochs=12,24 train_dataset.wrapper.metadata.p=0.1,0.2 hydra.sweeper.n_trials=24"

BENCH_PARAMS[LDS]="train_dataset=cifar10_train train_dataset.dataset_split='train' eval_dataset=cifar10_test eval_dataset.dataset_split='test'"
BENCH_SWEEP[LDS]=""
