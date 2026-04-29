#!/bin/bash
# Benchmark definitions: dataset params and sweep hyperparams for AwA2 / ResNet50.
# Source this file, then use: ${BENCH_PARAMS[Name]} and ${BENCH_SWEEP[Name]}

declare -A BENCH_PARAMS
declare -A BENCH_SWEEP


BENCH_PARAMS[ClassDetection]="train_dataset=awa2_train train_dataset.dataset_split='train' eval_dataset=awa2_test +filter_by_prediction=true device=cuda:0"
BENCH_SWEEP[ClassDetection]=""

BENCH_PARAMS[SubclassDetection]="model=awa2_resnet50_subclass train_dataset=awa2_train_subclass train_dataset.dataset_split='train' eval_dataset=awa2_test_subclass val_dataset=awa2_val_subclass +filter_by_prediction=true device=cuda:0"
BENCH_SWEEP[SubclassDetection]=""

BENCH_PARAMS[MixedDatasets]="train_dataset=awa2_train_mixed train_dataset.dataset_split='train' eval_dataset=awa2_test_mixed_main +adv_dataset=imagenet_sketch +filter_by_prediction=true device=cuda:0"
BENCH_SWEEP[MixedDatasets]="model.trainer.max_epochs=30,60 splits.imagenet_sketch.ratios.train=0.003,0.01,0.02 hydra.sweeper.n_trials=24"

BENCH_PARAMS[ShortcutDetection]="train_dataset=awa2_train_shortcut train_dataset.dataset_split='train' eval_dataset=awa2_test_shortcut +filter_by_shortcut_pred=true +filter_by_non_shortcut=true val_dataset=awa2_val_shortcut device=cuda:0"
BENCH_SWEEP[ShortcutDetection]="model.trainer.max_epochs=30,60 train_dataset.wrapper.metadata.p=0.8,0.9 hydra.sweeper.n_trials=36"

BENCH_PARAMS[MislabelingDetection]="train_dataset=awa2_train_mislabeling train_dataset.dataset_split='train' eval_dataset=awa2_test device=cuda:0"
BENCH_SWEEP[MislabelingDetection]="model.trainer.max_epochs=30,60 train_dataset.wrapper.metadata.p=0.1,0.2 hydra.sweeper.n_trials=24"

BENCH_PARAMS[LDS]="train_dataset=awa2_train train_dataset.dataset_split='train' eval_dataset=awa2_test device=cuda:0"
BENCH_SWEEP[LDS]=""
