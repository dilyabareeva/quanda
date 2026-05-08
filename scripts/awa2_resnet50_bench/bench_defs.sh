#!/bin/bash

declare -A BENCH_PARAMS
declare -A BENCH_SWEEP


BENCH_PARAMS[ClassDetection]="train_dataset=awa2_train train_dataset.dataset_split='train' eval_dataset=awa2_test device=cuda:1"
BENCH_SWEEP[ClassDetection]=""

BENCH_PARAMS[SubclassDetection]="model=awa2_resnet50_subclass train_dataset=awa2_train_subclass train_dataset.dataset_split='train' eval_dataset=awa2_test_subclass val_dataset=awa2_val_subclass device=cuda:1"
BENCH_SWEEP[SubclassDetection]=""

BENCH_PARAMS[MixedDatasets]="train_dataset=awa2_train_mixed train_dataset.dataset_split='train' eval_dataset=awa2_test_mixed_main +adv_dataset=imagenet_sketch device=cuda:0"
BENCH_SWEEP[MixedDatasets]="model.trainer.lr=0.1,0.2 model.trainer.max_epochs=30,96 splits.imagenet_sketch.ratios.train=0.01 hydra.sweeper.n_trials=24"

BENCH_PARAMS[ShortcutDetection]="train_dataset=awa2_train_shortcut train_dataset.dataset_split='train' eval_dataset=awa2_test_shortcut +filter_by_shortcut_pred=true val_dataset=awa2_val_shortcut device=cuda:0"
BENCH_SWEEP[ShortcutDetection]="model.trainer.lr=0.1,0.2 model.trainer.max_epochs=64 train_dataset.wrapper.metadata.p=0.85 hydra.sweeper.n_trials=36"

BENCH_PARAMS[MislabelingDetection]="train_dataset=awa2_train_mislabeling train_dataset.dataset_split='train' eval_dataset=awa2_test device=cuda:1"
BENCH_SWEEP[MislabelingDetection]="model.trainer.max_epochs=64,96 hydra.sweeper.n_trials=24"

BENCH_PARAMS[LDS]="train_dataset=awa2_train train_dataset.dataset_split='train' eval_dataset=awa2_test device=cuda:1"
BENCH_SWEEP[LDS]=""
