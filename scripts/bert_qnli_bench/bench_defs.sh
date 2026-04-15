#!/bin/bash
# Benchmark definitions: dataset params and sweep hyperparams for QNLI / BERT.
# Source this file, then use: ${BENCH_PARAMS[Name]} and ${BENCH_SWEEP[Name]}

declare -A BENCH_PARAMS
declare -A BENCH_SWEEP


BENCH_PARAMS[ClassDetection]="train_dataset=qnli_train train_dataset.dataset_split='train' eval_dataset=qnli_test eval_dataset.dataset_split='validation' +filter_by_prediction=true device=cuda:0 hydra.launcher.n_jobs=1"
BENCH_SWEEP[ClassDetection]=""

BENCH_PARAMS[MislabelingDetection]="train_dataset=qnli_train_mislabeling train_dataset.dataset_split='train' eval_dataset=qnli_test eval_dataset.dataset_split='validation' device=cuda:1 hydra.launcher.n_jobs=3"
BENCH_SWEEP[MislabelingDetection]="model.trainer.max_epochs=50,100,200 train_dataset.wrapper.metadata.p=0.1,0.2 hydra.sweeper.n_trials=6 device=cuda:1"

BENCH_PARAMS[LDS]="train_dataset=qnli_train train_dataset.dataset_split='train' eval_dataset=qnli_test eval_dataset.dataset_split='validation' device=cuda:0  hydra.launcher.n_jobs=1"
BENCH_SWEEP[LDS]=""
