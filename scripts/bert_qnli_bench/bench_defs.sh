#!/bin/bash
# Benchmark definitions: dataset params and sweep hyperparams for QNLI / BERT.
# Source this file, then use: ${BENCH_PARAMS[Name]} and ${BENCH_SWEEP[Name]}

declare -A BENCH_PARAMS
declare -A BENCH_SWEEP


BENCH_PARAMS[ClassDetection]="train_dataset=qnli_train train_dataset.dataset_split='train' eval_dataset=qnli_test eval_dataset.dataset_split='validation' device=cuda:0 hydra.launcher.n_jobs=1"
BENCH_SWEEP[ClassDetection]=""

BENCH_PARAMS[MislabelingDetection]="train_dataset=qnli_train_mislabeling train_dataset.dataset_split='train' eval_dataset=qnli_test eval_dataset.dataset_split='validation' device=cuda:0 hydra.launcher.n_jobs=3"
BENCH_SWEEP[MislabelingDetection]=""

BENCH_PARAMS[LDS]="train_dataset=qnli_train train_dataset.dataset_split='train' eval_dataset=qnli_test eval_dataset.dataset_split='validation' device=cuda:0  hydra.launcher.n_jobs=3"
BENCH_SWEEP[LDS]=""

BENCH_PARAMS[MixedDatasets]="train_dataset=qnli_train train_dataset.dataset_split='train' eval_dataset=qnli_test_mixed_main eval_dataset.dataset_split='validation' +adv_dataset=mrpc +filter_by_prediction=true device=cuda:0 hydra.launcher.n_jobs=1"
BENCH_SWEEP[MixedDatasets]=""
