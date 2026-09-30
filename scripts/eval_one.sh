#!/bin/bash
# Run one benchmark x explainer eval (SLURM job body, single GPU).
# Usage: eval_one.sh <bench_dir> <config_name> <bench> <method> [--regenerate-explanations]
cd "$(dirname "$0")/.."
source "scripts/$1/eval_defs.sh"
EVAL_CONFIG_NAME=$2; benchmarks=("$3"); methods=("$4")
EXPL_SWEEP[$4]+=" device=cuda:0"
source scripts/eval.sh "${@:5}"
