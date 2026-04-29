#!/bin/bash
# Train a single benchmark. Hydra n_jobs is capped by MAX_PARALLEL
# (default 6 — sized for one 40GB GPU at batch_size=64; lower if you
# hit OOM, raise on a bigger GPU). The sweep cardinality (n_trials) is
# whatever bench_defs.sh sets per benchmark; optuna runs them in
# batches of MAX_PARALLEL.
#
# Usage: train_awa2_per_bench.sh BENCH_NAME [extra train.sh args]
# Env:   MAX_PARALLEL — override the parallelism cap.

source "$(dirname "$0")/bench_defs.sh"

CONFIG_NAME="awa2_resnet50"
CONFIG_MAP_PREFIX="awa2"

BENCH="$1"
shift

MAX_PARALLEL=${MAX_PARALLEL:-4}

if [ -z "${BENCH_PARAMS[$BENCH]+x}" ]; then
    echo "Error: unknown benchmark '$BENCH'" >&2
    exit 1
fi

BENCH_SWEEP[$BENCH]="${BENCH_SWEEP[$BENCH]} hydra.launcher.n_jobs=${MAX_PARALLEL} hydra.sweeper.n_jobs=${MAX_PARALLEL}"

benchmarks=("$BENCH")

source "$(dirname "$0")/../train.sh" "$@"
