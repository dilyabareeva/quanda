#!/bin/bash
# Interactive one-off run of a Python script inside the quanda container.
# Usage: ./slurm/run.sh path/to/script.py [args...]
set -euo pipefail

BINDS=(--bind "$(pwd):/workspace")
[ -n "${DATAPOOL3:-}" ] && BINDS+=(--bind "${DATAPOOL3}/datasets:/datasets")

apptainer exec --nv "${BINDS[@]}" --pwd /workspace \
    "$(dirname "$0")/env_quanda.sif" python "$@"
