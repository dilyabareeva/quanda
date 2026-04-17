#!/bin/bash
# Interactive one-off run of a Python or Bash script inside the quanda container.
# Usage: ./slurm/run.sh path/to/script.{py,sh} [args...]
set -euo pipefail

case "$1" in
    *.sh) interpreter=bash ;;
    *)    interpreter=python ;;
esac

apptainer exec --nv \
    --bind "$(pwd):/workspace" \
    --bind /data/cluster/users/bareeva:/data/cluster/users/bareeva \
    --pwd /workspace \
    "$(dirname "$0")/env_quanda.sif" "$interpreter" "$@"
