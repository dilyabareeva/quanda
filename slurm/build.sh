#!/bin/bash
# Build the quanda apptainer image.

set -euo pipefail

cd "$(dirname "$0")/.."
apptainer build --force --fakeroot slurm/env_quanda.sif slurm/env_quanda.def
