#!/bin/bash
# Build the quanda apptainer image.
# Run from the repository root so that pyproject.toml/README.md are in the
# build context for the %files section of env_quanda.def.
set -euo pipefail

cd "$(dirname "$0")/.."
apptainer build --force --fakeroot slurm/env_quanda.sif slurm/env_quanda.def
