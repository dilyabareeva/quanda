#!/bin/bash
# Drop into an interactive shell inside the quanda container.
set -euo pipefail

# visible device: cuda:2
export CUDA_VISIBLE_DEVICES=3

apptainer shell --nv \
    --env HF_HOME=/data/cluster/users/bareeva/.hf_cache \
    --bind "$(pwd):/workspace" \
    --bind /data/cluster/users/bareeva:/data/cluster/users/bareeva \
    --pwd /workspace \
    "$(dirname "$0")/env_quanda2.sif"
