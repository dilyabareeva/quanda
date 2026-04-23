#!/bin/bash

#SBATCH --job-name=quanda
#SBATCH --output=log/%j_%x.out
#SBATCH --error=log/%j_%x.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --partition=gpu3,gpu5

# Hydra surfaces full tracebacks for scripts under scripts/*.
export HYDRA_FULL_ERROR=1

COMMAND="${1:-echo 'No command specified. Pass a script path as argument to sbatch.'}"

mkdir -p log

apptainer exec --nv \
      --env HF_HOME=/data/cluster/users/bareeva/.hf_cache \
      --bind ${PWD}:/workspace \
      --bind /data/cluster/users/bareeva:/data/cluster/users/bareeva \
      --pwd /workspace \
      $PWD/slurm/env_quanda.sif \
      bash "$COMMAND"
