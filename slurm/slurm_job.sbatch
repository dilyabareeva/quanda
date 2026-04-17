#!/bin/bash

#SBATCH --job-name=quanda
#SBATCH --output=log/%j_%x.out
#SBATCH --error=log/%j_%x.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --partition=gpu3,gpu5

# Hydra surfaces full tracebacks for scripts under scripts/*.
export HYDRA_FULL_ERROR=1

COMMAND="${1:-echo 'No command specified. Pass a script path as argument to sbatch.'}"

mkdir -p log

apptainer exec --nv \
      --bind ${PWD}:/workspace \
      --bind /data/cluster/users/bareeva:/data/cluster/users/bareeva \
      --pwd /workspace \
      $PWD/slurm/env_quanda2.sif \
      bash "$COMMAND"
