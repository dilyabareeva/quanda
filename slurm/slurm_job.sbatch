#!/bin/bash

#SBATCH --job-name=quanda
#SBATCH --output=log/%j_%x.out
#SBATCH --error=log/%j_%x.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --partition=gpu

# Hydra surfaces full tracebacks for scripts under scripts/*.
export HYDRA_FULL_ERROR=1
mkdir -p log

apptainer exec --nv \
      $PWD/slurm/env_quanda.sif \
      bash "$@"
