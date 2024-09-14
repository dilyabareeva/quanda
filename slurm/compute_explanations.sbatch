#!/bin/bash
#SBATCH --output=%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --array=0-9
#SBATCH --job-name=quanda_explanations

MODEL_ID=$SLURM_ARRAY_TASK_ID

python compute_explanations.py --model_id $MODEL_ID