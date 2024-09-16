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

python compute_explanations.py --method "similarity" --tiny_in_path "/data1/datapool" --panda_sketch_path "/home/bareeva/Projects/data_attribution_evaluation/assets/demo/sketch/" --output_dir "/home/bareeva/Projects/data_attribution_evaluation/assets/demo/" --checkpoints_dir "/home/bareeva/Projects/data_attribution_evaluation/assets/demo/" --metadata_dir "/home/bareeva/Projects/data_attribution_evaluation/assets/demo/"