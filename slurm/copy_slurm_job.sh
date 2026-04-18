#!/bin/bash

#SBATCH --job-name=quanda-copy
#SBATCH --output=log/%j_%x.out
#SBATCH --error=log/%j_%x.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0

# Variant of slurm_job.sbatch that stages work in node-local scratch and
# copies artefacts (cache, eval results, logs) back to the submission dir
# on completion. Useful for long benchmark sweeps where /workspace is slow.
#
# Usage: sbatch copy_slurm_job.sh <script-to-run-inside-container>

source "/etc/slurm/local_job_dir.sh"

COMMAND="${1:-echo 'No command specified. Pass a script path as argument to sbatch.'}"

mkdir -p log "${LOCAL_JOB_DIR}/cache" "${LOCAL_JOB_DIR}/eval_results"

apptainer exec --nv \
    --env HF_HOME=/data/cluster/users/bareeva/.hf_cache \
    --bind ${PWD}:/workspace \
    --bind ${LOCAL_JOB_DIR}:/scratch \
    --bind /data/cluster/users/bareeva:/data/cluster/users/bareeva \
    --pwd /workspace \
    $PWD/env_quanda.sif \
    bash "$COMMAND"

# Copy results back from node-local scratch to the submission directory.
mkdir -p "$SLURM_SUBMIT_DIR/eval_results" "$SLURM_SUBMIT_DIR/cache"
cp -r "${LOCAL_JOB_DIR}/eval_results/." "$SLURM_SUBMIT_DIR/eval_results/" 2>/dev/null || true
cp -r "${LOCAL_JOB_DIR}/cache/." "$SLURM_SUBMIT_DIR/cache/" 2>/dev/null || true
