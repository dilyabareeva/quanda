#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=train_quanda
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

source "/etc/slurm/local_job_dir.sh"

mkdir -p ${LOCAL_JOB_DIR}/outputs

jobname= $1
shift

apptainer run --nv \
            --bind /data/datapool3/datasets/quanda_metadata:/mnt/quanda_metadata \
            --bind ${LOCAL_JOB_DIR}/outputs:/mnt/outputs \
            ../singularity/train.sif \
            --metadata_root /mnt/quanda_metadata \
            --dataset_cache_    dir /mnt/quanda_metadata/hf_cache \
            --output_path /mnt/outputs \
            "$@"

tar -czf quanda_train_${SLURM_JOB_ID}.tgz outputs
cp train_$jobname_${SLURM_JOB_ID}.tgz ${SLURM_SUBMIT_DIR}
