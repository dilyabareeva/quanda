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

jobname="$1_$2_$3_$4_$5"
echo ${jobname}_${SLURM_JOB_ID}

apptainer run --nv \
            --bind /data/datapool3/datasets/quanda_metadata:/mnt/quanda_metadata \
            --bind /data/datapool3/datasets/quanda_metadata/hf_cache:/mnt/quanda_metadata/hf_cache \
            --bind ${LOCAL_JOB_DIR}/outputs:/mnt/outputs \
            ../singularity/train.sif \
            --metadata_root /mnt/quanda_metadata \
            --dataset_cache_dir /mnt/quanda_metadata/hf_cache \
            --output_path /mnt/outputs \
            --dataset_name $1 \
            --module_name $2 \
            --dataset_type $3 \
            --augmentation $4 \
            --lr $5 \
            --adversarial_dir /mnt/quanda_metadata/$1/adversarial_dataset \
            --seed 42 \
            --device "cuda" \
            --pretrained \
            --batch_size 64 \
            --save_each 5 \
            --validate_each 5 \
            --epochs 100


cd ${LOCAL_JOB_DIR}
tar -czf ${jobname}_${SLURM_JOB_ID}.tgz outputs
cp ${jobname}_${SLURM_JOB_ID}.tgz ${SLURM_SUBMIT_DIR}
