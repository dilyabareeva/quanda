#!/bin/bash
#SBATCH --output=%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --array=0-0
#SBATCH --job-name=train_tin

source "/etc/slurm/local_job_dir.sh"

# Make a folder locally on the node for job_results. This folder ensures that data is copied back even when the job fails
mkdir -p "${LOCAL_JOB_DIR}/outputs"

start=`date +%s`

apptainer run --nv  --env "PYTHONPATH=." \
    --bind /data/datapool3/datasets:/mnt/dataset \
    --bind ${LOCAL_JOB_DIR}/outputs:/mnt/output \
    --bind /data/datapool3/datasets/quanda_metadata:/mnt/metadata\
     ../singularity/train.sif \
    --dataset_path "/mnt/dataset" \
    --metadata_path "/mnt/metadata" \
    --output_dir "/mnt/output" \
    --device "cuda" \
    "$@"

    # "--dataset_type",
    # "--download",
    # "--pretrained",
    # "--epochs",
    # "--lr",
    # "--batch_size",
    # "--save_each",
    # "--optimizer"


end=`date +%s`

runtime=$((end-start))
echo "Runtime: $runtime"
# This command copies all results generated in $LOCAL_JOB_DIR back to the submit folder regarding the job id.
cd "$LOCAL_JOB_DIR"
tar -czf train_${SLURM_JOB_ID}.tgz outputs
cp train_${SLURM_JOB_ID}.tgz $SLURM_SUBMIT_DIR
rm -rf ${LOCAL_JOB_DIR}/train_${SLURM_JOB_ID}.tgz