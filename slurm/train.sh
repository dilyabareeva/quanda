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

# The next line is optional and for job statistics only. You may omit it if you do not need statistics.
#echo "$PWD/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg

echo "Extract tiny-imagegetnet-200.tar to ${LOCAL_JOB_DIR}"
time unzip -qq $DATAPOOL3/datasets/tiny-imagenet-200.zip -d $LOCAL_JOB_DIR

# Make a folder locally on the node for job_results. This folder ensures that data is copied back even when the job fails
mkdir -p "${LOCAL_JOB_DIR}/outputs"

# List of methods
methods=("similarity" "representer_points" "tracincpfast" "arnoldi" "trak" "random")

# Select the method based on the SLURM_ARRAY_TASK_ID
method=${methods[$SLURM_ARRAY_TASK_ID]}

apptainer run --nv  --env "PYTHONPATH=." \
    --bind /data/datapool3/datasets:/mnt/dataset \
    --bind ${LOCAL_JOB_DIR}/outputs:/mnt/output \
    --bind /data/datapool3/datasets/quanda_metadata:/mnt/metadata\
     ../singularity/train.sif \
    --method "$method" \
    --tiny_imgnet_path "/mnt/dataset" \
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

# This command copies all results generated in $LOCAL_JOB_DIR back to the submit folder regarding the job id.
cd "$LOCAL_JOB_DIR"
tar -cf quanda_train_${SLURM_JOB_ID}.tar outputs
cp quanda_train_${SLURM_JOB_ID}.tar $SLURM_SUBMIT_DIR/outputs
rm -rf ${LOCAL_JOB_DIR}/outputs
