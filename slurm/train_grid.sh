#!/bin/bash

for augmentation in null crop crop_flip crop_flip_rotate crop_rotate flip flip_rotate rotate;
do
  for dataset_type in vanilla subclass mislabeled shortcut mixed;
  do
    for lr in 0.1 0.001 1e-4;
    do
      sbatch train_job.sh --dataset_name $1 \
                          --augmentation $augmentation \
                          --dataset_type $dataset_type \
                          --lr $lr \
                          --dataset_name $1 \
                          --adversarial_dir /mnt/quantus_metadata/$1/adversarial_dataset \
                          --seed 4242 \
                          --device "cuda" \
                          --module_name $2 \
                          --pretrained \
                          --batch_size 64 \
                          --save_each 10 \
                          --validate_each 5 \
                          --epochs 100
    done
  done
done
