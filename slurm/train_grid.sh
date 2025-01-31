#!/bin/bash

for augmentation in null crop crop_flip crop_flip_rotate crop_rotate flip flip_rotate rotate;
do
  for dataset_type in vanilla subclass mislabeled shortcut mixed;
  do
    for lr in 0.1 0.001 1e-4;
    do
      sbatch train_job.sh $1 $2 $dataset_type $augmentation $lr {$1}_{$dataset_type}_{$augmentation}_{$lr}
    done
  done
done
