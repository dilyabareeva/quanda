for lr in 0.1
do
  for scheduler in constant step
  do
    for opt in adam
    do
      for weight_decay in 0.0 0.1
      do
        for augmentation in flip flip_rotate
        do
          sbatch train.sh --dataset_name tiny_imagenet --dataset_type mislabeled --epochs 150 --validate_each 15 --save_each 2 --batch_size 64 --device cuda --optimizer $opt --lr $lr --scheduler $scheduler --weight_decay $weight_decay  --pretrained --augmentation $augmentation
        done
      done
    done
  done
done

for lr in 0.1
do
  for scheduler in constant step
  do
    for opt in adam
    do
      for weight_decay in 0.0 0.1
      do
        for augmentation in flip flip_rotate
        do
          sbatch train.sh --dataset_name tiny_imagenet --dataset_type mixed --epochs 150 --validate_each 15 --save_each 2 --batch_size 64 --device cuda --optimizer $opt --lr $lr --scheduler $scheduler --weight_decay $weight_decay  --pretrained --augmentation $augmentation
        done
      done
    done
  done
done

for lr in 0.1
do
  for scheduler in constant step
  do
    for opt in adam
    do
      for weight_decay in 0.0 0.1
      do
        for augmentation in flip flip_rotate
        do
          sbatch train.sh --dataset_name tiny_imagenet --dataset_type shortcut --epochs 150 --validate_each 15 --save_each 2 --batch_size 64 --device cuda --optimizer $opt --lr $lr --scheduler $scheduler --weight_decay $weight_decay  --pretrained --augmentation $augmentation
        done
      done
    done
  done
done