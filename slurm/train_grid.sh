for lr in 0.1 0.01
do
  for scheduler in constant
  do
    for opt in adam
    do
      for weight_decay in 0.0
      do
        sbatch train.sh --dataset_name tiny_imagenet --dataset_type $1 --epochs 150 --validate_each 15 --save_each 2 --batch_size 64 --device cuda --optimizer $opt --lr $lr --scheduler $scheduler --weight_decay $weight_decay  --pretrained --augmentation flip_rotate
        sbatch train.sh --dataset_name tiny_imagenet --dataset_type $1 --epochs 150 --validate_each 15 --save_each 2 --batch_size 64 --device cuda --optimizer $opt --lr $lr --scheduler $scheduler --weight_decay $weight_decay --pretrained --augmentation flip
      done
    done
  done
done
