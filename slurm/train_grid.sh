for lr in 0.1 0.01 0.001
do
  for scheduler in constant step
  do
    for opt in sgd adam
    do
      for weight_decay in 0.0 0.01 0.001
      do
        sbatch train.sh --dataset_type $1 --epochs 100 --save_each 10 --batch_size 32 --device cuda --optimizer $opt --lr $lr --scheduler $scheduler --weight_decay $weight_decay 
        sbatch train.sh --dataset_type $1 --epochs 100 --save_each 10 --batch_size 32 --device cuda --optimizer $opt --lr $lr --scheduler $scheduler --weight_decay $weight_decay --pretrained
      done
    done
  done
done
