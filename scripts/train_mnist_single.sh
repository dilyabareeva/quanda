#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(dirname $(dirname $(realpath $0)))"

bench=$1
params=$2
sweep=$3
id=$4
cfg_output_dir=$5

python scripts/generate_config.py hydra.run.dir="hydra_logs" $params id=$id +cfg_file_name=$id +cfg_output_dir=$cfg_output_dir
python scripts/train.py bench="$bench" $params $sweep id=$id +cfg_output_dir=$cfg_output_dir +cfg_file_name=$id --multirun
python scripts/opt_results_to_cfg.py bench="$bench" $params id=$id +cfg_output_dir=$cfg_output_dir +cfg_file_name=$id
python scripts/train_and_push_to_hub.py --config-name $id --config-dir $cfg_output_dir
