#!/bin/bash

SRC="bareeva@vca-gpu-0503-01:/data/cluster/users/bareeva/quanda_output_new"
DST="/data2/bareeva/Projects/quanda/cluster_output_new"

mkdir -p "$DST"
rsync -a "$SRC/" "$DST/"


SRC_LOCAL="/data2/bareeva/Projects/quanda/cluster_output_new/eval_results"
DST_LOCAL="./local_eval_results"

mkdir -p "$DST_LOCAL"
rsync -a "$SRC_LOCAL/" "$DST_LOCAL/"