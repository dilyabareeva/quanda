#!/bin/bash

SRC="bareeva@vca-gpu-0503-01:/data/cluster/users/bareeva/quanda_output_new2"
DST="/data2/bareeva/Projects/quanda/cluster_output_new2"

mkdir -p "$DST"
rsync -au "$SRC/" "$DST/"


SRC_LOCAL="/data2/bareeva/Projects/quanda/cluster_output_new2/eval_results"
DST_LOCAL="./local_eval_results"

mkdir -p "$DST_LOCAL"
rsync -au "$SRC_LOCAL/" "$DST_LOCAL/"