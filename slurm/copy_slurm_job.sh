#!/bin/bash

SRC="bareeva@vca-gpu-0503-01:/data/cluster/users/bareeva/quanda_output"
DST="/data2/bareeva/Projects/quanda/cluster_output"

mkdir -p "$DST"
rsync -a "$SRC/" "$DST/"
