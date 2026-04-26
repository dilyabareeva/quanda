#!/bin/bash

SRC="bareeva@vca-gpu-0503-01:/home/fe/bareeva/Projects/quanda/logs"
DST="/home/bareeva/Projects/quanda/logs"

mkdir -p "$DST"
rsync -a "$SRC/" "$DST/"

