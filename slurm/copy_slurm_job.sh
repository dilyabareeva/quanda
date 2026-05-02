#!/bin/bash

SRC="bareeva@vca-gpu-0503-01:/data/cluster/users/bareeva/quanda_output_new2"
DST="/data2/bareeva/Projects/quanda/cluster_output_new2"

mkdir -p "$DST"
rsync -au "$SRC/" "$DST/"
rsync -au "$DST/" "$SRC/"



#before=$(find /data/cluster/users/bareeva/quanda_output_new2/eval_results -type f | wc -l); find /data/cluster/users/bareeva/quanda_output_new2/eval_results -type f -not -newermt 2026-04-29 -delete; after=$(find /data/cluster/users/bareeva/quanda_output_new2/eval_results -type f | wc -l); echo "Files before: $before"; echo "Files after:  $after"; echo "Deleted:      $((before - after))"; echo; echo "Remaining oldest files:"; find /data/cluster/users/bareeva/quanda_output_new2/eval_results -type f -printf '%TY-%Tm-%Td %TH:%TM  %p\n' | sort | head -3