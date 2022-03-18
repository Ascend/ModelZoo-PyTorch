#!/usr/bin/env bash
NPUS=${1-8}
MODEL=${2-./work_dirs/fsaf_r50_fpn_1x_coco/latest.pth}

source ../env.sh
nohup ./tools/dist_test.sh ./configs/fsaf/fsaf_r50_fpn_1x_coco.py $MODEL $NPUS --eval bbox > ./eval.log 2>&1 &