#!/usr/bin/env bash
source scripts/npu_set_env.sh

rm -rf kernel_meta/
PORT=29500 tools/dist_train.sh configs/ssd/ssd300_coco_npu_8p.py 8