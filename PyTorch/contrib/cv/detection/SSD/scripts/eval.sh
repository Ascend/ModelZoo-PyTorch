#!/usr/bin/env bash
source scripts/npu_set_env.sh

rm -rf kernel_meta/
PORT=29500 ./tools/dist_test.sh configs/ssd/ssd300_coco_npu_8p.py work_dirs/ssd300_coco_npu_8p/latest.pth 8 --eval bbox