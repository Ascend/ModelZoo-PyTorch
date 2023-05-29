#!/usr/bin/env bash
source scripts/npu_set_env.sh

rm -rf kernel_meta/
python3 tools/train.py configs/ssd/ssd300_coco_npu.py