#!/usr/bin/env bash
source scripts/npu_set_env.sh

rm -rf kernel_meta/
python3 tools/test.py configs/ssd/ssd300_coco_npu.py work_dirs/ssd300_coco_npu_8p/latest.pth --eval bbox