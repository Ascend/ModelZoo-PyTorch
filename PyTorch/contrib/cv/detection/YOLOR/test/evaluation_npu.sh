#!/bin/bash
source ./test/env_npu.sh

python3 test.py \
    --data data/coco.yaml \
    --img 1280 \
    --batch 32 \
    --conf 0.001 \
    --iou 0.65 \
    --cfg cfg/yolor_p6.cfg \
    --weights './pretrained/npu8p_300.pt' \
    --name yolor_val_npu \
    --device npu \
    --npu 0 \
    2>&1 | tee evaluation_npu.log

