#!/bin/bash
source ./test/env_npu.sh

taskset -c 0-23 python3 train.py \
    --cfg cfg/yolor_p6_finetune.cfg \
    --data data/coco.yaml \
    --weights 'pretrained/yolor_p6.pt' \
    --batch-size 8 \
    --img 1280 1280 \
    --device npu \
    --npu 0 \
    --name yolor_p6_npu_1p_finetune \
    --hyp hyp.scratch.1280.yaml \
    --epochs 1 \
    2>&1 | tee npu_1p_finetune.log
