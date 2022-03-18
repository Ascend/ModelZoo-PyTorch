#!/bin/bash

python3 src/train.py \
    --enc-backbone 101 \
    --num-stages 1 \
    --num-classes 21 \
    --train-dir './VOC' \
    --val-dir './VOC' \
    --dataset-type 'torchvision' \
    --stage-names 'SBD' 'VOC' \
    --epochs-per-stage 3 3 \
    --augmentations-type 'albumentations' \
    --train-batch-size 16 16 \
    --ckpt-dir 'model/refinenet_101_O2_b16_npu_1P_performance' \
    --device-list '0' \
    --device-type 'npu' \
    2>&1 | tee log/refinenet_101_O2_b16_npu_1P_performance.log # save log
