#!/usr/bin/env bash
source scripts/npu_set_env.sh

nohup python3.7.5 -u main.py \
    --data /home/dataset/imagenet \
    --workers 24 \
    --epoch 1 \
    --batch-size=256 \
    --lr 0.1 \
    --momentum 0.9 \
    --wd 1e-4 \
    --device npu \
    --gpu 0 \
    --amp \
    --opt-level "O2" \
    --eval-freq 1 \
    --loss-scale-value 16 > seresnext50_32x4d_1p.log 2>&1 &
