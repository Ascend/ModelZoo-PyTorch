#!/usr/bin/env bash
source scripts/set_npu_env.sh

nohup python3.7.5 -u train.py \
    --optimizer Adam \
    --epochs 1 \
    --batch_size 16 \
    --lr 1e-3 \
    --num_workers 4 > unet_1p.log 2>&1 &
