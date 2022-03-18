#!/bin/bash
source ../../test/env_npu.sh

python3 train.py  \
        --model densenet169 \
        --epochs 90 \
        --apex \
        --apex-opt-level O2 \
        --device_id 5 \
        --data-path=/home/ImageNet2012 \
        --batch-size 128 \
        --workers 16 \
        --lr 0.1 \
        --momentum 0.9 \
        --loss_scale_value 1024 \
        --weight-decay 1e-4 \
        --print-freq 1 |& tee train_1p.log
