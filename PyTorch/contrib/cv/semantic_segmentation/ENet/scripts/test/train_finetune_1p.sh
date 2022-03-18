#!/bin/bash

source env_npu.sh

nohup python3 train.py \
        --model enet \
        --dataset citys \
        --lr 5e-4 \
        --weight-decay 2e-4 \
        --epochs 50 \
        --batch-size 4 \
        --amp \
        --aux \
        --resume &
