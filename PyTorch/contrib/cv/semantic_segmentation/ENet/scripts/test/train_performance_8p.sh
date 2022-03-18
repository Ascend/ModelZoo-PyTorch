#!/bin/bash

source env_npu.sh

nohup python3 -u -m torch.distributed.launch --nproc_per_node=8 train.py \
        --model enet \
        --dataset citys \
        --lr 1.2e-3  \
        --weight-decay 2e-4 \
        --epochs 1 \
        --batch-size 4 \
        --aux \
        --amp \
        > train_performance_8p.log &

