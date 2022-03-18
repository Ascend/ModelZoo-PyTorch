#!/usr/bin/env bash

source ./env.sh
taskset -c 0-19 python3.7 trainval_net_8p.py \
    --net=res101 \
    --bs=32 \
    --lr=0.008 \
    --lr_decay_step=8 \
    --device=npu \
    --npu_id="npu:1" \
    --amp \
    --opt_level=O1 \
    --loss_scale=1024.0 \
    --epochs=20 \
    --disp_interval=1 \
    --nw=0
