#!/bin/bash
source env.sh
rm -rf ./models
nohup python3 main.py  \
        --model_type R2U_Net \
        --data_path ./dataset \
        --batch_size 16 \
        --lr 0.0002 \
        --num_workers 32 \
        --apex 1 \
        --apex-opt-level O2 \
        --loss_scale_value 1024 \
        --npu_idx 1\
        --num_epochs 150 &
