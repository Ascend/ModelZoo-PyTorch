#!/bin/bash
export COMBINED_ENABLE=0
nohup python3 main.py \
	--nodes 1\
        --gpus 8\
        --model "RDN" \
        --data_path "data" \
        --distributed \
        --train_file "data/DIV2K_x2.h5" \
        --eval_file "data/Set5_x2.h5" \
        --scale 2 \
        --num-features 64 \
        --growth-rate 64 \
        --num-blocks 16 \
        --num-layers 8 \
        --lr 5e-4 \
        --batch-size 64 \
        --patch-size 32 \
        --epochs 800 \
        --workers 16 \
        --apex \
        --device_id 0 \
        --apex-opt-level O1\
        --loss_scale_value 32\
        --weight-decay 1e-4\
        --seed 123\
        --print-freq 1 &
