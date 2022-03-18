#!/usr/bin/env bash
source env_npu.sh

python3.7 fine_tune_new.py \
    --seed 12345 \
    --amp_cfg \
    --opt_level O2 \
    --loss_scale_value 1024 \
    --device_list '0' \
    --batch_size 512 \
    --epochs 8 \
    --epochs_per_save 1 \
    --lr 0.001 \
    --workers 8 \
    --data_dir '/home/VGG-Face2/data/train_cropped'
