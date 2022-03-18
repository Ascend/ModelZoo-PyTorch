#!/bin/bash
source env_npu.sh
rm -rf ./models
python3.7 main.py  \
    --model_type AttU_Net \
    --data_path /home/dataset \
    --num_epochs 150 \
    --npu_idx 1\
    --batch_size 16 \
    --num_worker 32 \
    --seed 12345 \
    --apex \
    --apex_opt_level O2 \
    --loss_scale_value 1024 \
    --result_path ./result_1p/ \
    --lr 0.0002 > ./train_1p.log 2>&1 &