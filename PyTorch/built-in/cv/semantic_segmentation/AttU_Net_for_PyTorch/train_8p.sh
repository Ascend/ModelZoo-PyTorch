#!/bin/bash
source env_npu.sh
rm -rf ./models

export RANK_SIZE=8

for((RANK_ID=0;RANK_ID<8;RANK_ID++));
do
    export RANK_ID=$RANK_ID
    python3.7 main.py  \
        --model_type AttU_Net \
        --data_path /home/dataset \
        --num_epochs 150 \
        --distributed \
        --npu_idx $RANK_ID\
        --batch_size 128 \
        --num_worker 196 \
        --seed 12345 \
        --apex \
        --apex_opt_level O2 \
        --loss_scale_value 1024 \
        --result_path ./result_8p/ \
        --lr 0.0016 > ./train_8p.log 2>&1 &
done