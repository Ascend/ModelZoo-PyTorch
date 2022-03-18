#!/usr/bin/env bash
source env.sh

python3.7 main_train.py \
    --backbone res101 \
    --opt_level O1 \
    --dataset UCAS_AOD \
    --train_path ${UCAS_AOD_PATH}/train.txt \
    --test_path ${UCAS_AOD_PATH}/test.txt \
    --root_path datasets/evaluate \
    --training_size "800,1344" \
    --distributed 1 \
    --device_list "0,1,2,3,4,5,6,7" \
    --npus_per_node 8 \
    --manual_seed 0 \
    --epochs 100 \
    --batch_size 64 \
    --n_threads 64 \
    --lr0 0.0001 \
    --warmup_lr 0.00001 \
    --warm_epoch 5
