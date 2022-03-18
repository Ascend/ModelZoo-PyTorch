#!/bin/bash
source env_npu.sh
currentDir=$(cd "$(dirname "$0")";pwd)/..

nohup python3 ${currentDir}/main.py \
        --gpus 8\
        --distributed \
        --lr 0.0008 \
        --batch_size 128 \
        --n_epochs 200 \
        --workers 0 \
        --apex \
        --device_id 0 \
        --test_only 1 &
