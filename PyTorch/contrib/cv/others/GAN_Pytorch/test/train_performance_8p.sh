#!/bin/bash
source env_npu.sh
currentDir=$(cd "$(dirname "$0")";pwd)/..

nohup python3 ${currentDir}/main.py \
        --gpus 8\
        --distributed \
        --lr 0.0008 \
        --batch_size 128 \
        --n_epochs 1 \
        --workers 16 \
        --apex \
        --device_id 0 &
