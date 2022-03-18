#!/bin/bash
source env_npu.sh
currentDir=$(cd "$(dirname "$0")";pwd)/..

nohup python3 ${currentDir}/main.py \
        --gpus 1\
        --lr 0.0002 \
        --batch_size 64 \
        --n_epochs 200 \
        --workers 0 \
        --apex \
        --device_id 0 &
