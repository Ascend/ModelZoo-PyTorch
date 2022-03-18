#!/usr/bin/env bash
source /etc/profile
source ./scripts/set_npu_env.sh
python3.7.5 ./main.py \
        /opt/npu/imagenet/ \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49 \
        --workers=64 \
        --learning-rate=0.8 \
        --mom=0.9 \
        --weight-decay=1.0e-04  \
        --print-freq=1 \
        --dist-url='tcp://127.0.0.1:50001' \
        --multiprocessing-distributed \
        --world-size=1 \
        --rank=0 \
        --device='npu' \
        --dist-backend 'hccl' \
        --epochs=100 \
        --amp \
        --batch-size=2048 > ./train_8p.log 2>&1