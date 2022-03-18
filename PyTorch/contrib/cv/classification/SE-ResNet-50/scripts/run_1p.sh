#!/usr/bin/env bash
source ./set_npu_env.sh
python3.7.5 ./main.py \
        /opt/npu/imagenet/ \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49 \
        --workers=64 \
        --learning-rate=0.1 \
        --mom=0.9 \
        --weight-decay=1.0e-04  \
        --print-freq=1 \
        --dist-backend 'hccl' \
        --rank=0 \
        --world-size=1 \
        --device='npu' \
        --epochs=1 \
        --gpu=3 \
        --amp \
        --batch-size=128 > ./senet_1p.log 2>&1
