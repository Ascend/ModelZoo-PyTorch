#!/usr/bin/env bash
source scripts/set_npu_env.sh 
python3 ./main.py \
	/opt/npu/imagenet/ \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49 \
        --workers=$(nproc) \
        --learning-rate=0.1 \
        --mom=0.9 \
        --weight-decay=1.0e-04  \
        --print-freq=1 \
        --dist-url='tcp://127.0.0.1:50000' \
        --dist-backend 'hccl' \
        --device='npu' \
        --gpu=2 \
        --epochs=1\
        --world-size=1 \
        --amp \
        --batch-size=128 > ./resnext101_32x8d_1p.log 2>&1