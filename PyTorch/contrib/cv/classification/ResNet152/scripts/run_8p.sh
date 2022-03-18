#!/usr/bin/env bash
source scripts/set_npu_env.sh 
python3.7.5 ./main.py \
	/opt/npu/imagenet/ \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49 \
        --workers=128 \
        --learning-rate=0.8 \
        --mom=0.9 \
        --weight-decay=1.0e-04  \
        --print-freq=1 \
        --dist-url='tcp://127.0.0.1:50001' \
        --dist-backend 'hccl' \
        --multiprocessing-distributed \
        --world-size=1 \
        --rank=0 \
        --device='npu' \
        --epochs=140\
        --amp \
        --batch-size=2048 > ./resnet152_8p.log 2>&1