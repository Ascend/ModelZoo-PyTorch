#!/usr/bin/env bash
source scripts/set_npu_env.sh
python3 ./main.py \
	/opt/npu/imagenet/ \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49 \
        --evaluate \
        --resume checkpoint.pth.tar \
        --workers=64 \
        --learning-rate=0.4 \
        --mom=0.9 \
        --weight-decay=1.0e-04  \
        --print-freq=1 \
        --dist-url='tcp://127.0.0.1:50001' \
        --dist-backend 'hccl' \
        --multiprocessing-distributed \
        --world-size=1 \
        --rank=0 \
        --device='npu' \
        --epochs=90\
        --amp \
        --batch-size=1024 > ./resnext_101_32x8d_8p_eval.log 2>&1