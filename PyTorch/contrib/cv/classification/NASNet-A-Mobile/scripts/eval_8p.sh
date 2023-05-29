#!/usr/bin/env bash
source ./scripts/env_npu.sh
python3 ./main_npu_8p.py \
	    /opt/npu/imagenet/ \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49 \
        --evaluate \
        --resume checkpoint.pth.tar \
        --workers=64 \
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
        --epochs=100 \
        --amp \
        --batch-size=2048 > ./log/nasnet-a-mobile_8p_eval.log 2>&1 &
