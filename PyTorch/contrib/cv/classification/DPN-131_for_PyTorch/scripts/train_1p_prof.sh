#!/usr/bin/env bash
source scripts/set_npu_env.sh 
python3.7 ./main.py \
	/opt/npu/imagenet/ \
	--addr=$(hostname -I |awk '{print $1}') \
	--seed=49 \
	--workers=64 \
	--learning-rate=0.05 \
	--mom=0.9 \
	--weight-decay=1.0e-04  \
	--print-freq=1 \
	--dist-url='tcp://127.0.0.1:50001' \
	--dist-backend 'hccl' \
	--world-size=1 \
	--device='npu' \
	--epochs=1 \
	--amp \
    --gpu=0 \
    --prof \
	--batch-size=128 > ./dpn131_8p.log 2>&1