#!/usr/bin/env bash
source /etc/profile
source scripts/set_npu_env.sh 
python3.7 ./main.py \
	/opt/npu/imagenet/ \
	--evaluate \
	--resume checkpoint.pth.tar \
	--addr=$(hostname -I |awk '{print $1}') \
	--seed=49 \
	--workers=128 \
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
	--epochs=120 \
	--amp \
	--batch-size=1024 > ./dpn131_8p_eval.log 2>&1