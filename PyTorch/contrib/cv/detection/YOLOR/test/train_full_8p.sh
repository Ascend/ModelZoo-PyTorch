#!/bin/bash
source ./test/env_npu.sh

# addr is the ip of training server
if [ $(uname -m) = "aarch64" ]
then
	for i in $(seq 0 7)
	do 
	let p_start=0+24*i
	let p_end=23+24*i
	taskset -c $p_start-$p_end $CMD python3.7 train_mp.py \
        --cfg cfg/yolor_p6.cfg \
        --data data/coco.yaml \
        --addr 127.0.0.1 \
        --weights '' \
        --batch-size 64 \
        --img 1280 1280 \
        --local_rank $i \
        --device npu \
        --device-num 8 \
        --name yolor_p6_npu_8p_full \
        --hyp hyp.scratch.1280.yaml \
        --epochs 300 \
        --full \
        2>&1 | tee npu_8p_full.log &
	done
else
   python3.7 train.py \
        --cfg cfg/yolor_p6.cfg \
        --data data/coco.yaml \
        --addr 127.0.0.1 \
        --weights '' \
        --batch-size 64 \
        --img 1280 1280 \
        --local_rank 0 \
        --device npu \
        --device-num 8 \
        --name yolor_p6_npu_8p_full \
        --hyp hyp.scratch.1280.yaml \
        --epochs 300 \
        --full \
        2>&1 | tee npu_8p_full.log
fi
