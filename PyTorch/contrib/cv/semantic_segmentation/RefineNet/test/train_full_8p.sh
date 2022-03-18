#!/bin/bash

RANK_ID_START=0
RANK_SIZE=8
rm -f nohup.out

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do

KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))

nohup \
taskset -c $PID_START-$PID_END python3 src/train.py \
            --enc-backbone 101 \
            --num-stages 2 \
            --num-classes 21 \
            --train-dir './VOC' \
            --val-dir './VOC' \
            --dataset-type 'torchvision' \
            --stage-names 'SBD' 'VOC' \
            --epochs-per-stage 100 100 \
            --augmentations-type 'albumentations' \
            --train-batch-size 8 8 \
            --ckpt-dir 'model/refinenet_101_O2_b8_npu_8P' \
            --device-list '0,1,2,3,4,5,6,7' \
            --device-type 'npu' \
	        --local-rank $RANK_ID 2>&1 | tee log/refinenet_101_O2_b8_npu_8P_ampsgd_1024.log & 
done