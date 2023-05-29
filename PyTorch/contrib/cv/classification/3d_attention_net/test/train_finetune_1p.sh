#!/usr/bin/env bash

source test/env_npu.sh



RANK_ID_START=0
RANK_SIZE=1


for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do

KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))

taskset -c $PID_START-$PID_END python3 train.py \
    --device_type="NPU" \
    --device_id=$RANK_ID \
    --device_num=$RANK_SIZE \
    --is_train="True" \
    --is_pretrain="True" \
    --num_classes=11 \
    --total_epochs=300 \
    --dist_url='tcp://127.0.0.1:49876' \
    --train_batch_size=512 \
    --test_batch_size=128 &
done



