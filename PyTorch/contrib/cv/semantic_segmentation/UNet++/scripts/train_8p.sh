#!/usr/bin/env python

source scripts/set_npu_env.sh

RANK_ID_START=0
RANK_SIZE=8

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do

KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))

nohup taskset -c $PID_START-$PID_END python -u train.py \
--batch_size 128 \
--optimizer Adam \
--lr 2e-4 \
--min_lr 1e-5 \
--seed_val 7 \
--loss-scale 128 \
--device npu \
--num_gpus 8 \
--num_workers 16 \
--rank_id $RANK_ID > unetpp_8p.log 2>&1 &
done
