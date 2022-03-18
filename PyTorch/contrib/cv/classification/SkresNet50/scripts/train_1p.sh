#!/usr/bin/env bash

source scripts/npu_setenv.sh

RANK_ID_START=0
RANK_SIZE=1

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do

KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))

nohup \
taskset -c $PID_START-$PID_END python3.7.5 -u imagenet_fast.py "$@" \
  --data /opt/npu/imagenet \
  --epochs 100 \
  --schedule 30 60 90 \
  --wd 1e-4 \
  --gamma 0.1 \
  -c checkpoints/ \
  --lr 0.2 \
  --train-batch 256 \
  --opt-level O2 \
  --wd-all \
  --label-smoothing 0. \
  --warmup 0 \
  --device-list 0 \
  --loss-scale 16.0 \
  --rank 0 \
  --world-size 1\
  --local_rank $RANK_ID \
  --log-name 'train_1p.log'\
  -j $(($(nproc)/8)) &\
done
