#!/bin/bash
RANK_ID_START=0
RANK_SIZE=$1
OUTPUT_LOG=${OUTPUT_LOG:-"output.log"}
shift

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))
nohup taskset -c $PID_START-$PID_END python \
    train.py --local_rank $RANK_ID "$@" > ${OUTPUT_LOG} 2>&1 &
done
