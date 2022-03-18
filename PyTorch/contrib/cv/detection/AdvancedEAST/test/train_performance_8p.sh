#!/bin/bash

source test/env.sh

RANK_ID_START=0
KERNEL_NUM=$(($(nproc)/8))
export WORLD_SIZE=8
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='29688'

if [ -n "$*" ]
then
    SIZES=$*
else
    SIZES="256 384 512 640 736"
fi

for SIZE in $SIZES
do
    for((RANK_ID=$RANK_ID_START;RANK_ID<$((WORLD_SIZE+RANK_ID_START));RANK_ID++));
    do
        PID_START=$((KERNEL_NUM*RANK_ID))
        PID_END=$((PID_START+KERNEL_NUM-1))
        if [ $RANK_ID == $((WORLD_SIZE+RANK_ID_START-1)) ]
        then
            nohup taskset -c $PID_START-$PID_END python3.7 -u train.py --size $SIZE --local_rank $RANK_ID --apex --epoch_num 3 --val_interval 1
        else
            nohup taskset -c $PID_START-$PID_END python3.7 -u train.py --size $SIZE --local_rank $RANK_ID --apex --epoch_num 3 --val_interval 1 &
        fi
    done
    sleep 5s
done
