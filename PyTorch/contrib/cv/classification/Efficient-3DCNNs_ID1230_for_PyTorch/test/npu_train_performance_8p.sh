#!/usr/bin/env bash
RANK_ID_START=0
RANK_SIZE=8
root_path=$1

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do

KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))

nohup taskset -c $PID_START-$PID_END python3.7 ../main.py --local_rank $RANK_ID   \
    --root_path ${root_path} \
    --gpu_or_npu npu \
    --use_prof 0 \
    --use_apex 1 \
    --device_lists 0,1,2,3,4,5,6,7 \
    --distributed 1 \
    --n_classes 101 \
    --n_finetune_classes 101 \
    --learning_rate 0.04 \
    --droupout_rate 0.2 \
    --n_epochs 2 \
    --batch_size 640 \
    --n_threads 64 \
	  --ft_portion complete \
	  > ${root_path}/results/npu_train_performance_8p.log 2>&1 &
done