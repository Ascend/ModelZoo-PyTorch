#!/usr/bin/env bash

data_path_info=$1
data_path=`echo ${data_path_info#*=}`
if [[ $data_path == "" ]];then
    echo "[Warning] para \"data_path\" not set"
    echo "[Warning] use default data_path"
    data_path="../data/Market-1501/"
fi

RANK_ID_START=0
RANK_SIZE=1

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))
nohup taskset -c $PID_START-$PID_END python3.7 \
    main.py --local_rank $RANK_ID --device_num 1 --npu --lr 8e-4 --mode evaluate --weight "weights/model_500.pt" --data_path ${data_path} &
done
