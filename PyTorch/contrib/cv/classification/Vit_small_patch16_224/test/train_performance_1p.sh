#!/usr/bin/env bash

data_path_info=$1
data_path=`echo ${data_path_info#*=}`
if [[ $data_path == "" ]];then
    echo "[Warning] para \"data_path\" not set"
    echo "[Warning] use default data_path"
    data_path="/home/imagenet/"
fi

./distributed_train_npu.sh 1 ${data_path} --model vit_small_patch16_224 --sched cosine --epochs 1 --opt fusedadamw -j 8 --mixup .1 --apex-amp --lr 5e-4 --weight-decay .05 --drop 0.1 --drop-path .1 -b 288 --device_num 1 --npu --combine_grad --warmup-epochs 0
