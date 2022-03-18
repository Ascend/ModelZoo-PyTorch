#!/usr/bin/env bash

data_path_info=$1
data_path=`echo ${data_path_info#*=}`
if [[ $data_path == "" ]];then
    echo "[Warning] para \"data_path\" not set"
    echo "[Warning] use default data_path"
    data_path="/home/imagenet/"
fi

./distributed_train_npu.sh 8 ${data_path} --model vit_small_patch16_224 --sched cosine --epochs 100 --opt fusedadamw -j 16 --warmup-lr 1e-6 --mixup .1 --apex-amp --lr 1e-3 --weight-decay .05 --drop 0.1 --drop-path .1 -b 288 --device_num 8 --npu --combine_grad
