#!/usr/bin/env bash

data_path_info=$1
data_path=`echo ${data_path_info#*=}`
if [[ $data_path == "" ]];then
    echo "[Warning] para \"data_path\" not set"
    exit 1
fi


weight_info=$2
weight=`echo ${weight_info#*=}`
if [[ $weight == "" ]];then
    echo "[Warning] para \"weight\" not set"
    exit 1
fi

python3 \
    main.py --local_rank 0 --npu --data_path ${data_path} --weight ${weight} --finetune
