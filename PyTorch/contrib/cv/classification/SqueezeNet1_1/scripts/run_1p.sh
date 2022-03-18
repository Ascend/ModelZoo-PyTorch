#!/usr/bin/env bash
source scripts/npu_set_env.sh

device_id=0

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_1p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

python3.7 ${currentDir}/../main.py \
        --data /opt/npu/dataset/imagenet \
        --npu 0 \
        -a squeezenet1_1 \
        -b 128 \
        -p 30 \
        --lr 0.01 \
        --epochs 22 \
        -j 32 \
        --amp \
        --wd 0.0001 > ./squeezenet_npu_1p.log 2>&1 &

