#!/usr/bin/env bash

source npu_set_env.sh

/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error
/usr/local/Ascend/driver/tools/msnpureport -e disable

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_1p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"


python3.7 ${currentDir}/mobilenetv2_8p_main_anycard.py \
    --addr=$(hostname -I |awk '{print $1}') \
    --seed 49  \
    --workers 128 \
    --lr 0.05 \
    --print-freq 1 \
    --eval-freq 1 \
    --dist-url 'tcp://127.0.0.1:50002' \
    --dist-backend 'hccl' \
    --multiprocessing-distributed \
    --world-size 1 \
    --class-nums 1000 \
    --batch-size 512 \
    --epochs 600 \
    --rank 0 \
    --device-list '0' \
    --amp \
    --benchmark 0 \
    --data /home/imagenet > ${train_log_dir}/train_1p.log 2>&1 &
