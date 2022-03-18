#!/usr/bin/env bash

source npu_set_env.sh

/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 1 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 2 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 3 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 4 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 5 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 6 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 7 -g error

/usr/local/Ascend/driver/tools/msnpureport -e disable

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_8p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"


nohup python3.7 ${currentDir}/mobilenetv2_8p_main_anycard.py \
    --addr=$(hostname -I |awk '{print $1}') \
    --seed 49  \
    --workers 128 \
    --lr 0.4 \
    --print-freq 1 \
    --eval-freq 1 \
    --dist-url 'tcp://127.0.0.1:50002' \
    --dist-backend 'hccl' \
    --multiprocessing-distributed \
    --world-size 1 \
    --class-nums 1000 \
    --batch-size 4096 \
    --epochs 600 \
    --rank 0 \
    --device-list '0,1,2,3,4,5,6,7' \
    --amp \
    --benchmark 0 \
    --data /home/imagenet > ${train_log_dir}/train_8p.log 2>&1 &

