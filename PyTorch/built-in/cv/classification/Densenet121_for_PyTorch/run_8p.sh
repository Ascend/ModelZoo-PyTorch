#!/usr/bin/env bash
source env_npu.sh

/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 4 -g error

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_8p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

if [ $(uname -m) = "aarch64" ]
then
    KERNEL_NUM=$(($(nproc)/8))
    for i in $(seq 0 7)
    do
    PID_START=$((KERNEL_NUM * i))
    PID_END=$((PID_START + KERNEL_NUM - 1))
    taskset -c $PID_START-$PID_END python3.7 ${currentDir}/densenet121_8p_main.py \
        --addr=$(hostname -I|awk '{print $1}') \
        --seed 49 \
        --workers 160 \
        --arch densenet121 \
        --lr 0.8 \
        --print-freq 1 \
        --eval-freq 5 \
        --batch-size 2048 \
        --epochs 90 \
        --dist-url 'tcp://127.0.0.1:50000' \
        --dist-backend 'hccl' \
        --multiprocessing-distributed \
        --world-size 1 \
        --rank 0 \
        --gpu $i \
        --device-list '0,1,2,3,4,5,6,7' \
        --amp \
        --benchmark 0 \
        --data /data/imagenet/ > ./densenet121_8p.log 2>&1 &
    done
else
    for i in $(seq 0 7)
    do
    python3.7 ${currentDir}/densenet121_8p_main.py \
        --addr=$(hostname -I|awk '{print $1}') \
        --seed 49 \
        --workers 160 \
        --arch densenet121 \
        --lr 0.8 \
        --print-freq 1 \
        --eval-freq 5 \
        --batch-size 2048 \
        --epochs 90 \
        --dist-url 'tcp://127.0.0.1:50000' \
        --dist-backend 'hccl' \
        --multiprocessing-distributed \
        --world-size 1 \
        --rank 0 \
        --gpu $i \
        --device-list '0,1,2,3,4,5,6,7' \
        --amp \
        --benchmark 0 \
        --data /data/imagenet/ > ./densenet121_8p.log 2>&1 &
    done
fi