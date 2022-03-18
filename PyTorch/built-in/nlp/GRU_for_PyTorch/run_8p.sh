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

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_8p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"
ln -s ${currentDir}/.data ${train_log_dir}/.data

if [ $(uname -m) = "aarch64" ]
then
    KERNEL_NUM=$(($(nproc)/8))
    for i in $(seq 0 7)
    do
    PID_START=$((KERNEL_NUM * i))
    PID_END=$((PID_START + KERNEL_NUM - 1))
    taskset -c $PID_START-$PID_END python3.7 ${currentDir}/gru_8p.py \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed 123456 \
        --workers 20 \
        --print-freq 1 \
        --dist-url 'tcp://127.0.0.1:50000' \
        --dist-backend 'hccl' \
        --multiprocessing-distributed \
        --world-size 1 \
        --batch-size 12288 \
        --epoch 10 \
        --rank 0 \
        --npu $i \
        --device-list '0,1,2,3,4,5,6,7' \
        --amp  > ./gru_8p_${i}.log 2>&1 &
    done
else
    for i in $(seq 0 7)
    do
    python3.7 ${currentDir}/gru_8p.py \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed 123456 \
        --workers 80 \
        --print-freq 1 \
        --dist-url 'tcp://127.0.0.1:50000' \
        --dist-backend 'hccl' \
        --multiprocessing-distributed \
        --world-size 1 \
        --batch-size 12288 \
        --epoch 10 \
        --rank 0 \
        --npu $i \
        --device-list '0,1,2,3,4,5,6,7' \
        --amp  > ./gru_8p.log 2>&1 &
    done
fi