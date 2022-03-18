#!/usr/bin/env bash
source scripts/env_npu.sh

device_id_list=0,1,2,3,4,5,6,7
export RANK_SIZE=8
currentDir=$(cd "$(dirname "$0")";pwd)/..
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_8p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

KERNEL_NUM=$(($(nproc)/8))
for i in $(seq 0 7)
do 
if [ $(uname -m) = "aarch64" ]
then
    PID_START=$((KERNEL_NUM * i))
    PID_END=$((PID_START + KERNEL_NUM - 1))
    taskset -c $PID_START-$PID_END python3.7 -u ${currentDir}/8p_main_med.py \
        --data=/data/imagenet \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49  \
        --workers=$(nproc) \
        --learning-rate=4 \
        --print-freq=1 \
        --eval-freq=5 \
        --arch=shufflenet_v2_x1_0  \
        --dist-url='tcp://127.0.0.1:50000' \
        --dist-backend='hccl' \
        --multiprocessing-distributed \
        --world-size=1 \
        --batch-size=8192 \
        --epochs=240 \
        --warm_up_epochs=5 \
        --device_num=8 \
        --rank=0 \
        --amp \
        --momentum=0 \
        --device-list=${device_id_list} \
        --num-classes=1000 \
        --local_rank=$i \
        --benchmark 0 > ./shufflenetv2_8p.log 2>&1 &
else
    python3.7 -u ${currentDir}/8p_main_med.py \
        --data=/data/imagenet \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49  \
        --workers=$(nproc) \
        --learning-rate=4 \
        --print-freq=1 \
        --eval-freq=5 \
        --arch=shufflenet_v2_x1_0  \
        --dist-url='tcp://127.0.0.1:50000' \
        --dist-backend='hccl' \
        --multiprocessing-distributed \
        --world-size=1 \
        --batch-size=8192 \
        --epochs=240 \
        --warm_up_epochs=5 \
        --device_num=8 \
        --rank=0 \
        --amp \
        --momentum=0 \
        --device-list=${device_id_list} \
        --num-classes=1000 \
        --local_rank=$i \
        --benchmark 0 > ./shufflenetv2_8p.log 2>&1 &
fi
done