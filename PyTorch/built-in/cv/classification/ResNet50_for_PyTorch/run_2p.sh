#!/usr/bin/env bash
source env_npu.sh
export WHICH_OP=GEOP
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1

ip=$(hostname -I |awk '{print $1}')
device_id_list=0,1

/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_2p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

python3.7 ${currentDir}/DistributedResnet50/main_apex_d76_npu.py \
        --data /data/imagenet \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49 \
        --workers=128 \
        --learning-rate=0.4 \
        --warmup=8 \
        --label-smoothing=0.1 \
        --mom=0.9 \
        --weight-decay=1.0e-04 \
        --static-loss-scale=128 \
        --print-freq=1 \
        --dist-url='tcp://127.0.0.1:50000' \
        --dist-backend='hccl' \
        --multiprocessing-distributed \
        --world-size=1 \
        --rank=0 \
        --device-list=${device_id_list} \
        --benchmark=0 \
        --device='npu' \
        --epochs=90 \
        --num_classes=1000 \
        --batch-size=1024 > ./resnet50_2p.log 2>&1 &



