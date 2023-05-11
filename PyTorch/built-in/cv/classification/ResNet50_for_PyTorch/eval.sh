#!/usr/bin/env bash
source env_npu.sh
export WHICH_OP=GEOP
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=1

device_id=0

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_1p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"
python3 ${currentDir}/pytorch_resnet50_apex.py \
        --data /data/imagenet \
        --npu ${device_id} \
        -j64 \
        -b512 \
        --lr 0.2 \
        --warmup 5 \
        --label-smoothing=0.1 \
        --epochs 90 \
        --num_classes=1000 \
        --evaluate=True \
        --resume checkpoint.pth.tar \
        --optimizer-batch-size 512 > ./resnet50_1p.log 2>&1 &

