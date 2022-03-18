#!/usr/bin/env bash
source env_npu.sh

/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error
export RANK_SIZE=1
currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_1p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

taskset -c 0-24 python3.7 ${currentDir}/densenet121_1p_main.py \
        --workers 40 \
        --arch densenet121 \
        --npu 0 \
        --lr 0.1 \
        --momentum 0.9 \
        --amp \
        --print-freq 1 \
        --eval-freq 5 \
        --batch-size 256 \
        --epochs 90 \
        --data /data/imagenet/ > ./densenet121_1p.log 2>&1 &