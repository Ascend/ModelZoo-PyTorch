#!/usr/bin/env bash
source npu_set_env.sh

/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_1p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"
ln -s ${currentDir}/.data ${train_log_dir}/.data

taskset -c 0-32 python3.7 ${currentDir}/gru_1p.py \
    --workers 32 \
    --dist-url 'tcp://127.0.0.1:50000' \
    --world-size 1 \
    --npu 0 \
    --batch-size 1536 \
    --epochs 10 \
    --rank 0 \
    --amp  > ./gru_1p.log 2>&1 &