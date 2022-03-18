#!/usr/bin/env bash
source scripts/npu_set_env.sh

ip=$(hostname -I |awk '{print $1}')
currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_8p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

python3.7 ${currentDir}/../main_8p.py \
	-a squeezenet1_1 \
	--amp \
        --data /opt/npu/dataset/imagenet \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49 \
        --workers=184 \
        --learning-rate=0.08 \
        --momentum=0.9 \
        --weight-decay=1.0e-04  \
        --print-freq=30 \
        --dist-url='tcp://127.0.0.1:50000' \
        --dist-backend='hccl' \
        --multiprocessing-distributed \
        --world-size=1 \
        --rank=0 \
        --device='npu' \
        --epochs=240 \
        --warm_up_epochs=5 \
        --batch-size=2048	> ./squeezenet1_1-npu_8p_eval.log 2>&1 &
