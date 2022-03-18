#!/usr/bin/env bash
source scripts/pt_set_env.sh 

currentDir=$(cd "$(dirname "$0")";pwd)/..
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_1p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"
python3.7 ${currentDir}/main.py \
	/opt/npu/imagenet/ \
	-a resnet152 \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49 \
        --workers=$(nproc) \
        --learning-rate=0.1 \
        --mom=0.9 \
        --weight-decay=1.0e-04  \
        --print-freq=1 \
        --device='npu' \
        --gpu=0 \
        --dist-backend='hccl' \
        --epochs=110 \
        --amp \
        --FusedSGD \
        --batch-size=256 > ./resnet101_1p.log 2>&1 &