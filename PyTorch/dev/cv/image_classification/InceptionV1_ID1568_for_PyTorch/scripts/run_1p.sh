#!/usr/bin/env bash
source scripts/npu_set_env.sh


currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_1p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

python3.7 ${currentDir}/../main.py \
    --data /opt/npu/dataset/imagenet \
    --npu 0 \
	-a inception_v3 \
    -b 128 \
    --lr 0.045 \
    --epochs 22 \
	-j 32 \
	-p 100 \
	--amp \
	--label-smoothing 0.1 \
 	--wd 0.0002  > ./inception_v3_train_1p.log 2>&1 &
