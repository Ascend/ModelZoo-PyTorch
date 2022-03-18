#!/usr/bin/env bash
source scripts/pt.sh
device_id=0

/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_1p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

python3.7 ${currentDir}/../main.py \
        --data /opt/npu/dataset/imagenet \
        --npu ${device_id} \
	-a googlenet \
        -b 512 \
        --lr 0.01 \
        --epochs 1 \
	-j 32 \
	--amp \
 	--wd 0.0001 > ./goolenet_1p.log 2>&1 &
