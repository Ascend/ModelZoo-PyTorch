#!/usr/bin/env bash
source scripts/npu_set_env.sh

ip=$(hostname -I |awk '{print $1}')
currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_8p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

python3.7 ${currentDir}/../main-8p.py \
	-a inception_v3 \
	--amp \
	--loss-scale 128 \
        --data /opt/npu/dataset/imagenet \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49 \
        --workers=184 \
        --learning-rate=1.2 \
        --mom=0.9 \
        --weight-decay=1.0e-04  \
        --print-freq=30 \
        --dist-url='tcp://127.0.0.1:50000' \
        --dist-backend='hccl' \
        --multiprocessing-distributed \
        --world-size=1 \
        --rank=0 \
        --device='npu' \
        --epochs=250 \
	--label-smoothing=0.1 \
	--evaluate \
	--resume=/root/myxWorkSpace/inception_v3/result/training_8p_job_20210319172439/model_best_acc77.0487_epoch244.pth.tar \
        --batch-size=2048	> ./inception-v3-npu_8p.log 2>&1 &
