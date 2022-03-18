#!/usr/bin/env bash
source scripts/pt_set_env.sh 

device_id_list=0,1,2,3,4,5,6,7
currentDir=$(cd "$(dirname "$0")";pwd)/..
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_8p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"
python3.7 ${currentDir}/main.py \
	/opt/npu/imagenet/ \
	-a resnet152 \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49 \
        --workers=$(nproc) \
        --learning-rate=0.5 \
        --mom=0.9 \
        --weight-decay=1.0e-04  \
        --print-freq=1 \
        --dist-url='tcp://127.0.0.1:50000' \
        --multiprocessing-distributed \
        --world-size=1 \
        --rank=0 \
        --device='npu' \
        --dist-backend='hccl' \
	      --epochs=110 \
        --batch-size=2048 \
	      --amp \
        --device_list=${device_id_list} \
        --FusedSGD \
	      --loss-scale=1024 >  ./resnet101_8p.log 2>&1 &