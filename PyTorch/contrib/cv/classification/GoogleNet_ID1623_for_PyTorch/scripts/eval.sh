#!/usr/bin/env bash
source scripts/pt.sh

/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 4 -g error

ip=$(hostname -I |awk '{print $1}')
currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_8p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

python3.7 ${currentDir}/../main-8p.py \
	-a googlenet \
	--amp \
        --data /opt/npu/imagenet \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49 \
        --workers=184 \
        --learning-rate=0.5 \
        --mom=0.9 \
        --weight-decay=1.0e-04  \
        --print-freq=30 \
        --dist-url='tcp://127.0.0.1:50000' \
        --dist-backend='hccl' \
        --multiprocessing-distributed \
        --world-size=1 \
        --rank=0 \
        --benchmark=0 \
        --device='npu' \
        --epochs=150 \
	--evaluate \
        --resume /root/myxWorkSpace/googlenet/result/training_8p_job_20210304101618/model_best_acc69.8074_epoch139.pth.tar \
        --batch-size=4096	> ./googlenet_npu_evaluate_8p.log 2>&1 &
