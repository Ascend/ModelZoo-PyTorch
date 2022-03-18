#!/usr/bin/env bash
source npu_set_env.sh

/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 4 -g error

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_8p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

python3.7 ${currentDir}/examples/imagenet/main.py \
    --data=/data/imagenet \
    --arch=efficientnet-b0 \
    --batch-size=4096 \
    --lr=1.6 \
    --momentum=0.9 \
    --epochs=100 \
    --autoaug \
    --amp \
    --pm=O1 \
    --loss_scale=32 \
    --val_feq=10 \
    --addr=$(hostname -I |awk '{print $1}') \
    --dist-backend=hccl \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --device_list '0,1,2,3,4,5,6,7' > ${train_log_dir}/train_8p.log 2>&1 &