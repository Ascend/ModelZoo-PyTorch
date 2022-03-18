#!/usr/bin/env bash
source scripts/pt.sh

device_id_list=0,1,2,3,4,5,6,7

currentDir=$(cd "$(dirname "$0")/..";pwd)
currtime=`date +%Y%m%d%H%M%S`

python3.7 ${currentDir}/main.py \
    --addr=$(hostname -I |awk '{print $1}') \
    --seed=49 \
    --workers=184 \
    --learning-rate=0.04 \
    --print-freq=1 \
    --eval-freq=1 \
    --dist-url 'tcp://127.0.0.1:50002' \
    --multiprocessing-distributed \
    --world-size 1 \
    --batch-size 1024 \
    --device 'npu' \
    --epochs 90 \
    --rank 0 \
    --device-list '0,1,2,3,4,5,6,7' \
    --amp \
    --opt-level 'O1' \
    --dist-backend 'hccl' \
    --loss-scale-value 8 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --data=/opt/npu/dataset/imagenet >output_8p.log
